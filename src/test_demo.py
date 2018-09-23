from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import os.path
import sys
import time
import glob
import numpy as np
from six.moves import xrange
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

from config import *
from imdb import  kitti
from utils.util import *
from nets import *

import os

import rospy
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

path = os.getcwd()

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
        'checkpoint', path[:-4]+'/data/best/model.ckpt-49999',
        """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
        'input_path', path[:-4]+'/data/samples/*',
        """Input lidar scan to be detected. Can process glob input such as """
        """./data/samples/*.npy or single input.""")
tf.app.flags.DEFINE_string(
        'out_dir', path[:-4]+'/data/samples_out/', """Directory to dump output.""")
tf.app.flags.DEFINE_string('gpu', '2', """gpu id.""")

class Ros_tensor():
    def __init__(self):
        self.input_lidar = None
        self.mc = kitti_squeezeSeg_config()
        self.mc.LOAD_PRETRAINED_MODEL = False
        self.mc.BATCH_SIZE = 1 # TODO(bichen): fix this hard-coded batch size.
        self.model = SqueezeSeg(self.mc)
        rospy.init_node('rostensorflow')
        rospy.Rate(10)
        self.header = std_msgs.msg.Header()
        self.header.stamp = rospy.Time.now()
        self.header.frame_id = 'velody'
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

        self._saver = tf.train.Saver(self.model.model_params)
        self._session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self._saver.restore(self._session, FLAGS.checkpoint)

        self._sub =rospy.Subscriber('/kitti/velo/pointcloud',PointCloud2,self.callback,queue_size=1000)
        self.pub = rospy.Publisher('/points_raw',PointCloud2,queue_size=1000)
        self.pub_car =rospy.Publisher("/points_raw1",Marker,queue_size=1000)
        self.pub_per =rospy.Publisher("/points_raw2",Marker,queue_size=1000)
        self.pub_cyc=rospy.Publisher("/points_raw3",Marker,queue_size=1000)
    
        #self.detect()
        
    def callback(self,data):
        lidar_ros=pc2.read_points(data)
        points = []
        for point in lidar_ros:
            points.append(point)
        self.input_lidar=self.generate_data(np.array(points))

        #with tf.Graph().as_default():
          #for f in glob.iglob(FLAGS.input_path):
            #lidar = np.load(f).astype(np.float32, copy=False)[:, :, :5]
            #topic kitti/velo/pointcloud
            
             
        print(self.input_lidar.shape)
        lidar = self.input_lidar
        start =time.time()

        lidar_mask = np.reshape(
                (lidar[:, :, 4] > 0),
                [self.mc.ZENITH_LEVEL, self.mc.AZIMUTH_LEVEL, 1]
            )
        lidar_nor = (lidar - self.mc.INPUT_MEAN)/self.mc.INPUT_STD
      
        pred_cls = self._session.run(
                self.model.pred_cls,
                feed_dict={
                    self.model.lidar_input:[lidar_nor],
                    self.model.keep_prob: 1.0,
                    self.model.lidar_mask:[lidar_mask]
                }
            )
        print('time',time.time()-start)
    
            # save the data
            #print(pred_cls[0].shape)
            #plt.imshow(pred_cls[0])
        lidar_xyz = lidar[:,:,:3]
            #print(lidar_xyz.shape)
        lidar_raw = lidar_xyz.reshape(-1,3)
        print (lidar_raw.shape)
                
        Mask = pred_cls[0].reshape(-1,1)
        print(Mask[:,0].shape)
        id_0 = np.argwhere(Mask[:,0]==1)
        id_1 = np.argwhere(Mask[:,0]==2)
        id_2 = np.argwhere(Mask[:,0]==3)
            
        car= []
        for index in id_0:
              car.append(lidar_raw[index,:])
        car = np.array(car)
        self.car = car.reshape(-1,3)
            
        pedestrian=[]
        for index in id_1:
              pedestrian.append(lidar_raw[index,:])
        pedestrian = np.array(pedestrian)
        self.pedestrian = pedestrian.reshape(-1,3)
            
        cyc=[]
        for index in id_2:
              cyc.append(lidar_raw[index,:])
        cyc = np.array(cyc)
        self.cyc = cyc.reshape(-1,3)
            
            
        self.publish_pc(lidar_raw,self.car,self.pedestrian,self.cyc)
            
            
        
    def generate_data(self,points_ros): 
        start =time.time()

        filtered_lidar = self.filter_camera_angle(points_ros)
        cal_dis = self.cal_distance(filtered_lidar)
        sphere_lidar =  self.tansform_data_projection(cal_dis,64,512)
        print('create input',time.time()-start)
        return sphere_lidar

    def publish_pc(self,pc,car,person,cyc):
    
        
        points = pc2.create_cloud_xyz32(self.header,pc)
        '''
        pub2 = rospy.Publisher('/point_seg',PointCloud2,queue_size=10000)
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'velody'
        points2 = pc2.create_cloud_xyz32(header,obj)
        '''
        marker_car =self.print_car(self.pub_car,car)    
        marker_per =self.print_person(self.pub_per,person)    
        marker_cyc =self.print_cyc(self.pub_cyc,cyc)    
        self.pub.publish(points)
        self.pub_car.publish(marker_car)
        self.pub_per.publish(marker_per)
        self.pub_cyc.publish(marker_cyc)
        
    def print_car(self,pub,points):
        triplePoints=[]
        for (x,y,z) in points:
            p = Point()
            p.x = x
            p.y = y
            p.z = z
            triplePoints.append(p)
        marker = Marker()
        marker.header.frame_id = 'velody'
        marker.type = marker.POINTS
        marker.action = marker.ADD
        marker.pose.orientation.w = 1
        marker.points = triplePoints

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        return marker

    def print_person(self,pub,points):
        triplePoints=[]
        for (x,y,z) in points:
            p = Point()
            p.x = x
            p.y = y
            p.z = z
            triplePoints.append(p)
        marker = Marker()
        marker.header.frame_id = 'velody'
        marker.type = marker.POINTS
        marker.action = marker.ADD
        marker.pose.orientation.w = 1
        marker.points = triplePoints
    
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.g = 1.0
        return marker
    
    def print_cyc(self,pub,points):
        triplePoints=[]
        for (x,y,z) in points:
            p = Point()
            p.x = x
            p.y = y
            p.z = z
            triplePoints.append(p)
        marker = Marker()
        marker.header.frame_id = 'velody'
        marker.type = marker.POINTS
        marker.action = marker.ADD
        marker.pose.orientation.w = 1
        marker.points = triplePoints
    
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.b = 1.0
        return marker
    def tansform_data_projection(self,points,num_height,num_width):
        append_zero = np.zeros((points.shape[0],2))
        r = np.sqrt(points[:,0]*points[:,0]+points[:,1]*points[:,1])
        append_zero[:,0] += np.arcsin(points[:,2]/points[:,4])
        append_zero[:,1] += np.arcsin(points[:,1]/r)
        alig_data = np.hstack((points,append_zero))
        '''
        for p in alig_data:
        
            r = np.sqrt(p[0] * p[0] + p[1] * p[1])  # p[0] x
            p[5] = np.arcsin(p[2] / p[4])            # p[1] y
            p[6] = np.arcsin(p[1] / r)    # fie
        '''
        
        theta =alig_data[:,5]
        fie   =alig_data[:,6]
        t_a=theta.max()
        t_i= theta.min()
        f_a=fie.max()
        f_i=fie.min()
        t_range = (t_a - t_i)
        f_range = (f_a - f_i)

        resolution_h = t_range / num_height
        resolution_w = (f_range)/ num_width   
        x_min = (f_i)/resolution_w
        y_min = (t_i)/resolution_h
    
        append_64 = np.zeros((num_height,num_width,5))
        for p in alig_data:
        #print 'check',t_i,p[7],t_a
                index_h=(p[5]/resolution_h-y_min)
                index_w=(p[6]/resolution_w-x_min)
                shitf_h =-round(index_h-64)
                #shitf_h=math.floor(shitf_h)
                #print 't_i',p[7],resolution_w
                #print 'check',index_w,math.floor(index_w),round(x_min)
                shitf_w = -round(index_w-512)
              
                append_64[int(shitf_h) - 1, int(shitf_w) - 1, 0] = p[0]  # x
                append_64[int(shitf_h) - 1, int(shitf_w) - 1, 1] = p[1]  # y
                append_64[int(shitf_h) - 1, int(shitf_w) - 1, 2] = p[2]  # z
                append_64[int(shitf_h) - 1, int(shitf_w) - 1, 3] = p[3]  # in
                append_64[int(shitf_h) - 1, int(shitf_w) - 1, 4] = p[4]  # dis 


        return append_64



    def filter_camera_angle(self,places):
        bool_in = np.logical_and((places[:, 1] < places[:, 0] - 0.27), (-places[:, 1] < places[:, 0] - 0.27))
    # bool_in = np.logical_and((places[:, 1] < places[:, 0]), (-places[:, 1] < places[:, 0]))
        return places[bool_in]

    def cal_distance(self,points):
        '''
        input points: (--,4)
        '''
        print (points.shape)
        append_zero = np.sqrt((points[:,0]*points[:,0]+points[:,1]*points[:,1]+points[:,2]*points[:,2]))
        
        
        append_zero = append_zero.reshape(-1,1)
        points = np.hstack((points,append_zero))
        #for xyz in points:
            #xyz[4] =np.sqrt((xyz[0]*xyz[0]+xyz[1]*xyz[1]+xyz[2]*xyz[2]))
        print ('test',points.shape)

        return points
    def _normalize(self,x):
      return (x - x.min())/(x.max() - x.min())


    def detect(self):
      """Detect LiDAR data."""

      os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

      with tf.Graph().as_default():
        mc = kitti_squeezeSeg_config()
        mc.LOAD_PRETRAINED_MODEL = False
        mc.BATCH_SIZE = 1 # TODO(bichen): fix this hard-coded batch size.
        model = SqueezeSeg(mc)

        saver = tf.train.Saver(model.model_params)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            saver.restore(sess, FLAGS.checkpoint)
      #for f in glob.iglob(FLAGS.input_path):
        #lidar = np.load(f).astype(np.float32, copy=False)[:, :, :5]
        #topic kitti/velo/pointcloud
        
         
            start =time.time()
            print(self.input_lidar.shape)
            lidar = self.input_lidar
            lidar_mask = np.reshape(
            (lidar[:, :, 4] > 0),
            [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1]
        )
            lidar_nor = (lidar - mc.INPUT_MEAN)/mc.INPUT_STD

            pred_cls = sess.run(
            model.pred_cls,
            feed_dict={
                model.lidar_input:[lidar_nor],
                model.keep_prob: 1.0,
                model.lidar_mask:[lidar_mask]
            }
        )
            print("pass")

        # save the data
        #print(pred_cls[0].shape)
        #plt.imshow(pred_cls[0])
            lidar_xyz = lidar[:,:,:3]
            print(lidar_xyz.shape)
            lidar_raw = lidar_xyz.reshape(-1,3)
            print (lidar_raw.shape)
            
            Mask = pred_cls[0].reshape(-1,1)
            print(Mask[:,0].shape)
            id_0 = np.argwhere(Mask[:,0]==1)
            id_1 = np.argwhere(Mask[:,0]==2)
            id_2 = np.argwhere(Mask[:,0]==3)
        
            car= []
            for index in id_0:
                car.append(lidar_raw[index,:])
            car = np.array(car)
            self.car = car.reshape(-1,3)
        
            pedestrian=[]
            for index in id_1:
                pedestrian.append(lidar_raw[index,:])
            pedestrian = np.array(pedestrian)
            self.pedestrian = pedestrian.reshape(-1,3)
        
            cyc=[]
            for index in id_2:
                cyc.append(lidar_raw[index,:])
            cyc = np.array(cyc)
            self.cyc = cyc.reshape(-1,3)
        
        
            self.publish_pc(lidar_raw,self.car,self.pedestrian,self.cyc)
        
        
            print('time',time.time()-start)
    def main(self):
        rospy.spin()

if __name__ =='__main__':
   
    tensor = Ros_tensor()
    tensor.main()





