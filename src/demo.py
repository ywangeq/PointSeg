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

from config import *
from imdb import kitti
from utils.util import *
from nets import *

import os
#from sensor_msgs.msg import Image
import sensor_msgs.msg
import rospy
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
import pcl
path = os.getcwd()

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
        'checkpoint', path[:-4]+'/log/best_record/model.ckpt-49999',
        """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
        'input_path', path[:-4]+'/data/samples/*',
        """Input lidar scan to be detected. Can process glob input such as """
        """./data/samples/*148.npy or single input.""")
tf.app.flags.DEFINE_string(
        'out_dir', path[:-4]+'/data/samples_out/', """Directory to dump output.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")

def _normalize(x):
  return (x - x.min())/(x.max() - x.min())
def publish_pc(pc,car,person,cyc,mask,depth):
    
    pub = rospy.Publisher("/points_raw",PointCloud2,queue_size=1000)
    rospy.init_node("pc2_publisher")
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'velody'
    points = pc2.create_cloud_xyz32(header,pc)
    '''
    pub2 = rospy.Publisher('/point_seg',PointCloud2,queue_size=10000)
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'velody'
    points2 = pc2.create_cloud_xyz32(header,obj)
    '''
    
    pub_car =rospy.Publisher("/points_raw1",Marker,queue_size=1000)
   # marker =print_point(pub3,obj)
    pub_per =rospy.Publisher("/points_raw2",Marker,queue_size=1000)
   # marker =print_point(pub3,obj)
    pub_cyc =rospy.Publisher("/points_raw3",Marker,queue_size=1000)
  # marker =print_point(pub3,obj)
    pub_mask = rospy.Publisher("image",sensor_msgs.msg.Image,queue_size=1000)
    pub_depth = rospy.Publisher("depth",sensor_msgs.msg.Image,queue_size=1000)

    bridge = CvBridge()

    mask =  bridge.cv2_to_imgmsg(mask, encoding="passthrough")
    depth =bridge.cv2_to_imgmsg(depth, encoding="passthrough")
    r = rospy.Rate(10) 
    marker_car =print_car(pub_car,car)    
    marker_per =print_person(pub_per,person)    
    marker_cyc =print_cyc(pub_cyc,cyc)    
    while not rospy.is_shutdown():

        pub.publish(points)
        pub_car.publish(marker_car)
        pub_per.publish(marker_per)
        pub_cyc.publish(marker_cyc)
        pub_mask.publish(mask)
        pub_depth.publish(depth)
def print_car(pub,points):
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

def print_person(pub,points):
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

def print_cyc(pub,points):
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
def detect():
  """Detect LiDAR data."""

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  with tf.Graph().as_default():
    mc = kitti_squeezeSeg_config()
    mc.LOAD_PRETRAINED_MODEL = False
    mc.BATCH_SIZE = 1 # TODO(bichen): fix this hard-coded batch size.
    model = SqueezeSeg(mc)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    saver = tf.train.Saver(model.model_params)
    with tf.Session(config=config) as sess:
      saver.restore(sess, FLAGS.checkpoint)
    
      for f in sorted(glob.iglob(FLAGS.input_path),key=lambda name:int(name[-9:-4])):
        #a=sorted(FLAGS.input_path,key=lambda name:int(name[-6:-4]))
        #a = FLAGS.input_path
        #print (f)
        lidar = np.load(f).astype(np.float32, copy=False)[:, :, :5]
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
        lidar_xyz = lidar[:,:,:3]
            #print(lidar_xyz.shape)
        lidar_raw = lidar_xyz.reshape(-1,3)
        print (lidar_raw.shape)
                
        Mask = pred_cls[0].reshape(-1,1)
        print(Mask[:,0].shape)
        id_0 = np.argwhere(Mask[:,0]==1)
        id_1 = np.argwhere(Mask[:,0]==2)
        id_2 = np.argwhere(Mask[:,0]==3)
        file_name = f.strip('.npy').split('/')[-1]

        car= []
        for index in id_0:
              car.append(lidar_raw[index,:])
        car = np.array(car,dtype=np.float32)
        car = car.reshape(-1,3)
            
        pedestrian=[]
        for index in id_1:
              pedestrian.append(lidar_raw[index,:])
        pedestrian = np.array(pedestrian,dtype=np.float32)
        pedestrian = pedestrian.reshape(-1,3)
            
        cyc=[]
        for index in id_2:
              cyc.append(lidar_raw[index,:])
        cyc = np.array(cyc,dtype=np.float32)
        cyc = cyc.reshape(-1,3)
        p_cy = pcl.PointCloud(cyc)
        time1 = time.time()
        fil1 = p_cy.make_statistical_outlier_filter()
        fil1.set_mean_k(15)
        fil1.set_std_dev_mul_thresh(0.1)
        fil1.set_negative(False)
        
        p_car = pcl.PointCloud(car)
        fil2 = p_car.make_statistical_outlier_filter()
        fil2.set_mean_k(50)
        fil2.set_std_dev_mul_thresh(0.8)
        fil2.set_negative(False)
        
        
        p_p = pcl.PointCloud(pedestrian)
        fil3 = p_p.make_statistical_outlier_filter()
        fil3.set_mean_k(15)
        fil3.set_std_dev_mul_thresh(0.1)
        fil3.set_negative(False)
        
        print('ransac',time.time()-time1)
        
        
        depth_map = Image.fromarray(
             (255 * _normalize(lidar[:, :, 3])).astype(np.uint8))
        depth_map.save(
            os.path.join(FLAGS.out_dir, 'depth'+file_name+'.png'))
        label_map = Image.fromarray(
            (255 * visualize_seg(pred_cls, mc)[0]).astype(np.uint8))

        blend_map = Image.blend(
             depth_map.convert('RGBA'),
             label_map.convert('RGBA'),
             alpha=0.5
         )

        blend_map.save(
            os.path.join(FLAGS.out_dir, 'plot_'+file_name+'.png'))
        blend_map =np.array(blend_map)
        depth_map =np.array(depth_map)
        
        #publish_pc(lidar_raw,fil2.filter(),fil3.filter(),fil1.filter(),blend_map,depth_map)
        # save the data
        publish_pc(lidar_raw,car,pedestrian,cyc,blend_map,depth_map)



def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  detect()
  print('Detection output written to {}'.format(FLAGS.out_dir))


if __name__ == '__main__':
    tf.app.run()