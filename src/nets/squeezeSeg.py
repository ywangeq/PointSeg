# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016
# Modified : Wang Yuan
"""SqueezeSeg model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
from utils import util
import numpy as np
import tensorflow as tf
from nn_skeleton import ModelSkeleton

class SqueezeSeg(ModelSkeleton):
  def __init__(self, mc, gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)

      self._add_forward_graph()
      self._add_output_graph()
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()
      self._add_summary_ops()
  
  def _add_forward_graph(self):
    """NN architecture."""

    mc = self.mc
    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)
    #different Scale 
    #lidar_scale_2 = self._pooling_layer('scale_2', self.lidar_input, size=5, stride=2, padding='SAME')

    #lidar_scale_2=self._conv_layer('scale_2',self.lidar_input,filters=64,size=3,stride=2,padding='SAME')
    #lidar_scale_2 =self._scale_layer('scale_2',self.lidar_input,size=3,stride=2,padding='SAME')
    #lidar_scale_4=self._conv_layer('scale_4',self.lidar_input,filters=64,size=3,stride=4,padding='SAME')
    #concat_1 =self._squee_(lidar_scale_2,mc,count=3,name='scale_2')
    #concat_2 =self._squee_(lidar_scale_4,mc,count=3,name='scale_4')
    self.output_prob = self._squee_ori(self.lidar_input,mc)
    
    

    

  def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3,dilations=[1,1,1,1], stddev=0.001,
      freeze=False):
    """Fire layer constructor.

    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      freeze: if true, do not train parameters in this layer.
    Returns:
      fire layer operation.
    """

    sq1x1 = self._conv_layer(
        layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=[1,1],
        padding='SAME',freeze=freeze, stddev=stddev,BN=False)
    ex1x1 = self._conv_layer(
        layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=[1,1],
        padding='SAME', freeze=freeze, stddev=stddev,BN=False)
    ex3x3 = self._conv_layer(
        layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=[1,1],
        padding='SAME', dilations=dilations,freeze=freeze, stddev=stddev,BN=False)

    a=tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')
    return a
  def _fire_deconv(self, layer_name, inputs, s1x1, e1x1, e3x3, 
                   factors=[1, 2], freeze=False, stddev=0.001):
    """Fire deconvolution layer constructor.

    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      factors: spatial upsampling factors.
      freeze: if true, do not train parameters in this layer.
    Returns:
      fire layer operation.
    """

    assert len(factors) == 2,'factors should be an array of size 2'

    ksize_h = factors[0] * 2 - factors[0] % 2
    ksize_w = factors[1] * 2 - factors[1] % 2

    sq1x1 = self._conv_layer(
        layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=[1,1],
        padding='SAME', freeze=freeze, stddev=stddev,BN=False)
    deconv = self._deconv_layer(
        layer_name+'/deconv', sq1x1, filters=s1x1, size=[ksize_h, ksize_w],
        stride=factors, padding='SAME', init='bilinear')
    ex1x1 = self._conv_layer(
        layer_name+'/expand1x1', deconv, filters=e1x1, size=1, stride=[1,1],
        padding='SAME', freeze=freeze, stddev=stddev,BN=False)
    ex3x3 = self._conv_layer(
        layer_name+'/expand3x3', deconv, filters=e3x3, size=3, stride=[1,1],
        padding='SAME', freeze=freeze, stddev=stddev,BN=False)
    #lkn f_r= self.LKN_layer(deconv,k1=e1x1,k2=e3x3,stride=1,size_w=3,size_h=1,name =layer_name,freeze=False,xavier=True)
    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')
  def _single_deconv(self, layer_name, inputs, s1x1, 
                   factors=[1, 2], freeze=False, stddev=0.001):
    """Fire deconvolution layer constructor.

    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      factors: spatial upsampling factors.
      freeze: if true, do not train parameters in this layer.
    Returns:
      fire layer operation.
    """

    assert len(factors) == 2,'factors should be an array of size 2'

    ksize_h = factors[0] * 2 - factors[0] % 2
    ksize_w = factors[1] * 2 - factors[1] % 2

 
    deconv = self._deconv_layer(
        layer_name+'/deconv', inputs, filters=s1x1, size=[ksize_h, ksize_w],
        stride=factors, padding='SAME', init='bilinear')
    
    #lkn f_r= self.LKN_layer(deconv,k1=e1x1,k2=e3x3,stride=1,size_w=3,size_h=1,name =layer_name,freeze=False,xavier=True)
    return deconv