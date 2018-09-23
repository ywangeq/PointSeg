# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016
# Modified : Wang Yuan
"""Neural network model base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from utils import util
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim  
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
from tensorflow.python.ops import control_flow_ops

def _variable_on_device(name, shape, initializer, trainable=True):
  """Helper to create a Variable.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  # TODO(bichen): fix the hard-coded data type below
  dtype = tf.float32
  if not callable(initializer):
    var = tf.get_variable(name, initializer=initializer, trainable=trainable)
  else:
    var = tf.get_variable(
        name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_device(name, shape, initializer, trainable)
  if wd is not None and trainable:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

class ModelSkeleton:
  """Base class of NN detection models."""
  def __init__(self, mc):
    self.mc = mc

    # a scalar tensor in range (0, 1]. Usually set to 0.5 in training phase and
    # 1.0 in evaluation phase
    self.ph_keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # projected lidar points on a 2D spherical surface
    self.ph_lidar_input = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 5],
        name='lidar_input'
    )
    # A tensor where an element is 1 if the corresponding cell contains an
    # valid lidar measurement. Or if the data is missing, then mark it as 0.
    self.ph_lidar_mask = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1],
        name='lidar_mask')
    # A tensor where each element contains the class of each pixel
    self.ph_label = tf.placeholder(
        tf.int32, [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL],
        name='label')
    # weighted loss for different classes
    self.ph_loss_weight = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL],
        name='loss_weight')

    # define a FIFOqueue for pre-fetching data
    self.q = tf.FIFOQueue(
        capacity=mc.QUEUE_CAPACITY,
        dtypes=[tf.float32, tf.float32, tf.float32, tf.int32, tf.float32],
        shapes=[[],
                [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 5],
                [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1],
                [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL],
                [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL]]
    )
    self.enqueue_op = self.q.enqueue(
        [self.ph_keep_prob, self.ph_lidar_input, self.ph_lidar_mask,
          self.ph_label, self.ph_loss_weight]
    )

    self.keep_prob, self.lidar_input, self.lidar_mask, self.label, \
        self.loss_weight = self.q.dequeue()

    # model parameters
    self.model_params = []

    # model size counter
    self.model_size_counter = [] # array of tuple of layer name, parameter size
    # flop counter
    self.flop_counter = [] # array of tuple of layer name, flop number
    # activation counter
    self.activation_counter = [] # array of tuple of layer name, output activations
    self.activation_counter.append(('input', mc.AZIMUTH_LEVEL*mc.ZENITH_LEVEL*3))


  def _add_forward_graph(self):
    """NN architecture specification."""
    raise NotImplementedError

  def _add_output_graph(self):
    """Define how to intepret output."""
    mc = self.mc
    with tf.variable_scope('interpret_output') as scope:
      self.prob = tf.multiply(
          tf.nn.softmax(self.output_prob, dim=-1), self.lidar_mask,
          name='pred_prob')
      self.pred_cls = tf.argmax(self.prob, axis=3, name='pred_cls')

      # add summaries
      for cls_id, cls in enumerate(mc.CLASSES):
        self._activation_summary(self.prob[:, :, :, cls_id], 'prob_'+cls)

  def _add_loss_graph(self):
    """Define the loss operation."""
    mc = self.mc

    with tf.variable_scope('cls_loss') as scope:
      self.cls_loss = tf.identity(
          tf.reduce_sum(
              tf.nn.sparse_softmax_cross_entropy_with_logits(
                  labels=tf.reshape(self.label, (-1, )),
                  logits=tf.reshape(self.output_prob, (-1, mc.NUM_CLASS))
              ) \
              * tf.reshape(self.lidar_mask, (-1, )) \
              * tf.reshape(self.loss_weight, (-1, ))
          )/tf.reduce_sum(self.lidar_mask)*mc.CLS_LOSS_COEF,
          name='cls_loss'
      )
      
      tf.add_to_collection('losses', self.cls_loss)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      print('-------',update_ops)
      if update_ops:
        updates = tf.group(*update_ops)
        self.cls_loss = control_flow_ops.with_dependencies([updates], self.cls_loss)

    # add above losses as well as weight decay losses to form the total loss
    self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    # add loss summaries
    # _add_loss_summaries(self.loss)
    tf.summary.scalar(self.cls_loss.op.name, self.cls_loss)
    tf.summary.scalar(self.loss.op.name, self.loss)

  def _add_train_graph(self):
    """Define the training operation."""
    mc = self.mc

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.exponential_decay(mc.LEARNING_RATE,
                                    self.global_step,
                                    mc.DECAY_STEPS,
                                    mc.LR_DECAY_FACTOR,
                                    staircase=True)

    tf.summary.scalar('learning_rate', lr)

    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=mc.MOMENTUM)
    grads_vars = opt.compute_gradients(self.loss, tf.trainable_variables())

    with tf.variable_scope('clip_gradient') as scope:
      for i, (grad, var) in enumerate(grads_vars):
        grads_vars[i] = (tf.clip_by_norm(grad, mc.MAX_GRAD_NORM), var)

    apply_gradient_op = opt.apply_gradients(grads_vars, global_step=self.global_step)

    with tf.control_dependencies([apply_gradient_op]):
      self.train_op = tf.no_op(name='train')

  def _add_viz_graph(self):
    """Define the visualization operation."""
    mc = self.mc
    self.label_to_show = tf.placeholder(
        tf.float32, [None, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 3],
        name='label_to_show'
    )
    self.depth_image_to_show = tf.placeholder(
        tf.float32, [None, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1],
        name='depth_image_to_show'
    )
    self.pred_image_to_show = tf.placeholder(
        tf.float32, [None, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 3],
        name='pred_image_to_show'
    )
    self.show_label = tf.summary.image('label_to_show',
        self.label_to_show, collections='image_summary',
        max_outputs=mc.BATCH_SIZE)
    self.show_depth_img = tf.summary.image('depth_image_to_show',
        self.depth_image_to_show, collections='image_summary',
        max_outputs=mc.BATCH_SIZE)
    self.show_pred = tf.summary.image('pred_image_to_show',
        self.pred_image_to_show, collections='image_summary',
        max_outputs=mc.BATCH_SIZE)

  def _add_summary_ops(self):
    """Add extra summary operations."""
    mc = self.mc

    iou_summary_placeholders = []
    iou_summary_ops = []

    for cls in mc.CLASSES:
      ph = tf.placeholder(tf.float32, name=cls+'_iou')
      iou_summary_placeholders.append(ph)
      iou_summary_ops.append(
          tf.summary.scalar('Eval/'+cls+'_iou', ph, collections='eval_summary')
      )

    self.iou_summary_placeholders = iou_summary_placeholders
    self.iou_summary_ops = iou_summary_ops

  def _conv_bn_layer(
      self, inputs, conv_param_name, bn_param_name, scale_param_name, filters,
      size, stride, padding='SAME',dilations=[1,1,1,1],freeze=False, relu=True,
      conv_with_bias=False, stddev=0.001):
    """ Convolution + BatchNorm + [relu] layer. Batch mean and var are treated
    as constant. Weights have to be initialized from a pre-trained model or
    restored from a checkpoint.

    Args:
      inputs: input tensor
      conv_param_name: name of the convolution parameters
      bn_param_name: name of the batch normalization parameters
      scale_param_name: name of the scale parameters
      filters: number of output filters.
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
      freeze: if true, then do not train the parameters in this layer.
      xavier: whether to use xavier weight initializer or not.
      relu: whether to use relu or not.
      conv_with_bias: whether or not add bias term to the convolution output.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A convolutional layer operation.
    """
    mc = self.mc

    with tf.variable_scope(conv_param_name) as scope:
      channels = inputs.get_shape()[3]

      if mc.LOAD_PRETRAINED_MODEL:
        cw = self.caffemodel_weight
        kernel_val = np.transpose(cw[conv_param_name][0], [2,3,1,0])
        if conv_with_bias:
          bias_val = cw[conv_param_name][1]
        mean_val   = cw[bn_param_name][0]
        var_val    = cw[bn_param_name][1]
        gamma_val  = cw[scale_param_name][0]
        beta_val   = cw[scale_param_name][1]
      else:
        kernel_val = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        if conv_with_bias:
          bias_val = tf.constant_initializer(0.0)
        mean_val   = tf.constant_initializer(0.0)
        var_val    = tf.constant_initializer(1.0)
        gamma_val  = tf.constant_initializer(1.0)
        beta_val   = tf.constant_initializer(0.0)

      # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      # shape [h, w, in, out]
      kernel = _variable_with_weight_decay(
          'kernels', shape=[size, size, int(channels), filters],
          wd=mc.WEIGHT_DECAY, initializer=kernel_val, trainable=(not freeze))
      self.model_params += [kernel]
      if conv_with_bias:
        biases = _variable_on_device('biases', [filters], bias_val,
                                     trainable=(not freeze))
        self.model_params += [biases]
      gamma = _variable_on_device('gamma', [filters], gamma_val,
                                  trainable=(not freeze))
      beta  = _variable_on_device('beta', [filters], beta_val,
                                  trainable=(not freeze))
      mean  = _variable_on_device('mean', [filters], mean_val, trainable=False)
      var   = _variable_on_device('var', [filters], var_val, trainable=False)
      self.model_params += [gamma, beta, mean, var]

      conv = tf.nn.conv2d(
          inputs, kernel, [1, 1, stride, 1], padding=padding,
          name='convolution',dilations=dilations)
      if conv_with_bias:
        conv = tf.nn.bias_add(conv, biases, name='bias_add')

      conv = tf.nn.batch_normalization(
          conv, mean=mean, variance=var, offset=beta, scale=gamma,
          variance_epsilon=mc.BATCH_NORM_EPSILON, name='batch_norm')

      self.model_size_counter.append(
          (conv_param_name, (1+size*size*int(channels))*filters)
      )
      out_shape = conv.get_shape().as_list()
      num_flops = \
        (1+2*int(channels)*size*size)*filters*out_shape[1]*out_shape[2]
      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]
      self.flop_counter.append((conv_param_name, num_flops))

      self.activation_counter.append(
          (conv_param_name, out_shape[1]*out_shape[2]*out_shape[3])
      )

      if relu:
        return tf.nn.relu(conv)
      else:
        return conv


  def _conv_layer(
      self, layer_name, inputs, filters, size, stride, padding='SAME',dilations=[1,1,1,1],
      freeze=False, xavier=False, relu=True, BN=True,stddev=0.001, bias_init_val=0.0):
    """Convolutional layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      filters: number of output filters.
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
      freeze: if true, then do not train the parameters in this layer.
      xavier: whether to use xavier weight initializer or not.
      relu: whether to use relu or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A convolutional layer operation.
    """

    mc = self.mc
    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        kernel_val = np.transpose(cw[layer_name][0], [2,3,1,0])
        bias_val = cw[layer_name][1]
        # check the shape
        if (kernel_val.shape == 
              (size, size, inputs.get_shape().as_list()[-1], filters)) \
           and (bias_val.shape == (filters, )):
          use_pretrained_param = True
        else:
          print ('Shape of the pretrained parameter of {} does not match, '
              'use randomly initialized parameter'.format(layer_name))
      else:
        print ('Cannot find {} in the pretrained model. Use randomly initialized '
               'parameters'.format(layer_name))

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      channels = inputs.get_shape()[3]

      # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      # shape [h, w, in, out]
      if use_pretrained_param:
        if mc.DEBUG_MODE:
          print ('Using pretrained model for {}'.format(layer_name))
        kernel_init = tf.constant(kernel_val , dtype=tf.float32)
        bias_init = tf.constant(bias_val, dtype=tf.float32)
      elif xavier:
        kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(bias_init_val)
      else:
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(bias_init_val)

      kernel = _variable_with_weight_decay(
          'kernels', shape=[size, size, int(channels), filters],
          wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

      biases = _variable_on_device('biases', [filters], bias_init, 
                                trainable=(not freeze))
      self.model_params += [kernel, biases]

      conv = tf.nn.conv2d(
          inputs, kernel, [1, stride[0], stride[1], 1], padding=padding,
          name='convolution',dilations=dilations)
      conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
      
      if BN:
       # gamma_val  = tf.constant_initializer(1.0)
       # beta_val   = tf.constant_initializer(0.0)
       # gamma = _variable_on_device('gamma', [filters], gamma_val,
        #                          trainable=(not freeze))
       # beta  = _variable_on_device('beta', [filters], beta_val,
       #                           trainable=(not freeze))
        train = tf.cast((not freeze),tf.bool)
        #conv_bias = self.Batch_Normalization(conv_bias,training=train,scope=layer_name+'_BN')
        #conv_bias=self.batch_norm(conv_bias,beta,gamma,train)
        conv_bias=self.batch_norm2(conv_bias,filters,train)
      if relu:
        out = tf.nn.relu(conv_bias, 'relu')
      else:
        out = conv_bias

      self.model_size_counter.append(
          (layer_name, (1+size*size*int(channels))*filters)
      )
      out_shape = out.get_shape().as_list()
      num_flops = \
        (1+2*int(channels)*size*size)*filters*out_shape[1]*out_shape[2]
      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]
      self.flop_counter.append((layer_name, num_flops))

      self.activation_counter.append(
          (layer_name, out_shape[1]*out_shape[2]*out_shape[3])
      )

      return out
  
  def _conv_layer_stride(self, layer_name, inputs, filters, size, stride, padding='SAME',freeze=False, xavier=False, relu=True, stddev=0.001, bias_init_val=0.0):
    """Convolutional layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      filters: number of output filters.
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
      freeze: if true, then do not train the parameters in this layer.
      xavier: whether to use xavier weight initializer or not.
      relu: whether to use relu or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A convolutional layer operation.
    """

    mc = self.mc
    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        kernel_val = np.transpose(cw[layer_name][0], [2,3,1,0])
        bias_val = cw[layer_name][1]
        # check the shape
        if (kernel_val.shape == 
              (size, size, inputs.get_shape().as_list()[-1], filters)) \
           and (bias_val.shape == (filters, )):
          use_pretrained_param = True
        else:
          print ('Shape of the pretrained parameter of {} does not match, '
              'use randomly initialized parameter'.format(layer_name))
      else:
        print ('Cannot find {} in the pretrained model. Use randomly initialized '
               'parameters'.format(layer_name))

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      channels = inputs.get_shape()[3]

      # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      # shape [h, w, in, out]
      if use_pretrained_param:
        if mc.DEBUG_MODE:
          print ('Using pretrained model for {}'.format(layer_name))
        kernel_init = tf.constant(kernel_val , dtype=tf.float32)
        bias_init = tf.constant(bias_val, dtype=tf.float32)
      elif xavier:
        kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(bias_init_val)
      else:
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(bias_init_val)

      kernel = _variable_with_weight_decay(
          'kernels', shape=[size, size, int(channels), filters],
          wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

      biases = _variable_on_device('biases', [filters], bias_init, 
                                trainable=(not freeze))
      self.model_params += [kernel, biases]

      conv = tf.nn.conv2d(
          inputs, kernel, [1, stride, stride, 1], padding=padding,
          name='convolution')
      conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
  
      if relu:
        out = tf.nn.relu(conv_bias, 'relu')
      else:
        out = conv_bias

      self.model_size_counter.append(
          (layer_name, (1+size*size*int(channels))*filters)
      )
      out_shape = out.get_shape().as_list()
      num_flops = \
        (1+2*int(channels)*size*size)*filters*out_shape[1]*out_shape[2]
      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]
      self.flop_counter.append((layer_name, num_flops))

      self.activation_counter.append(
          (layer_name, out_shape[1]*out_shape[2]*out_shape[3])
      )

      return out
  def _conv_layer_k(
      self, layer_name, inputs, filters, size_w,size_h, stride, padding='SAME',
      freeze=False, xavier=False, relu=True, stddev=0.001, bias_init_val=0.0):
    """Convolutional layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      filters: number of output filters.
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
      freeze: if true, then do not train the parameters in this layer.
      xavier: whether to use xavier weight initializer or not.
      relu: whether to use relu or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A convolutional layer operation.
    """

    mc = self.mc
    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        kernel_val = np.transpose(cw[layer_name][0], [2,3,1,0])
        bias_val = cw[layer_name][1]
        # check the shape
        if (kernel_val.shape == 
              (size_w, size_h, inputs.get_shape().as_list()[-1], filters)) \
           and (bias_val.shape == (filters, )):
          use_pretrained_param = True
        else:
          print ('Shape of the pretrained parameter of {} does not match, '
              'use randomly initialized parameter'.format(layer_name))
      else:
        print ('Cannot find {} in the pretrained model. Use randomly initialized '
               'parameters'.format(layer_name))

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      channels = inputs.get_shape()[3]

      # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      # shape [h, w, in, out]
      if use_pretrained_param:
        if mc.DEBUG_MODE:
          print ('Using pretrained model for {}'.format(layer_name))
        kernel_init = tf.constant(kernel_val , dtype=tf.float32)
        bias_init = tf.constant(bias_val, dtype=tf.float32)
      elif xavier:
        kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(bias_init_val)
      else:
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(bias_init_val)

      kernel = _variable_with_weight_decay(
          'kernels', shape=[size_w, size_h, int(channels), filters],
          wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

      biases = _variable_on_device('biases', [filters], bias_init, 
                                trainable=(not freeze))
      self.model_params += [kernel, biases]

      conv = tf.nn.conv2d(
          inputs, kernel, [1, stride, stride, 1], padding=padding,
          name='convolution')
      conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
      
      if relu:
        out = tf.nn.relu(conv_bias, 'relu')
      else:
        out = conv_bias

      self.model_size_counter.append(
          (layer_name, (1+size_w*size_h*int(channels))*filters)
      )
      out_shape = out.get_shape().as_list()
      num_flops = \
        (1+2*int(channels)*size_h*size_w)*filters*out_shape[1]*out_shape[2]
      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]
      self.flop_counter.append((layer_name, num_flops))

      self.activation_counter.append(
          (layer_name, out_shape[1]*out_shape[2]*out_shape[3])
      )

      return out

  def _deconv_layer(
      self, layer_name, inputs, filters, size, stride, padding='SAME',
      freeze=False, init='trunc_norm', relu=True, stddev=0.001):
    """Deconvolutional layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      filters: number of output filters.
      size: kernel size. An array of size 2 or 1.
      stride: stride. An array of size 2 or 1.
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
      freeze: if true, then do not train the parameters in this layer.
      init: how to initialize kernel weights. Now accept 'xavier',
          'trunc_norm', 'bilinear'
      relu: whether to use relu or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A convolutional layer operation.
    """

    assert len(size) == 1 or len(size) == 2, \
        'size should be a scalar or an array of size 2.'
    assert len(stride) == 1 or len(stride) == 2, \
        'stride should be a scalar or an array of size 2.'
    assert init == 'xavier' or init == 'bilinear' or init == 'trunc_norm', \
        'initi mode not supported {}'.format(init)

    if len(size) == 1:
      size_h, size_w = size[0], size[0]
    else:
      size_h, size_w = size[0], size[1]

    if len(stride) == 1:
      stride_h, stride_w = stride[0], stride[0]
    else:
      stride_h, stride_w = stride[0], stride[1]

    mc = self.mc
    # TODO(bichen): Currently do not support pretrained parameters for deconv
    # layer.

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      in_height = int(inputs.get_shape()[1])
      in_width = int(inputs.get_shape()[2])
      channels = int(inputs.get_shape()[3])

      if init == 'xavier':
          kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
          bias_init = tf.constant_initializer(0.0)
      elif init == 'bilinear':
        assert size_h == 1, 'Now only support size_h=1'
        assert channels == filters, \
            'In bilinear interporlation mode, input channel size and output' \
            'filter size should be the same'
        assert stride_h == 1, \
            'In bilinear interpolation mode, stride_h should be 1'

        kernel_init = np.zeros(
            (size_h, size_w, channels, channels),
            dtype=np.float32)

        factor_w = (size_w + 1)//2
        assert factor_w == stride_w, \
            'In bilinear interpolation mode, stride_w == factor_w'

        center_w = (factor_w - 1) if (size_w % 2 == 1) else (factor_w - 0.5)
        og_w = np.reshape(np.arange(size_w), (size_h, -1))
        up_kernel = (1 - np.abs(og_w - center_w)/factor_w)
        for c in xrange(channels):
          kernel_init[:, :, c, c] = up_kernel

        bias_init = tf.constant_initializer(0.0)
      else:
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(0.0)

      # Kernel layout for deconv layer: [H_f, W_f, O_c, I_c] where I_c is the
      # input channel size. It should be the same as the channel size of the
      # input tensor. 
      kernel = _variable_with_weight_decay(
          'kernels', shape=[size_h, size_w, filters, channels],
          wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))
      biases = _variable_on_device(
          'biases', [filters], bias_init, trainable=(not freeze))
      self.model_params += [kernel, biases]

      # TODO(bichen): fix this
      deconv = tf.nn.conv2d_transpose(
          inputs, kernel, 
          [mc.BATCH_SIZE, stride_h*in_height, stride_w*in_width, filters],
          [1, stride_h, stride_w, 1], padding=padding,
          name='deconv')
      deconv_bias = tf.nn.bias_add(deconv, biases, name='bias_add')

      if relu:
        out = tf.nn.relu(deconv_bias, 'relu')
      else:
        out = deconv_bias

      self.model_size_counter.append(
          (layer_name, (1+size_h*size_w*channels)*filters)
      )
      out_shape = out.get_shape().as_list()
      num_flops = \
        (1+2*channels*size_h*size_w)*filters*out_shape[1]*out_shape[2]
      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]
      self.flop_counter.append((layer_name, num_flops))

      self.activation_counter.append(
          (layer_name, out_shape[1]*out_shape[2]*out_shape[3])
      )

      return out
  
    
    
    
  def _scale_layer(self, layer_name, inputs, size, stride, padding='SAME'):
    with tf.variable_scope(layer_name) as scope:
      out =  tf.nn.avg_pool(inputs, 
                            ksize=[1, size, size, 1], 
                            strides=[1, 1, stride, 1],
                            padding=padding)
      #activation_size = np.prod(out.get_shape().as_list()[1:])
      #self.activation_counter.append((layer_name, activation_size))
      return out
    
  def _pooling_layer(self, layer_name, inputs, size, stride, padding='SAME'):
    """Pooling layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
    Returns:
      A pooling layer operation.
    """

    with tf.variable_scope(layer_name) as scope:
      out =  tf.nn.max_pool(inputs, 
                            ksize=[1, size, size, 1], 
                            strides=[1, 1, stride, 1],
                            padding=padding)
      activation_size = np.prod(out.get_shape().as_list()[1:])
      self.activation_counter.append((layer_name, activation_size))
      return out

  
  def _fc_layer(
      self, layer_name, inputs, hiddens, flatten=False, relu=True,
      xavier=False, stddev=0.001, bias_init_val=0.0):
    """Fully connected layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      hiddens: number of (hidden) neurons in this layer.
      flatten: if true, reshape the input 4D tensor of shape 
          (batch, height, weight, channel) into a 2D tensor with shape 
          (batch, -1). This is used when the input to the fully connected layer
          is output of a convolutional layer.
      relu: whether to use relu or not.
      xavier: whether to use xavier weight initializer or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A fully connected layer operation.
    """
    mc = self.mc

    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        use_pretrained_param = True
        kernel_val = cw[layer_name][0]
        bias_val = cw[layer_name][1]

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      input_shape = inputs.get_shape().as_list()
      if flatten:
        dim = input_shape[1]*input_shape[2]*input_shape[3]
        inputs = tf.reshape(inputs, [-1, dim])
        if use_pretrained_param:
          try:
            # check the size before layout transform
            assert kernel_val.shape == (hiddens, dim), \
                'kernel shape error at {}'.format(layer_name)
            kernel_val = np.reshape(
                np.transpose(
                    np.reshape(
                        kernel_val, # O x (C*H*W)
                        (hiddens, input_shape[3], input_shape[1], input_shape[2])
                    ), # O x C x H x W
                    (2, 3, 1, 0)
                ), # H x W x C x O
                (dim, -1)
            ) # (H*W*C) x O
            # check the size after layout transform
            assert kernel_val.shape == (dim, hiddens), \
                'kernel shape error at {}'.format(layer_name)
          except:
            # Do not use pretrained parameter if shape doesn't match
            use_pretrained_param = False
            print ('Shape of the pretrained parameter of {} does not match, '
                   'use randomly initialized parameter'.format(layer_name))
      else:
        dim = input_shape[1]
        if use_pretrained_param:
          try:
            kernel_val = np.transpose(kernel_val, (1,0))
            assert kernel_val.shape == (dim, hiddens), \
                'kernel shape error at {}'.format(layer_name)
          except:
            use_pretrained_param = False
            print ('Shape of the pretrained parameter of {} does not match, '
                   'use randomly initialized parameter'.format(layer_name))

      if use_pretrained_param:
        if mc.DEBUG_MODE:
          print ('Using pretrained model for {}'.format(layer_name))
        kernel_init = tf.constant(kernel_val, dtype=tf.float32)
        bias_init = tf.constant(bias_val, dtype=tf.float32)
      elif xavier:
        kernel_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(bias_init_val)
      else:
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(bias_init_val)

      weights = _variable_with_weight_decay(
          'weights', shape=[dim, hiddens], wd=mc.WEIGHT_DECAY,
          initializer=kernel_init)
      biases = _variable_on_device('biases', [hiddens], bias_init)
      self.model_params += [weights, biases]
  
      outputs = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
      if relu:
        outputs = tf.nn.relu(outputs, 'relu')

      # count layer stats
      self.model_size_counter.append((layer_name, (dim+1)*hiddens))

      num_flops = 2 * dim * hiddens + hiddens
      if relu:
        num_flops += 2*hiddens
      self.flop_counter.append((layer_name, num_flops))

      self.activation_counter.append((layer_name, hiddens))

      return outputs

  def _recurrent_crf_layer(
      self, layer_name, inputs, bilateral_filters, sizes=[3, 5],
      num_iterations=1, padding='SAME'):
    """Recurrent conditional random field layer. Iterative meanfield inference is
    implemented as a reccurent neural network.

    Args:
      layer_name: layer name
      inputs: input tensor with shape [batch_size, zenith, azimuth, num_class].
      bilateral_filters: filter weight with shape 
          [batch_size, zenith, azimuth, sizes[0]*size[1]-1].
      sizes: size of the local region to be filtered.
      num_iterations: number of meanfield inferences.
      padding: padding strategy
    Returns:
      outputs: tensor with shape [batch_size, zenith, azimuth, num_class].
    """
    assert num_iterations >= 1, 'number of iterations should >= 1'

    mc = self.mc
    with tf.variable_scope(layer_name) as scope:
      # initialize compatibilty matrices
      compat_kernel_init = tf.constant(
          np.reshape(
              np.ones((mc.NUM_CLASS, mc.NUM_CLASS)) - np.identity(mc.NUM_CLASS),
              [1, 1, mc.NUM_CLASS, mc.NUM_CLASS]
          ),
          dtype=tf.float32
      )
      bi_compat_kernel = _variable_on_device(
          name='bilateral_compatibility_matrix',
          shape=[1, 1, mc.NUM_CLASS, mc.NUM_CLASS],
          initializer=compat_kernel_init*mc.BI_FILTER_COEF,
          trainable=True
      )
      self._activation_summary(bi_compat_kernel, 'bilateral_compat_mat')

      angular_compat_kernel = _variable_on_device(
          name='angular_compatibility_matrix',
          shape=[1, 1, mc.NUM_CLASS, mc.NUM_CLASS],
          initializer=compat_kernel_init*mc.ANG_FILTER_COEF,
          trainable=True
      )
      self._activation_summary(angular_compat_kernel, 'angular_compat_mat')

      self.model_params += [bi_compat_kernel, angular_compat_kernel]

      condensing_kernel = tf.constant(
          util.condensing_matrix(sizes[0], sizes[1], mc.NUM_CLASS),
          dtype=tf.float32,
          name='condensing_kernel'
      )

      angular_filters = tf.constant(
          util.angular_filter_kernel(
              sizes[0], sizes[1], mc.NUM_CLASS, mc.ANG_THETA_A**2),
          dtype=tf.float32,
          name='angular_kernel'
      )

      bi_angular_filters = tf.constant(
          util.angular_filter_kernel(
              sizes[0], sizes[1], mc.NUM_CLASS, mc.BILATERAL_THETA_A**2),
          dtype=tf.float32,
          name='bi_angular_kernel'
      )

      for it in range(num_iterations):
        unary = tf.nn.softmax(
            inputs, dim=-1, name='unary_term_at_iter_{}'.format(it))

        ang_output, bi_output = self._locally_connected_layer(
            'message_passing_iter_{}'.format(it), unary,
            bilateral_filters, angular_filters, bi_angular_filters,
            condensing_kernel, sizes=sizes,
            padding=padding
        )

        # 1x1 convolution as compatibility transform
        ang_output = tf.nn.conv2d(
            ang_output, angular_compat_kernel, strides=[1, 1, 1, 1],
            padding='SAME', name='angular_compatibility_transformation')
        self._activation_summary(
            ang_output, 'ang_transfer_iter_{}'.format(it))

        bi_output = tf.nn.conv2d(
            bi_output, bi_compat_kernel, strides=[1, 1, 1, 1], padding='SAME',
            name='bilateral_compatibility_transformation')
        self._activation_summary(
            bi_output, 'bi_transfer_iter_{}'.format(it))

        pairwise = tf.add(ang_output, bi_output,
                          name='pairwise_term_at_iter_{}'.format(it))

        outputs = tf.add(unary, pairwise,
                         name='energy_at_iter_{}'.format(it))

        inputs = outputs

    return outputs

  def _locally_connected_layer(
      self, layer_name, inputs, bilateral_filters,
      angular_filters, bi_angular_filters, condensing_kernel, sizes=[3, 5],
      padding='SAME'):
    """Locally connected layer with non-trainable filter parameters)

    Args:
      layer_name: layer name
      inputs: input tensor with shape 
          [batch_size, zenith, azimuth, num_class].
      bilateral_filters: bilateral filter weight with shape 
          [batch_size, zenith, azimuth, sizes[0]*size[1]-1].
      angular_filters: angular filter weight with shape 
          [sizes[0], sizes[1], in_channel, in_channel].
      condensing_kernel: tensor with shape 
          [size[0], size[1], num_class, (sizes[0]*size[1]-1)*num_class]
      sizes: size of the local region to be filtered.
      padding: padding strategy
    Returns:
      ang_output: output tensor filtered by anguler filter with shape 
          [batch_size, zenith, azimuth, num_class].
      bi_output: output tensor filtered by bilateral filter with shape 
          [batch_size, zenith, azimuth, num_class].
    """
    assert padding=='SAME', 'only support SAME padding strategy'
    assert sizes[0] % 2 == 1 and sizes[1] % 2 == 1, \
        'Currently only support odd filter size.'

    mc = self.mc
    size_z, size_a = sizes
    pad_z, pad_a = size_z//2, size_a//2
    half_filter_dim = (size_z*size_a)//2
    batch, zenith, azimuth, in_channel = inputs.shape.as_list()

    with tf.variable_scope(layer_name) as scope:
      # message passing
      ang_output = tf.nn.conv2d(
          inputs, angular_filters, [1, 1, 1, 1], padding=padding,
          name='angular_filtered_term'
      )

      bi_ang_output = tf.nn.conv2d(
          inputs, bi_angular_filters, [1, 1, 1, 1], padding=padding,
          name='bi_angular_filtered_term'
      )

      condensed_input = tf.reshape(
          tf.nn.conv2d(
              inputs*self.lidar_mask, condensing_kernel, [1, 1, 1, 1], padding=padding,
              name='condensed_prob_map'
          ),
          [batch, zenith, azimuth, size_z*size_a-1, in_channel]
      )

      bi_output = tf.multiply(
          tf.reduce_sum(condensed_input*bilateral_filters, axis=3),
          self.lidar_mask,
          name='bilateral_filtered_term'
      )
      bi_output *= bi_ang_output

    return ang_output, bi_output
  def _squee_ori(self,input,mc):
    
    #scale_2 = self._conv_layer('conv2_1', input, filters=48, size=3, stride=2,padding='SAME', freeze=False, xavier=True)
    #scale_2 = self._pooling_layer('scale_2', input, size=1, stride=2, padding='SAME')
    #concat_layer_2_1 = self._squzee_down_8_without_input_down(scale_2,mc,'scale_2')
    
    conv1 = self._conv_layer('conv1', input, filters=64, size=3, stride=[1,2],padding='SAME', freeze=False, xavier=True,BN=False)

    conv1_skip = self._conv_layer(
        'conv1_skip', input, filters=64, size=1, stride=[1,1],padding='SAME', freeze=False, xavier=True,BN=False)

    pool1 = self._pooling_layer(
        'pool1', conv1, size=3, stride=2, padding='SAME')

    fire2 = self._fire_layer(
        'fire2', pool1, s1x1=16,e1x1=64, e3x3=64, freeze=False)
    fire3 = self._fire_layer(
        'fire3', fire2, s1x1=16, e1x1=64,e3x3=64,freeze=False)
    fire3 = self.Squeeze_ex_layer(fire3,128,2,name='SE1')
    pool3 = self._pooling_layer(
        'pool3', fire3, size=3, stride=2, padding='SAME')

    fire4 = self._fire_layer(
        'fire4', pool3, s1x1=32,e1x1=128, e3x3=128, freeze=False)
    fire5 = self._fire_layer(
        'fire5', fire4, s1x1=32,e1x1=128,  e3x3=128,freeze=False)
    fire5 = self.Squeeze_ex_layer(fire5,256,2,'SE2')
    pool5 = self._pooling_layer(
        'pool5', fire5, size=3, stride=2, padding='SAME')

    fire6 = self._fire_layer(
        'fire6', pool5, s1x1=48, e1x1=192,e3x3=192, freeze=False)
    fire7 = self._fire_layer(
        'fire7', fire6, s1x1=48, e1x1=192,e3x3=192 ,freeze=False)
    fire8 = self._fire_layer(
        'fire8', fire7, s1x1=64 ,e1x1=256,e3x3=256,freeze=False)
    fire9 = self._fire_layer(
        'fire9', fire8, s1x1=64, e1x1=256,e3x3=256, freeze=False)
    fire9 = self.Squeeze_ex_layer(fire9,512,2,'SE3')
    # Deconvolation
    ##fire9 =self.LKN_layer(fire9,k1=256,k2=256,stride=1,size_w=7,size_h=1,name ='LKN_2',freeze=False,xavier=True)
    ##fire9 =self.SR_layer(fire9,k1=256,k2=256,size=7,stride=1,name='SR_2',freeze=False,xavier=True)

    ASPP=self.atrous_spatial_pyramid_pooling(fire9,scope='ASPP')
    #deconv_ASPP_0 = self._conv_layer('conv0_ASPP',ASPP, filters=128, size=3, stride=[1,1],padding='SAME', dilations=[1,1,1,1],BN=False,freeze=False, xavier=True)
    fire_ASPP = self._fire_deconv('fire_ASPP',ASPP,s1x1=32,e1x1=128,e3x3=128,factors=[1,2],stddev=0.1)
                                 
                                 
    fire10 = self._fire_deconv(
        'fire_deconv10', fire9, s1x1=64, e1x1=128,e3x3=128,factors=[1, 2],
        stddev=0.1)
    #--R_sub_5_10 = tf.subtract(fire5,fire10,name='fire5-10')
    #--fire10_fuse = tf.concat([fire10,R_sub_5_10],3,name='sub_concat_5')
    ##fire5 =self.LKN_layer(fire5,k1=128,k2=192,stride=1,size_w=7,size_h=1,name ='LKN_3',freeze=False,xavier=True)
    ##fire5 =self.SR_layer(fire5,k1=128,k2=192,size=7,stride=1,name='SR_3',freeze=False,xavier=True)

    fire10_fuse = tf.add(fire10, fire5, name='fure10_fuse')  #64*64
    fire10_fuse = tf.concat([fire10_fuse,fire_ASPP],3,name='concat_64x64')
    
    fire11 = self._fire_deconv(
        'fire_deconv11', fire10_fuse, s1x1=64, e1x1=64, e3x3=64,factors=[1, 2],
        stddev=0.1)
    #--R_sub_3_11 = tf.subtract(fire3,fire11,name='fire3-11')
    #--fire11_fuse = tf.concat([fire11,R_sub_3_11],3,name='sub_concat_3')

    #fire3 =self.LKN_layer(fire3,k1=64,k2=128,stride=1,size_w=7,size_h=1,name ='LKN_4',freeze=False,xavier=True)
    #fire3 =self.SR_layer(fire3,k1=128,k2=128,size=7,stride=1,name='SR_4',freeze=False,xavier=True)

    fire11_fuse = tf.add(fire11, fire3, name='fire11_fuse') #64x128

    
    #conv1 =self.LKN_layer(conv1,k1=64,k2=64,stride=1,size_w=7,size_h=1,name ='LKN_5',freeze=False,xavier=True)
    #conv1 =self.SR_layer(conv1,k1=64,k2=64,size=7,stride=1,name='SR_5',freeze=False,xavier=True)
    #fire11_concat = tf.add(fire11_fuse,concat_layer_4_1,name='fire11_concat')
    fire12 = self._fire_deconv(
        'fire_deconv12', fire11_fuse, s1x1=16, e1x1=32, e3x3=32, factors=[1, 2],
        stddev=0.1)
    #--R_sub_1_12 = tf.subtract(conv1,fire12,name='fire1-12')
    #--fire12_fuse = tf.concat([fire12,R_sub_1_12],3,name='sub_concat_1')
    
    
    fire12_fuse = tf.add(fire12, conv1, name='fire12_fuse') #64x64x256
    
    #fire12_concat = tf.add(fire12_fuse,concat_layer_2_1,name='fire12_concat')
    
    fire13 = self._fire_deconv(
        'fire_deconv13', fire12_fuse, s1x1=16, e1x1=32, e3x3=32, factors=[1, 2],
        stddev=0.1)
    #R_sub_0_13 = tf.subtract(conv1_skip,fire13,name='skip-13')
    #fire13_fuse = tf.concat([fire13,R_sub_0_13],3,name='sub_concat_0')
    fire13_fuse = tf.add(fire13, conv1_skip, name='fire13_fuse')#64 x512

    #fire13_concat = tf.concat([fire13_fuse,concat_layer_4_1],3,name='fire13_concat')
    drop13 = tf.nn.dropout(fire13_fuse, self.keep_prob, name='drop13')

    output_prob = self._conv_layer(
        'conv14_prob', drop13, filters=mc.NUM_CLASS, size=3, stride=[1,1],
        padding='SAME', relu=False, stddev=0.1)
    '''
    bilateral_filter_weights = self._bilateral_filter_layer(
        'bilateral_filter', self.lidar_input[:, :, :, :3], # x, y, z
        thetas=[mc.BILATERAL_THETA_A, mc.BILATERAL_THETA_R],
        sizes=[mc.LCN_HEIGHT, mc.LCN_WIDTH], stride=1)

    output_prob = self._recurrent_crf_layer(
        'recurrent_crf', conv14, bilateral_filter_weights, 
        sizes=[mc.LCN_HEIGHT, mc.LCN_WIDTH], num_iterations=mc.RCRF_ITER,
        padding='SAME'
    )'''
    return output_prob
  def Squeeze_ex_layer(self,input_x,out_dims,ratio,name):
        with tf.name_scope(name):
        # apply global average pooling
             squeeze = tf.reduce_mean(input_x, [1, 2], name=name+'/image_level_global_pool', keep_dims=False)
             ex = self._fc_layer(name+'fully_connected1',squeeze,out_dims/ratio,flatten=False,relu=True)
             ex = self._fc_layer(name+'fully_connected2',ex,out_dims,flatten=False,relu=False)
             ex = tf.nn.sigmoid(ex,name=name+'sigmod')
             ex = tf.reshape(ex,[-1,1,1,out_dims])
             scale = input_x *ex
             return scale
                
  def depthwise_separabel_conv(inputs,num_filters,width_multiplier,Relu=True,downsample = False,scope_nmae='Depthwise_Conv'):
    ds = 2 if downsample else 1
    dwc =slim.separable_conv2d(inputs,  
                                    num_outputs=None,  
                                    kernel_size=[3, 3],  
                                    stride = ds,  
                                    depth_multiplier=1,  
                                    activation_fn=None,  
                                    scope=scope_name+'/depthwise_conv')  
    bn = slim.batch_norm(dwc, scope=scope_name+'/dw_bn')  
    if Relu:
        out = tf.nn.relu(bn, name=scope_name+'/dw_relu') 
        return out
    else:
        return bn
  def _conv_bn_splayer(
      self, inputs, layer_name, filters,channel_multipl,
      size, stride, padding='SAME',rate=[1,1], freeze=False, relu=True,
      xavier=False, stddev=0.001,bias_init_val=0.0):
    """ Convolution + BatchNorm + [relu] layer. Batch mean and var are treated
    as constant. Weights have to be initialized from a pre-trained model or
    restored from a checkpoint.

    Args:
      inputs: input tensor
      conv_param_name: name of the convolution parameters
      bn_param_name: name of the batch normalization parameters
      scale_param_name: name of the scale parameters
      filters: number of output filters.
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
      freeze: if true, then do not train the parameters in this layer.
      xavier: whether to use xavier weight initializer or not.
      relu: whether to use relu or not.
      conv_with_bias: whether or not add bias term to the convolution output.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A convolutional layer operation.
    """
    mc = self.mc
    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        kernel_val = np.transpose(cw[layer_name][0], [2,3,1,0])
        bias_val = cw[layer_name][1]
        # check the shape
        if (kernel_val.shape == 
              (size, size, inputs.get_shape().as_list()[-1], filters)) \
           and (bias_val.shape == (filters, )):
          use_pretrained_param = True
        else:
          print ('Shape of the pretrained parameter of {} does not match, '
              'use randomly initialized parameter'.format(layer_name))
      else:
        print ('Cannot find {} in the pretrained model. Use randomly initialized '
               'parameters'.format(layer_name))

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      channels = inputs.get_shape().as_list()[3]
      #print (inputs.get_shape().as_list()[3])
      # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      # shape [h, w, in, out]
      if use_pretrained_param:
        if mc.DEBUG_MODE:
          print ('Using pretrained model for {}'.format(layer_name))
        kernel_init = tf.constant(kernel_val , dtype=tf.float32)
        bias_init = tf.constant(bias_val, dtype=tf.float32)
      elif xavier:
        kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(bias_init_val)
      else:
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(bias_init_val)

      # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      # shape [h, w, in, out]
      depth_kernel = _variable_with_weight_decay(
          'depth_kernels', shape=[size, size, channels,channel_multipl ],
          wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))
      point_wise_kernel = _variable_with_weight_decay(
          'point_kernel', shape=[1, 1, int(channels)*channel_multipl,filters],
          wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))
      biases = _variable_on_device('biases', [filters], bias_init, 
                                trainable=(not freeze))
      self.model_params += [depth_kernel,point_wise_kernel,biases]
   

      conv = tf.nn.separable_conv2d(
          inputs, depth_kernel,point_wise_kernel, [1, stride[0], stride[1], 1], padding=padding,
          name='separable',rate=rate)
     
      conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')

      #gamma_val  = tf.constant_initializer(1.0)
      #beta_val   = tf.constant_initializer(0.0)
      #gamma = _variable_on_device('gamma', [filters], gamma_val,trainable=(not freeze))
      #beta  = _variable_on_device('beta', [filters], beta_val,trainable=(not freeze))
      train = tf.cast((not freeze),tf.bool)
      #conv_bias = self.Batch_Normalization(conv_bias,training=train,scope=layer_name+'_BN')
      #conv_bias=self.batch_norm(conv_bias,beta,gamma,train)
      conv_bias=self.batch_norm2(conv_bias,filters,train)
      if relu:
        out = tf.nn.relu(conv_bias, 'relu')
      else:
        out = conv_bias

      self.model_size_counter.append(
          (layer_name, (1+size*size*int(channels))*filters)
      )
      out_shape = out.get_shape().as_list()
      num_flops = \
        (1+2*int(channels)*size*size)*filters*out_shape[1]*out_shape[2]
      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]
      self.flop_counter.append((layer_name, num_flops))

      self.activation_counter.append(
          (layer_name, out_shape[1]*out_shape[2]*out_shape[3])
      )

      return out
  def atrous_spatial_pyramid_pooling(self,net, scope, depth=128):



    with tf.variable_scope(scope):

        feature_map_size = tf.shape(net)
      # shape = net.get_shape().as_list()



        # apply global average pooling

        image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keep_dims=True)

        image_level_features = self._conv_layer("image_level_conv_1x1",image_level_features, depth, size=1,stride=[1,1],padding='SAME',dilations=[1,1,1,1],freeze=False,relu=True,BN=False)

        image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1], feature_map_size[2]))
        at_pool1x1 =  self._conv_layer("conv_1x1_0",net, depth,size=1,stride=[1,1],padding='SAME',dilations=[1,1,1,1],freeze=False,relu=True,BN=False)
     #   tf.summary.image('at_pool1x1',tf.slice(at_pool1x1,[0,0,0,0],[shape[0],shape[1],shape[2],3]),max_outputs=1)



        at_pool3x3_1 =  self._conv_layer("conv_3x3_1",net, depth,size=3,stride=[1,1],padding='SAME',dilations=[1,6,6,1],freeze=False,relu=True,BN=False)
      #  tf.summary.image('at_pool3x3_1',tf.slice(at_pool3x3_1,[0,0,0,0],[shape[0],shape[1],shape[2],3]),max_outputs=1)


        at_pool3x3_2 = self._conv_layer('conv_3x3_2',net, depth, size=3,stride=[1,1],padding='SAME',dilations=[1,9,9,1],freeze=False,relu=True,BN=False)
    #    tf.summary.image('at_pool3x3_2',tf.slice(at_pool3x3_2,[0,0,0,0],[shape[0],shape[1],shape[2],3]),max_outputs=1)

        at_pool3x3_3 = self._conv_layer('conv_3x3_3',net, depth, size=3,stride=[1,1],padding='SAME',dilations=[1,12,12,1],freeze=False,relu=True,BN=False)

     #   tf.summary.image('at_pool3x3_3',tf.slice(at_pool3x3_3,[0,0,0,0],[shape[0],shape[1],shape[2],3]),max_outputs=1)

        net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3,

                        name="concat")

        net = self._conv_layer('conv_1x1_output',net, depth, size=1,stride=[1,1],padding='SAME',dilations=[1,1,1,1],freeze=False,relu=True,BN=False)
        #net = self.Squeeze_ex_layer(net,depth,4,name=scope+'/SE')
        return net
  def batch_norm2(self,x, size, training, decay=0.999):
    #mean_val   = tf.constant_initializer(0.0)
    #var_val    = tf.constant_initializer(1.0)
    #gamma_val  = tf.constant_initializer(1.0)
    #beta_val   = tf.constant_initializer(0.0)
    beta = tf.Variable(tf.zeros([size]), name='beta')
    scale = tf.Variable(tf.ones([size]), name='scale')
    pop_mean = tf.Variable(tf.zeros([size]),'mean')
    pop_var = tf.Variable(tf.ones([size]),'var')
    epsilon = 1e-3
    self.model_params+=[beta,scale,pop_mean,pop_var]
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
    train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
    train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
 
    def batch_statistics():
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon, name='batch_norm')
 
    def population_statistics():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon, name='batch_norm')
 
    return tf.cond(training, batch_statistics, population_statistics)
  def batch_norm1(self,x,beta,gamma,phase_train,scope='bn',decay=0.9,eps=1e-5):
    with tf.variable_scope(scope):
        # beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0), trainable=True)
        # gamma = tf.get_variable(name='gamma', shape=[n_out],
        #                         initializer=tf.random_normal_initializer(1.0, stddev), trainable=True)
        batch_mean,batch_var = tf.nn.moments(x,[0,1,2],name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean,batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean),tf.identity(batch_var)
 
        mean,var = tf.cond(phase_train,
                           mean_var_with_update,
                           lambda: (ema.average(batch_mean),ema.average(batch_var)))
        self.model_params+=[beta,gamma]

        normed = tf.nn.batch_normalization(x, mean,var,beta,gamma,eps)
        return normed
  def Batch_Normalization(self,x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :

        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))
    
    
  def _squzee_(self,input_tensor,mc,name):

    block1_0 = self._conv_layer('conv1',input_tensor,filters=64,size =3,stride=[1,2],padding='SAME',dilations=[1,1,1,1],freeze=False,relu=True)
    ##block1_0_skip = self._conv_layer('conv1_skip',input_tensor,filters=32,size =3,stride=[1,1],padding='SAME',dilations=[1,1,1,1],freeze=False,relu=True)
    shape = block1_0.get_shape().as_list()
    #Block-1 in-64x256

    #tf.summary.image('block1_0',tf.slice(block1_0,[0,0,0,0],[shape[0],shape[1],shape[2],3]),max_outputs=1)
    block1_1 = self._conv_layer('block1_conv_1',block1_0,filters=64,size =3,stride=[1,1],padding='SAME',dilations=[1,1,1,1],freeze=False,relu=True)
    block1_residual = self._conv_layer('block1_conv_2',block1_1,filters=128,size =1,stride=[1,2],padding='SAME',dilations=[1,1,1,1],freeze=False,relu=False)
    #tf.summary.image('block1_residual',tf.slice(block1_residual,[0,0,0,0],[shape[0],int(shape[1]/2),int(shape[2]/2),3]),max_outputs=1)

    #Block-2 in-64x128
    
    block2 =self._conv_bn_splayer(block1_1,'block2_conv1',filters=128,channel_multipl=1,size=3,stride=[1,1],padding='SAME',rate=[1,1],freeze=False,relu=True,stddev=0.001)
   # tf.summary.image('block2',tf.slice(block2,[0,0,0,0],[shape[0],int(shape[1]/2),int(shape[2]/2),3]),max_outputs=6)

    block2_1 =self._conv_bn_splayer(block2,'block2_conv2',filters=128,channel_multipl=1,size=3,stride=[1,1],padding='SAME',rate=[1,1],freeze=False,relu=True,stddev=0.001)
    #tf.summary.image('block2_1',tf.slice(block2_1,[0,0,0,0],[shape[0],int(shape[1]/2),int(shape[2]/2),3]),max_outputs=6)

    #block2_2 =self._conv_bn_splayer(block2_1,'block2_conv_3',filters=128,channel_multipl=1,
    #                             size=3,stride=[2,2],padding='SAME',rate=[1,1],freeze=False,relu=True,stddev=0.001)
    block2_2 = self._conv_layer('block2_conv_3',block2_1,filters=128,size =1,stride=[1,2],padding='SAME',dilations=[1,1,1,1],freeze=False,relu=False)
   # tf.summary.image('block2_2',tf.slice(block2_2*10,[0,0,0,0],[shape[0],int(shape[1]/4),int(shape[2]/4),3]),max_outputs=4)

    
    block_ruse_1 = tf.add(block2_2,block1_residual,name='add_1')
    shape=block_ruse_1.get_shape().as_list()

    #tf.summary.image('block_ruse_1',tf.slice(block_ruse_1,[0,0,0,0],[shape[0],int(shape[1]),int(shape[2]),3]),max_outputs=4)

    block2_residual = self._conv_layer('block2_conv_4',block_ruse_1,filters=256,size =1,stride=[1,2],padding='SAME',dilations=[1,1,1,1],freeze=False,relu=False)
    shape=block2_residual.get_shape().as_list()

   # tf.summary.image('block2_residual',tf.slice(block2_residual,[0,0,0,0],[shape[0],int(shape[1]),int(shape[2]),3]),max_outputs=4)

    #Block_3 in 16x64
    
    #block3 =tf.nn.relu(block_ruse_1,name ='block3_relu')
    block3 = self._conv_bn_splayer(block_ruse_1,'block3_conv_0',filters=256,channel_multipl=1,
                                 size=3,stride=[1,1],padding='SAME',rate=[1,1],
                                  freeze=False,relu=True,stddev=0.001)
    
    block3_1 = self._conv_bn_splayer(block3,'block3_conv_1',filters=256,channel_multipl=1,
                                 size=3,stride=[1,1],padding='SAME',rate=[1,1],
                                  freeze=False,relu=True,stddev=0.001)
    block3_2 = self._conv_layer('block3_conv_2',block3_1,filters=256,
                                 size=3,stride=[1,2],padding='SAME',
                                  freeze=False,relu=True,stddev=0.001)
    #tf.summary.image('block3_2',tf.slice(block3_2,[0,0,0,0],[shape[0],int(shape[1]/8),int(shape[2]/8),3]),max_outputs=16)
    block_ruse_2 =tf.add(block3_2,block2_residual,name='add_2')
    #block3_residual = self._conv_layer('block3_conv_3',block_ruse_2,filters=512,size =1,stride=[1,2],padding='SAME',dilations=[1,1,1,1],freeze=False,relu=False,stddev=0.001)
    '''
    #Block_4 in 8x64
    block4 = self._conv_bn_splayer(block_ruse_2,'block4_conv_0',filters=512,channel_multipl=1,
                                 size=3,stride=[1,1],padding='SAME',rate=[2,2],
                                  freeze=False,relu=True,stddev=0.001)
    
    block4_1 = self._conv_bn_splayer(block4,'block4_conv_1',filters=512,channel_multipl=1,
                                 size=3,stride=[1,1],padding='SAME',rate=[2,2],
                                  freeze=False,relu=True,stddev=0.001)
    block4_2 = self._conv_layer('block4_conv_2',block4_1,filters=512,
                                 size=3,stride=[1,2],padding='SAME',
                                  freeze=False,relu=True,stddev=0.001)
    #tf.summary.image('block4_2',tf.slice(block4_2,[0,0,0,0],[shape[0],int(shape[1]/8),int(shape[2]/8),3]),max_outputs=3)

    block_ruse_3 =tf.add(block4_2,block3_residual,name='add_3')
    #block4_residual = self._conv_layer('block4_conv_3',block_ruse_2,filters=512,size =1,stride=2,padding='SAME',dilations=[1,1,1,1],freeze=False,relu=False,stddev=0.001)
    '''
    ASPP=self.atrous_spatial_pyramid_pooling(block_ruse_2,scope='ASPP')
    #deconv 
    shape=ASPP.get_shape().as_list()
    #tf.summary.image('ASPP_01',tf.slice(ASPP,[0,0,0,0],[shape[0],int(shape[1]),int(shape[2]),3]),max_outputs=3)
    deconv_ASPP_0 = self._conv_layer('conv0_ASPP',ASPP, filters=256, size=1, stride=[1,1],padding='SAME', dilations=[1,1,1,1],BN=False,freeze=False, xavier=True)
   
    deconv_ASPP_01 = self._deconv_layer('deconv_1', deconv_ASPP_0, filters=128,size=[3,3],stride=[1,2])
    
    #deconv_ASPP_01 = self._deconv_layer('deconv_2', deconv_ASPP_01, filters=128,size=[3,3],stride=[1,2])
   # tf.summary.image('deconv_ASPP_01',tf.slice(deconv_ASPP_01,[0,0,0,0],[shape[0],int(shape[1]*2),int(shape[2]*2),3]),max_outputs=16)
    #deconv_ASPP_01 = tf.add(deconv_ASPP_01,block2_2,name='add_in_8')
    
    deconv_ASPP_1 = self._conv_layer('conv1_ASPP', block_ruse_1, filters=128, size=1, stride=[1,1],padding='SAME', dilations=[1,1,1,1],BN=False,freeze=False, xavier=True)
    
    deconv_concat = tf.concat([deconv_ASPP_1,deconv_ASPP_01],3,name='upsamplex4')
    
    deconv_ASPP_2 = self._conv_layer('conv2_ASPP', deconv_concat, filters=128, size=3, stride=[1,1],padding='SAME', dilations=[1,1,1,1],BN=False,freeze=False, xavier=True)
    
    deconv_ASPP_21 = self._deconv_layer('deconv_3', deconv_ASPP_2, filters=64,size=[3,3],stride=[1,2])
    #tf.summary.image('deconv_ASPP_21',tf.slice(deconv_ASPP_21,[0,0,0,0],[shape[0],int(shape[1]*2),int(shape[2]*2),3]),max_outputs=3)
    deconv_ASPP_21 = tf.concat((deconv_ASPP_21,block1_1),3,name='concat_4')
    #tf.summary.image('output',tf.slice(deconv_ASPP_21,[0,0,0,0],[shape[0],int(shape[1]*2),int(shape[2]*4),3]),max_outputs=6)
    
    deconv_ASPP_3 = self._conv_layer('conv3_ASPP', deconv_ASPP_21, filters=64, size=3, stride=[1,1],padding='SAME', dilations=[1,1,1,1],BN=False,freeze=False, xavier=True)
    deconv_ASPP_31 = self._deconv_layer('deconv_4', deconv_ASPP_3, filters=32, size=[3,3],stride=[1,2])
    #deconv_ASPP_31 = tf.add(deconv_ASPP_31,block1_0_skip,name='add_in_ori')
    #tf.summary.image('output_map_deconv',tf.slice(deconv_ASPP_31,[0,0,0,0],[shape[0],int(shape[1]*),int(shape[2]*8),3]),max_outputs=6)
    
    drop13 = tf.nn.dropout(deconv_ASPP_31, self.keep_prob, name='drop13')
    
    conv14 = self._conv_layer(
        'conv14_prob', drop13, filters=mc.NUM_CLASS, size=3, stride=[1,1],
        padding='SAME',BN=True, relu=False, stddev=0.1)
  #  tf.summary.image('output_pro_without_rcf',tf.slice(conv14,[0,0,0,0],[shape[0],int(shape[1]),int(shape[2]*8),3]),max_outputs=6)

    bilateral_filter_weights = self._bilateral_filter_layer(
        'bilateral_filter', self.lidar_input[:, :, :, :3], # x, y, z
        thetas=[mc.BILATERAL_THETA_A, mc.BILATERAL_THETA_R],
        sizes=[mc.LCN_HEIGHT, mc.LCN_WIDTH], stride=1)

    output_prob = self._recurrent_crf_layer(
        'recurrent_crf', conv14, bilateral_filter_weights, 
        sizes=[mc.LCN_HEIGHT, mc.LCN_WIDTH], num_iterations=mc.RCRF_ITER,
        padding='SAME'
    )
    shape = output_prob.get_shape().as_list()
   # tf.summary.image('output_pro',tf.slice(output_prob,[0,0,0,0],[shape[0],int(shape[1]),int(shape[2]),3]),max_outputs=6)
    return output_prob

      
  def _squzee_down_8_without_input_down(self,input_tensor,mc,name):
        
        
        
    conv1 = self._conv_layer(
       name+'_'+'conv1', input_tensor, filters=64, size=3, stride=2,
        padding='SAME', freeze=False, xavier=True)
    conv1_skip = self._conv_layer(
        name+'_'+'conv1_skip', input_tensor, filters=64, size=1, stride=1,
        padding='SAME', freeze=False, xavier=True)
    pool1 = self._pooling_layer(
        name+'_'+'pool1', conv1, size=3, stride=2, padding='SAME')

    fire2 = self._fire_layer(
        name+'_'+'fire2', pool1, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    fire3 = self._fire_layer(
        name+'_'+'fire3', fire2, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    pool3 = self._pooling_layer(
        name+'_'+'pool3', fire3, size=3, stride=2, padding='SAME')

    fire4 = self._fire_layer(
        name+'_'+'fire4', pool3, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    fire5 = self._fire_layer(
        name+'_'+'fire5', fire4, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    fire6 = self._fire_layer(
        name+'_'+'fire8', fire5, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    
    # Deconvolation
    fire10 = self._fire_deconv(
        name+'_'+'fire_deconv10', fire6, s1x1=48, e1x1=64, e3x3=64, factors=[1, 2],
        stddev=0.1)
    fire10_fuse = tf.add(fire10, fire3, name=name+'_'+'fure10_fuse')

    fire11 = self._fire_deconv(
        name+'_'+'fire_deconv11', fire10_fuse, s1x1=32, e1x1=32, e3x3=32, factors=[1, 2],
        stddev=0.1)
    fire11_fuse = tf.add(fire11, conv1, name=name+'_'+'fire11_fuse')
    fire12 = self._fire_deconv(
        name+'_'+'fire_deconv12', fire11_fuse, s1x1=24, e1x1=32, e3x3=32, factors=[1, 2],
        stddev=0.1)
    fire12_fuse = tf.add(fire12, conv1_skip, name=name+'_'+'fire12_fuse')
    
    return fire12_fuse
  
 
  def _squzee_down_16_without_input_down(self,input_tensor,mc,name):
        
        
        
    conv1 = self._conv_layer(
       name+'_'+'conv1', input_tensor, filters=64, size=3, stride=2,
        padding='SAME', freeze=False, xavier=True)
    fire2 = self._fire_layer(
        name+'_'+'fire2', conv1, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    fire3 = self._fire_layer(
        name+'_'+'fire3', fire2, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    conv1_skip = self._conv_layer(
        name+'_'+'conv1_skip', input_tensor, filters=64, size=1, stride=1,
        padding='SAME', freeze=False, xavier=True)
    pool1 = self._pooling_layer(
        name+'_'+'pool1', fire3, size=3, stride=2, padding='SAME')
    
    fire4 = self._fire_layer(
        name+'_'+'fire4', pool1, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    fire5 = self._fire_layer(
        name+'_'+'fire5', fire4, s1x1=32, e1x1=128, e3x3=128, freeze=False)

    
    # Deconvolation
    fire10 = self._fire_deconv(
        name+'_'+'fire_deconv10', fire5, s1x1=48, e1x1=64, e3x3=64, factors=[1, 2],
        stddev=0.1)
    fire10_fuse = tf.add(fire10, fire3, name=name+'_'+'fure10_fuse')

    fire11 = self._fire_deconv(
        name+'_'+'fire_deconv11', fire10_fuse, s1x1=32, e1x1=32, e3x3=32, factors=[1, 2],
        stddev=0.1)
    fire11_fuse = tf.add(fire11, conv1_skip, name=name+'_'+'fire11_fuse')
    
    return fire11_fuse

  def LKN_layer(self,input_GC,k1,k2,size_w,size_h,stride,name,freeze,xavier,relu=False):
        k_one_conv_input=self._conv_layer_k(name+'conv1_kx1',input_GC,k1,size_w,size_h,stride=stride,padding='SAME',freeze=freeze,xavier=xavier,relu=relu)
        K_ONE_conv=self._conv_layer_k(name+'conv1_1xk',k_one_conv_input,k2,size_h,size_w,stride=stride,padding='SAME',freeze=freeze,xavier=xavier,relu=relu)
        
        one_k_conv_input= self._conv_layer_k(name+'conv2_1xk',input_GC,k1,size_h,size_w,stride=stride,padding='SAME',freeze=freeze,xavier=xavier,relu=relu)
        ONE_conv_input = self._conv_layer_k(name+'conv2_kx1',one_k_conv_input,k2,size_w,size_h,stride=stride,padding='SAME',freeze=freeze,xavier=xavier,relu=relu)
        out = tf.add(K_ONE_conv,ONE_conv_input,name=name+'lkn')
        print ('1_'+name,k_one_conv_input.get_shape().as_list())
        print ('2_'+name,K_ONE_conv.get_shape().as_list())
        print ('3_'+name,one_k_conv_input.get_shape().as_list())
        print ('4_'+name,ONE_conv_input.get_shape().as_list())

        return out
        
  def SR_layer(self,input_sr,k1,k2,size,stride,name,freeze,xavier):
    #assert input_sr.get_shape()[3]==k2
    #shape refinement layer
    high_way = self._conv_layer(name+'highway_1',input_sr,filters=k1,size=size,stride=stride,padding='SAME', freeze=freeze,xavier=xavier,relu=True)
    high_way_2=self._conv_layer(name+'highway_2',high_way,filters=k2,size=size,stride=stride,padding='SAME', freeze=freeze,xavier=xavier,relu=False)
    out =tf.add(input_sr,high_way_2,name=name+'SR')
    print (name,out.get_shape().as_list())

    return out
    
        
  def _bilateral_filter_layer(
      self, layer_name, inputs, thetas=[0.9, 0.01], sizes=[3, 5], stride=1,
      padding='SAME'):
    """Computing pairwise energy with a bilateral filter for CRF.

    Args:
      layer_name: layer name
      inputs: input tensor with shape [batch_size, zenith, azimuth, 2] where the
          last 2 elements are intensity and range of a lidar point.
      thetas: theta parameter for bilateral filter.
      sizes: filter size for zenith and azimuth dimension.
      strides: kernel strides.
      padding: padding.
    Returns:
      out: bilateral filter weight output with size
          [batch_size, zenith, azimuth, sizes[0]*sizes[1]-1, num_class]. Each
          [b, z, a, :, cls] represents filter weights around the center position
          for each class.
    """

    assert padding == 'SAME', 'currently only supports "SAME" padding stategy'
    assert stride == 1, 'currently only supports striding of 1'
    assert sizes[0] % 2 == 1 and sizes[1] % 2 == 1, \
        'Currently only support odd filter size.'

    mc = self.mc
    theta_a, theta_r = thetas
    size_z, size_a = sizes
    pad_z, pad_a = size_z//2, size_a//2
    half_filter_dim = (size_z*size_a)//2
    batch, zenith, azimuth, in_channel = inputs.shape.as_list()

    # assert in_channel == 1, 'Only support input channel == 1'

    with tf.variable_scope(layer_name) as scope:
      condensing_kernel = tf.constant(
          util.condensing_matrix(size_z, size_a, in_channel),
          dtype=tf.float32,
          name='condensing_kernel'
      )

      condensed_input = tf.nn.conv2d(
          inputs, condensing_kernel, [1, 1, stride, 1], padding=padding,
          name='condensed_input'
      )

      # diff_intensity = tf.reshape(
      #     inputs[:, :, :], [batch, zenith, azimuth, 1]) \
      #     - condensed_input[:, :, :, ::in_channel]

      diff_x = tf.reshape(
          inputs[:, :, :, 0], [batch, zenith, azimuth, 1]) \
              - condensed_input[:, :, :, 0::in_channel]
      diff_y = tf.reshape(
          inputs[:, :, :, 1], [batch, zenith, azimuth, 1]) \
              - condensed_input[:, :, :, 1::in_channel]
      diff_z = tf.reshape(
          inputs[:, :, :, 2], [batch, zenith, azimuth, 1]) \
              - condensed_input[:, :, :, 2::in_channel]

      bi_filters = []
      for cls in range(mc.NUM_CLASS):
        theta_a = mc.BILATERAL_THETA_A[cls]
        theta_r = mc.BILATERAL_THETA_R[cls]
        bi_filter = tf.exp(-(diff_x**2+diff_y**2+diff_z**2)/2/theta_r**2)
        bi_filters.append(bi_filter)
      out = tf.transpose(
          tf.stack(bi_filters),
          [1, 2, 3, 4, 0],
          name='bilateral_filter_weights'
      )

    return out

  def _activation_summary(self, x, layer_name):
    """Helper to create summaries for activations.

    Args:
      x: layer output tensor
      layer_name: name of the layer
    Returns:
      nothing
    """
    with tf.variable_scope('activation_summary') as scope:
      tf.summary.histogram(layer_name, x)
      tf.summary.scalar(layer_name+'/sparsity', tf.nn.zero_fraction(x))
      tf.summary.scalar(layer_name+'/average', tf.reduce_mean(x))
      tf.summary.scalar(layer_name+'/max', tf.reduce_max(x))
      tf.summary.scalar(layer_name+'/min', tf.reduce_min(x))
