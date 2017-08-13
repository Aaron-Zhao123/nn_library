from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
import tensorflow.contrib.slim as slim
import tensorflow as tf

def vgg_arg_scope(weight_decay=0.0005):
    """Defines the VGG arg scope.
    Args:
    weight_decay: The l2 regularization coefficient.
    Returns:
    An arg_scope.
    """
    with arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=nn_ops.relu,
            weights_regularizer=regularizers.l2_regularizer(weight_decay),
            biases_initializer=init_ops.zeros_initializer()):
        with arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc

def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):
    """Oxford Net VGG 16-Layers version D Example.
    Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
    Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    Returns:
    the last op containing the log predictions and end_points dict.
    """
  # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        with slim.arg_scope([slim.conv2d],
                        stride=1, padding='SAME'):
                        # stride=1, padding='SAME',
                        # normalizer_params=batch_norm_params):
            net = inputs
            net = slim.repeat(
                net, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected layers.
            net = slim.fully_connected(net, 4096, scope='fc6')
            net = slim.dropout(
                net, dropout_keep_prob, is_training=is_training, scope='dropout6')
            net = slim.fully_connected(net, 4096, scope='fc7')
            net = slim.dropout(
                net, dropout_keep_prob, is_training=is_training, scope='dropout7')
            net = slim.fully_connected(
                net,
                num_classes,
                activation_fn=None,
                scope='fc8')
    return net

vgg_16.default_image_size = 224
