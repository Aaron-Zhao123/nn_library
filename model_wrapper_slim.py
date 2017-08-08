from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf
import tensorflow.contrib.slim as slim

from preprocessing import preprocess_utility as ult
from datasets.imagenet_dataset import ImagenetData
# from datasets import imagenet_dataset
from models import vgg_model_slim

FLAGS = tf.app.flags.FLAGS
TOWER_NAME = 'tower'

# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999

num_examples_per_epoch = 1281167

def inference(images, num_classes, for_training=False, restore_logits=True,
              scope=None):

    # Parameters for BatchNorm.
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
    }
  # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.fc], weight_decay=0.00004):
        with slim.arg_scope([slim.conv2d],
                            stddev=0.1,
                            activation=tf.nn.relu,
                            batch_norm_params=batch_norm_params):
            logits, endpoints = vgg_model_slim.vgg16(
                images,
                dropout_keep_prob=0.5,
                num_classes=num_classes,
                is_training=for_training,
                restore_logits=restore_logits,
                scope=scope)

    # Add summaries for viewing model statistics on TensorBoard.
    _activation_summaries(endpoints)

    # Grab the logits associated with the side head. Employed during training.
    auxiliary_logits = endpoints['aux_logits']
    return logits, auxiliary_logits


def loss(logits, labels, batch_size=None):
    if not batch_size:
        batch_size = FLAGS.batch_size

    # Reshape the labels into a dense Tensor of
    # shape [FLAGS.batch_size, num_classes].
    sparse_labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
    concated = tf.concat(axis=1, values=[indices, sparse_labels])
    num_classes = logits[0].get_shape()[-1].value
    dense_labels = tf.sparse_to_dense(concated,
                                    [batch_size, num_classes],
                                    1.0, 0.0)
    # Cross entropy loss for the main softmax prediction.
    slim.losses.cross_entropy_loss(logits[0],
                                 dense_labels,
                                 label_smoothing=0.1,
                                 weight=1.0)
    # Cross entropy loss for the auxiliary softmax head.
    slim.losses.cross_entropy_loss(logits[1],
                                 dense_labels,
                                 label_smoothing=0.1,
                                 weight=0.4,
                                 scope='aux_loss')
def distorted_inputs(num_preprocess_threads):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    dataset_train = ImagenetData(subset='train')
    dataset_test = ImagenetData(subset='validation')
    assert dataset_train.data_files()
    assert dataset_test.data_files()

    imgs_train, labels_train= ult.distorted_inputs(
        dataset_train,
        isTrain=True,
        batch_size=FLAGS.batch_size,
        num_preprocess_threads=num_preprocess_threads)

    imgs_test, labels_test = ult.inputs(
        dataset_test,
        batch_size=FLAGS.batch_size,
        num_preprocess_threads=num_preprocess_threads)

    return (imgs_train, labels_train, imgs_test, labels_test)

def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
    x: Tensor
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _activation_summaries(endpoints):
    with tf.name_scope('summaries'):
        for act in endpoints.values():
            _activation_summary(act)
