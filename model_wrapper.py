from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

from preprocessing import preprocess_utility as ult
from datasets.imagenet_dataset import ImagenetData
# from datasets import imagenet_dataset
from models import mobilenet_model
from models import vgg_model


FLAGS = tf.app.flags.FLAGS

IMAGE_SIZE = ult.IMAGE_SIZE
NUM_CLASSES = ult.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = ult.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = ult.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 1.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.75  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
TOWER_NAME = 'tower'

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  # data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')

  dataset_train = ImagenetData(subset='train')
  dataset_test = ImagenetData(subset='validation')

  assert dataset_train.data_files()
  assert dataset_test.data_files()

  imgs_train, labels_train= ult.distorted_inputs(
    dataset_train,
    isTrain=True,
    batch_size=FLAGS.batch_size,
    num_preprocess_threads=FLAGS.num_preprocess_threads)

  imgs_test, labels_test = ult.inputs(
    dataset_test,
    batch_size=FLAGS.batch_size,
    num_preprocess_threads=FLAGS.num_preprocess_threads)

  return (imgs_train, labels_train, imgs_test, labels_test)
  # return ult.distorted_inputs(
  #   dataset,
  #   isTrain,
  #   batch_size=FLAGS.batch_size,
  #   num_preprocess_threads=FLAGS.num_preprocess_threads)

def inference(images, isTrain, isLoad):
  """Build the vggnet model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  if FLAGS.model_name == 'vggnet':
    model = vgg_model.vggnet(isLoad, isTrain)
  elif FLAGS.model_name == 'mobilenet':
    model = mobilenet_model.mobilenet(isLoad, isTrain)
  keep_prob = tf.cond(isTrain, lambda: 0.5, lambda: 1.0)
  pred = model.conv_network(images, keep_prob)
  return pred

def eval(logits, labels):
  labels = tf.cast(labels, tf.int64)
  predictions = tf.argmax(logits, 1)

  # top5_acc = tf.metrics.recall_at_k(
  #     labels = labels,
  #     predictions = logits,
  #     k = 5
  # )
  # acc = tf.metrics.accuracy(
  #     labels = labels,
  #     predictions = predictions
  # )

  acc = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
  top5_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32))
  return (acc, top5_acc)

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  # labels = tf.cast(labels, tf.float32)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def pickle_save(sess):
    if FLAGS.model_name == 'vggnet':
        vgg_model.save_model(sess)
    elif FLAGS.model_name == 'mobilenet':
        mobilenet_model.save_model(sess)


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op
