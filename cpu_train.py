from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorflow.contrib.slim as slim

from datasets import flowers
from preprocessing import preprocessing
import model_wrapper
import sys

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_string('data_dir', '/tmp/mydata',
                           """Path to the processed data, i.e. """
                           """TFRecord of Example protos.""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('isgpu', False,
                            """whether use gpus or not""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")

def cpu_loss(images, labels, scope):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # Get images and labels for CIFAR-10.
  # dataset = flowers.get_split(FLAGS.subset, FLAGS.data_dir)
  # provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
  # [images, labels] = provider.get(['image', 'label'])
  images = tf.cast(images, tf.float32)

  # Build inference Graph.
  logits = model_wrapper.inference(images)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  total_loss = model_wrapper.loss(logits, labels)

  # with tf.control_dependencies([loss_averages_op]):
  total_loss = tf.identity(total_loss)
  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.

      """gradient clipping"""
      def ClipIfNotNone(grad):
          if grad is None:
              return grad
          return tf.clip_by_value(grad, -1, 1)

      clipped_grads = ClipIfNotNone(g)

      expanded_g = tf.expand_dims(clipped_grads, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)
    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    opt = tf.train.AdamOptimizer(1e-4)

    # images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    # labels = tf.placeholder(tf.float32, shape=[1024])

    dataset = flowers.get_split(FLAGS.subset, FLAGS.data_dir)
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle = True,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size
    )
    [image, label] = provider.get(['image', 'label'])
    image = preprocessing.preprocess_image(image, 224, 224, is_training=True)
    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=4,
        capacity=5 * FLAGS.batch_size)


    with tf.variable_scope(tf.get_variable_scope()) as scope:
      loss = cpu_loss(images, labels, scope)
      # Calculate the gradients for the batch of data on this CIFAR tower.
      grads = opt.compute_gradients(loss)

    # Apply the gradients to adjust the shared variables.
    # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    train_op = opt.apply_gradients(grads)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())
    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)


    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))


      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  # if tf.gfile.Exists(FLAGS.train_dir):
  #   tf.gfile.DeleteRecursively(FLAGS.train_dir)
  # tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
