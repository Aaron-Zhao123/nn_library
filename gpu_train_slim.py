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
import progressbar

import model_wrapper_slim
import sys

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/tmp',
                           """Path to the processed data, i.e. """
                           """TFRecord of Example protos.""")
tf.app.flags.DEFINE_string('train_dir', '/tmp',
                            """Directory where to write event logs """
                            """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_epochs', 90,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")
tf.app.flags.DEFINE_bool('is_train', True,
                            """whether to perform training or not.""")
tf.app.flags.DEFINE_bool('is_load', True,
                            """whether to load from a model or not.""")
tf.app.flags.DEFINE_bool('ckpt_save', False,
                            """whether to save a ckpt file.""")
tf.app.flags.DEFINE_bool('pickle_save', True,
                            """whether to save into a pickle file""")
tf.app.flags.DEFINE_string('model_name', 'vggnet',
                            """whether to save into a pickle file""")


def tower_loss(images, labels, isTrain, isLoad, scope, reuse_variables = None):
    """
    Args
        images: [batch_size, image_size, image_size, 3]
        labels: 1001 classes
        scope: tower_0, tower_1 ...
    """
    restore_logits = not FLAGS.fine_tune

    with tf.variable_scope(tf.get_variable_scope(), reuse = reuse_variables):
        logits = model_wrapper_slim.inference(images, num_classes,
                for_training=isTrain,
                restore_logits=restore_logits,
                scope = scope)

    split_batch_size = images.get_shape().as_list()[0]
    model_wrapper_slim.loss(logits, labels, batch_size = split_batch_size)

    losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
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
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

        # decay the learning rate
        num_batches_per_epoch = (model_wrapper_slim.num_examples_per_epoch /
                             FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch/FLAGS.batch_size)
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)
        # Create an optimizer that performs gradient descent.
        opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
                                    momentum=RMSPROP_MOMENTUM,
                                    epsilon=RMSPROP_EPSILON)
        assert FLAGS.batch_size % FLAGS.num_gpus == 0, (
        'Batch size must be divisible by number of GPUs')

        split_batch_size = int(FLAGS.batch_size / FLAGS.num_gpus)
        total_num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
        images_train, labels_train, images_test, labels_test = model_wrapper_slim.distorted_inputs(total_num_preprocess_threads)

        num_classes = 1001
        images_split = tf.split(axis=0,
                                num_or_size_splits=FLAGS.num_gpus,
                                value=images_train)
        labels_split = tf.split(axis=0,
                                num_or_size_splits=FLAGS.num_gpus,
                                value=images_test)

        tower_grads = []
        reuse_variables = None

        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d'%i):
                with tf.name_scope('%s_%d' % ('tower',i)) as scope:
                    with slim.arg_scope([slim.variables.variable], device = '/cpu:0'):
                        loss = tower_loss(
                                images_split,
                                labels_split,
                                isTrain,
                                isLoad,
                                scope,
                                reuse_variables = reuse_variables)
                    #reuse for the second tower
                    reuse_variables = True
                    # Retain the Batch Normalization updates operations only from the
                    # final tower. Ideally, we should grab the updates from all towers
                    # but these stats accumulate extremely fast so we can ignore the
                    # other stats from the other towers without significant detriment.
                    batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION,
                                                            scope)
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        variable_averages = tf.train.ExponentialMovingAverage(
            model_wrapper_slim.MOVING_AVERAGE_DECAY, global_step)
        variables_to_average = (tf.trainable_variables() +
                            tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)

        batchnorm_updates_op = tf.group(*batchnorm_updates)

        train_op = tf.group(apply_gradient_op, variables_averages_op,
                        batchnorm_updates_op)

        saver = tf.train.Saver(tf.global_variables())

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))

        sess.run(init)

        if FLAGS.pretrained_model_checkpoint_path:
            assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
            variables_to_restore = tf.get_collection(
                slim.variables.VARIABLES_TO_RESTORE)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
            (datetime.now(), FLAGS.pretrained_model_checkpoint_path))
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                    examples_per_sec, duration))

            # Save the model checkpoint periodically.
            if step % 5000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()