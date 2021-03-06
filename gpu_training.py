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

import model_wrapper
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
def tower_loss(scope, isTrain, isLoad):
  # Get images and labels.
  # images_train, labels_train, images_test, labels_test = model_wrapper.distorted_inputs()
  images_train, labels_train, images_test, labels_test = model_wrapper.distorted_inputs()

  images = tf.cond(isTrain, lambda:images_train, lambda:images_test)
  labels = tf.cond(isTrain, lambda:labels_train, lambda:labels_test)

  # Build inference Graph.
  logits = model_wrapper.inference(images, isTrain, isLoad)
  # logits_test = model_wrapper.inference(images_test, train=False)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = model_wrapper.loss(logits, labels)

  (test_acc, top5) = model_wrapper.eval(logits, labels)
  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  # loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % model_wrapper.TOWER_NAME, '', l.op.name)
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(loss_name +' (raw)', l)
    # tf.summary.scalar(loss_name, loss_averages.average(l))

  # with tf.control_dependencies([loss_averages_op]):
  total_loss = tf.identity(total_loss)
  return (total_loss, test_acc, top5)


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
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    isTrain = FLAGS.is_train
    isLoad = FLAGS.is_load

    isTrain_ph = tf.placeholder(tf.bool, shape =None, name="is_train")
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    WEIGHTS_DECAY = False
    if WEIGHTS_DECAY:
      num_batches_per_epoch = (model_wrapper.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                               FLAGS.batch_size)
      decay_steps = int(num_batches_per_epoch * model_wrapper.NUM_EPOCHS_PER_DECAY)

      # Decay the learning rate exponentially based on the number of steps.
      lr = tf.train.exponential_decay(model_wrapper.INITIAL_LEARNING_RATE,
                                      global_step,
                                      decay_steps,
                                      model_wrapper.LEARNING_RATE_DECAY_FACTOR,
                                      staircase=True)

      # Create an optimizer that performs gradient descent.
      RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
      RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
      RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.
      opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
                                    momentum=RMSPROP_MOMENTUM,
                                    epsilon=RMSPROP_EPSILON)
    #   opt = tf.train.GradientDescentOptimizer(lr)
    else:
      opt = tf.train.AdamOptimizer(1e-1)

    # Calculate the gradients for each model tower.
    tower_grads = []
    tower_top1_accs = []
    tower_top5_accs = []
    with tf.variable_scope(tf.get_variable_scope()) as scope:
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (model_wrapper.TOWER_NAME, i)) as scope:
            # loss for one tower.
            loss, tower_top1_acc, tower_top5_acc = tower_loss(scope, isTrain_ph, isLoad)
            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()
            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(loss)
            # Keep track of the gradients across all towers.
            tower_grads.append(grads)
            tower_top1_accs.append(tower_top1_acc)
            tower_top5_accs.append(tower_top5_acc)
            # testing
            # with tf.variable_scope('conv_ds_2', reuse = True) as scope:
            #     with tf.device('/cpu:0'):
            #         w_dw = tf.get_variable('w_dw')
            #         w_pw = tf.get_variable('w_pw')
            # with tf.variable_scope('conv1', reuse = True) as scope:
            #     with tf.device('/cpu:0'):
            #         w = tf.get_variable('w')

    grads = average_gradients(tower_grads)
    top1_acc = tf.reduce_mean(tower_top1_accs, 0)
    top5_acc = tf.reduce_mean(tower_top5_accs, 0)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(
            tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))
    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    # if (isLoad):
    #     checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
    #     saver.restore(sess, checkpoint_path)
    # else:
    #     sess.run(init)
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    train_epoch_size = 1281167
    # train_epoch_size = 128
    val_epoch_size = 50000

    train_bar = progressbar.ProgressBar(maxval = train_epoch_size,
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    val_bar = progressbar.ProgressBar(maxval = val_epoch_size,
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    top1_acc_vals = []
    top5_acc_vals = []
    if (FLAGS.is_train):
        running_epochs = FLAGS.max_epochs
    else:
        running_epochs = 1

    for epoch in xrange(running_epochs):
      if (FLAGS.is_train):
        print('This is the {} epoch of training'.format(epoch))
        train_bar.start()
      start_time = time.time()
      step = 0

      while (step <= train_epoch_size and FLAGS.is_train):
        # _, loss_value, w_dw_val, w_pw_val, w_val, grads_val= sess.run([
        #     train_op,
        #     loss,
        #     w_dw,
        #     w_pw,
        #     w,
        #     grads], feed_dict = {isTrain_ph:FLAGS.is_train})
        _, loss_value = sess.run([
            train_op,
            loss], feed_dict = {isTrain_ph:FLAGS.is_train})
        step += FLAGS.batch_size * FLAGS.num_gpus
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        # print(loss_value)
        if (step % 100 == 0):
        #   print(loss_value)
          with open("loss.txt", "a") as f:
            f.write("{}\n".format(loss_value))
        #   print(np.mean(grads_val[0]))
        #   print(grads_val[0])
        #   print(grads_val)
        # #   print(w_dw_val)
          train_bar.update(step)

      if FLAGS.is_train:
        duration = time.time() - start_time
        train_bar.finish()
        print('Performed {} training epochs with a time of {} and a loss of {}'.format(
            epoch+1,
            duration,
            loss_value))

      if FLAGS.pickle_save:
        model_wrapper.pickle_save(sess)

      print('start validation')
      start_time = time.time()
      step = 0
      val_bar.start()

      while (step <= val_epoch_size):
        top1_acc_val, top5_acc_val, loss_value= sess.run([top1_acc, top5_acc, loss], feed_dict = {isTrain_ph:False})
        top1_acc_vals.append(top1_acc_val)
        top5_acc_vals.append(top5_acc_val)
        step += FLAGS.batch_size * FLAGS.num_gpus
        if step % 10 == 0:
          val_bar.update(step)
      val_bar.finish()
      top1_acc_avg = sum(top1_acc_vals)/float(len(top1_acc_vals))
      top5_acc_avg = sum(top5_acc_vals)/float(len(top5_acc_vals))
      print('Top1 acc is {}, top5 acc is {}'.format(
          top1_acc_avg,
          top5_acc_avg))
      # Save the model checkpoint periodically.
      if (epoch% 10 == 0 or (epoch + 1) == FLAGS.max_epochs) and isTrain and FLAGS.ckpt_save:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
