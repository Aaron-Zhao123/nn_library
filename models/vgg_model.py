import tensorflow as tf
import numpy as np
import pickle
import sys

"""
gives back self.pred, self.
"""
class vggnet(object):
    def __init__(self, isLoad, isTrain):
        self._get_variables(isLoad)
        self._init_weight_masks(isLoad)
        # self.conv_network()

    def loss(logits, labels):
        """for constructing the graph"""
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logtis, labels, name = 'cross_entropy_batchwise')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
        """
        puts cross entropy as a loss into collection
        potentially, there can be L1 and L2 losses added to this "losses" term
        in collections
        """
        tf.add_to_collection('losses', cross_entropy_mean)


    def error_rates(self,topk = 1):
        """
        Args:
            self.pred: shape [B,C].
            self.labels: shape [B].
            topk(int): topk
        Returns:
            a float32 vector of length N with 0/1 values. 1 means incorrect
            prediction.
        """
        return tf.cast(tf.logical_not(tf.nn.in_top_k(self.pred, self.labels, topk)),
            tf.float32)

    def conv_network(self, images, keep_prob):
        self.keep_prob = keep_prob
        imgs = images

        conv1_1 = self.conv_layer(imgs, 'conv1_1', padding = 'SAME', prune = True)
        conv1_2 = self.conv_layer(conv1_1, 'conv1_2', padding = 'SAME', prune = True)
        pool1 = self.maxpool(conv1_2, 'pool1', 2, 2, padding = 'SAME')

        conv2_1 = self.conv_layer(pool1, 'conv2_1', padding = 'SAME', prune = True)
        conv2_2 = self.conv_layer(conv2_1, 'conv2_2', padding = 'SAME', prune = True)
        pool2 = self.maxpool(conv2_2, 'pool2', 2, 2, padding = 'SAME')

        conv3_1 = self.conv_layer(pool2, 'conv3_1', padding = 'SAME', prune = True)
        conv3_2 = self.conv_layer(conv3_1, 'conv3_2', padding = 'SAME', prune = True)
        conv3_3 = self.conv_layer(conv3_2, 'conv3_3', padding = 'SAME', prune = True)
        pool3 = self.maxpool(conv3_3, 'pool3', 2, 2, padding = 'SAME')

        conv4_1 = self.conv_layer(pool3, 'conv4_1', padding = 'SAME', prune = True)
        conv4_2 = self.conv_layer(conv4_1, 'conv4_2', padding = 'SAME', prune = True)
        conv4_3 = self.conv_layer(conv4_2, 'conv4_3', padding = 'SAME', prune = True)
        pool4 = self.maxpool(conv4_3, 'pool4', 2, 2, padding = 'SAME')

        conv5_1 = self.conv_layer(pool4, 'conv5_1', padding = 'SAME', prune = True)
        conv5_2 = self.conv_layer(conv5_1, 'conv5_2', padding = 'SAME', prune = True)
        conv5_3 = self.conv_layer(conv5_2, 'conv5_3', padding = 'SAME', prune = True)
        pool5 = self.maxpool(conv5_3, 'pool5', 2, 2, padding = 'SAME')

        shape = int(np.prod(pool5.get_shape()[1:]))
        flattened = tf.reshape(pool5, [-1, shape])
        fc6 = self.fc_layer(flattened, 'fc6', prune = True)
        fc6_drop = self.dropout_layer(fc6)
        # norm6 = self.batch_norm(fc6, 'norm6', train_phase = self.isTrain)

        fc7 = self.fc_layer(fc6_drop, 'fc7', prune = True)
        fc7_drop = self.dropout_layer(fc7)
        # norm7 = self.batch_norm(fc7, 'norm7', train_phase = self.isTrain)

        fc8 = self.fc_layer(fc7_drop, 'fc8', prune = True, apply_relu = False)
        self.pred = fc8
        return self.pred

    def maxpool(self, x, name, filter_size, stride, padding = 'SAME'):
        return tf.nn.max_pool(x, ksize = [1, filter_size, filter_size, 1],
            strides = [1, stride, stride, 1], padding = padding, name = name)

    def lrn(self, x, name, depth_radius = 2, bias = 1.0, alpha = 2e-5, beta = 0.75):
        """
        local response normalization
        ref: https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
        """
        return tf.nn.lrn(x, depth_radius = depth_radius, bias = bias,
            alpha = alpha, beta = beta, name = name)

    def batch_norm(self, x, name, train_phase, data_format = 'NHWC', epsilon = 1e-3):
        """
        TODO: this batch norm has an error
        refs:
        1. https://github.com/ppwwyyxx/tensorpack/blob/a3674b47bfbf0c8b04aaa85d428b109fea0128ca/tensorpack/models/batch_norm.py
        2. https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412
        """
        shape = x.get_shape().as_list()
        ndims = len(shape)

        assert ndims in [2,4]
        if ndims == 2:
            data_format = 'NHWC'

        if data_format == 'NCHW':
            n_out = shape[1]
        else:
            n_out = shape[-1]  # channel

        assert n_out is not None, "Input to BatchNorm cannot have unknown channels!"

        with tf.variable_scope(name):
            beta = tf.Variable(tf.constant(0.0, shape = [n_out]),
                name = 'beta', trainable = True)
            gamma = tf.Variable(tf.constant(1.0, shape = [n_out]),
                name = 'gamma', trainable = True)
            axis = list(range(len(x.get_shape())-1))
            batch_mean, batch_var = tf.nn.moments(x, axis, name = 'moments')

            ema = tf.train.ExponentialMovingAverage(decay = 0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
            mean, var = tf.cond(train_phase,
                mean_var_with_update,
                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
        return normed

    def dropout_layer(self, x):
        return tf.nn.dropout(x, self.keep_prob)

    def fc_layer(self, x, name, prune = False, apply_relu = True):
        with tf.variable_scope(name, reuse = True):
            with tf.device('/cpu:0'):
                w = tf.get_variable('w')
                b = tf.get_variable('b')
            if prune:
                w = w * self.weights_masks[name]
            ret = tf.nn.xw_plus_b(x,w,b)
            if apply_relu:
                ret = tf.nn.relu(ret)
        return ret

    def conv_layer(self, x, name, padding = 'SAME', stride = 1,
        split = 1, data_format = 'NHWC', prune = False):

        channel_axis = 3 if data_format == 'NHWC' else 1
        with tf.variable_scope(name, reuse = True):
            with tf.device('/cpu:0'):
                w = tf.get_variable('w')
                b = tf.get_variable('b')
            if prune:
                w = w * self.weights_masks[name]
            if split == 1:
                conv = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding, data_format=data_format)
                # conv = tf.nn.conv2d(x, w, stride, padding)
            else:
                inputs = tf.split(x, split, channel_axis)
                kernels = tf.split(w, split, 3)
                # outputs = [tf.nn.conv2d(i, k, stride, padding)
                outputs = [tf.nn.conv2d(i, k, [1, stride, stride, 1], padding, data_format=data_format)
                           for i, k in zip(inputs, kernels)]
                conv = tf.concat(outputs, channel_axis)

            # using Relu
            ret = tf.nn.relu(tf.nn.bias_add(conv, b, data_format=data_format), name='output')
        return ret

    def _get_variables(self, isload, weights_path = 'vgg_vars'):
        """
        Network architecture definition
        """
        self.keys = ['conv1_1', 'conv1_2',
                    'conv2_1', 'conv2_2',
                    'conv3_1', 'conv3_2', 'conv3_3',
                    'conv4_1', 'conv4_2', 'conv4_3',
                    'conv5_1', 'conv5_2', 'conv5_3',
                    'fc6',
                    'fc7',
                    'fc8'
                    ]
        kernel_shapes = [
            [3, 3, 3, 64],
            [3, 3, 64, 64],
            [3, 3, 64, 128],
            [3, 3, 128, 128],
            [3, 3, 128, 256],
            [3, 3, 256, 256],
            [3, 3, 256, 256],
            [3, 3, 256, 512],
            [3, 3, 512, 512],
            [3, 3, 512, 512],
            [3, 3, 512, 512],
            [3, 3, 512, 512],
            [3, 3, 512, 512],
            [7*7*512, 4096],
            [4096, 4096],
            [4096, 1001]
        ]
        biase_shape = [
            [64],
            [64],
            [128],
            [128],
            [256],
            [256],
            [256],
            [512],
            [512],
            [512],
            [512],
            [512],
            [512],
            [4096],
            [4096],
            [1001]
        ]
        self.weight_shapes = kernel_shapes
        self.biase_shapes = biase_shape
        if isload:
            with open(weights_path+'.pkl', 'rb') as f:
                weights, biases = pickle.load(f)
            for i, key in enumerate(self.keys):
                w_key = key + '/w'
                b_key = key + '/b'

                self._init_layerwise_variables(w_shape = kernel_shapes[i],
                    b_shape = biase_shape[i],
                    name = key,
                    w_init = weights[w_key],
                    b_init = biases[b_key])
        else:
            for i,key in enumerate(self.keys):
                self._init_layerwise_variables(w_shape = kernel_shapes[i],
                    b_shape = biase_shape[i],
                    name = key)

    def _init_layerwise_variables(self, w_shape, b_shape, name, w_init = None, b_init = None):
        with tf.variable_scope(name):
            with tf.device('/cpu:0'):
                if w_init is None:
                    w_init = tf.contrib.layers.variance_scaling_initializer()
                    w = tf.get_variable('w', w_shape, initializer = w_init)
                else:
                    # w_init = tf.constant(w_init, dtype=tf.float32)
                    # w = tf.get_variable('w', initializer = w_init)
                    w = tf.Variable(w_init, name = 'w', dtype = tf.float32)
                if b_init is None:
                    b_init = tf.constant_initializer()
                    b = tf.get_variable('b', b_shape, initializer = b_init)
                else:
                    b_init = tf.constant(b_init, dtype=tf.float32)
                    # b = tf.get_variable('b', initializer = b_init)
                    b = tf.Variable(b_init, name = 'b', dtype = tf.float32)

    def _init_weight_masks(self, is_load):
        names = self.keys
        # if is_load:
        #     with open(weights_path+'mask.npy', 'rb') as f:
        #         self.weights_masks, self.biases_masks= pickle.load(f)
        # else:
        self.weights_masks = {}
        self.biases_masks = {}
        for i, key in enumerate(names):
            self.weights_masks[key] = np.ones(self.weight_shapes[i])
            self.biases_masks[key] = np.ones(self.biase_shapes[i])

    def _apply_a_mask(self, mask, var):
        return (var * mask)
