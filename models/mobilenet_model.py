import tensorflow as tf
import numpy as np
import pickle
import sys
slim = tf.contrib.slim
from tensorflow.contrib.framework import add_model_variable
from tensorflow.python.training import moving_averages



"""
gives back self.pred, self.
"""
class mobilenet(object):
    def __init__(self, isLoad, isTrain):
        self._get_variables(isLoad)
        self._init_weight_masks(isLoad)
        self.isTrain = isTrain
        # self.conv_network()

    def loss(self, logits, labels):
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
        # self.keep_prob = keep_prob
        imgs = images
        conv1 = self.conv_layer(imgs, 'conv1', stride = 2, padding = 'SAME', prune = True)
        conv_ds2 = self.depth_separable_layer(conv1, 'conv_ds_2', padding = 'SAME', prune = True)
        conv_ds3 = self.depth_separable_layer(conv_ds2, 'conv_ds_3', strides = 2, padding = 'SAME', prune = True)
        conv_ds4 = self.depth_separable_layer(conv_ds3, 'conv_ds_4', padding = 'SAME', prune = True)
        conv_ds5 = self.depth_separable_layer(conv_ds4, 'conv_ds_5', strides = 2, padding = 'SAME', prune = True)
        conv_ds6 = self.depth_separable_layer(conv_ds5, 'conv_ds_6', padding = 'SAME', prune = True)
        conv_ds7 = self.depth_separable_layer(conv_ds6, 'conv_ds_7', strides = 2,padding = 'SAME', prune = True)

        conv_ds8 = self.depth_separable_layer(conv_ds7, 'conv_ds_8', padding = 'SAME', prune = True)
        conv_ds9 = self.depth_separable_layer(conv_ds8, 'conv_ds_9', padding = 'SAME', prune = True)
        conv_ds10 = self.depth_separable_layer(conv_ds9, 'conv_ds_10', padding = 'SAME', prune = True)
        conv_ds11 = self.depth_separable_layer(conv_ds10, 'conv_ds_11', padding = 'SAME', prune = True)
        conv_ds12 = self.depth_separable_layer(conv_ds11, 'conv_ds_12', padding = 'SAME', prune = True)

        conv_ds13 = self.depth_separable_layer(conv_ds12, 'conv_ds_13', strides = 2, padding = 'SAME', prune = True)
        conv_ds14 = self.depth_separable_layer(conv_ds13, 'conv_ds_14', padding = 'SAME', prune = True)
        avg_pool = tf.nn.avg_pool(conv_ds14, ksize = [7,7],
                                  strides = [1,1,1,1], padding='VALID', name='avg_pool_15')
        # avg_pool = tf.nn.pool(conv_ds14, )

        # conv15 = self.conv_layer(avg_pool, 'conv15', stride = 1, padding = 'SAME', prune = True)
        squeeze = tf.squeeze(avg_pool, [1, 2], name='SpatialSqueeze')
        self.pred = self.fc_layer(squeeze, 'fc_16', prune = True, apply_relu = False)
        # self.pred= conv15
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

    def get_bn_variables(self, name, n_out, use_scale, use_bias, gamma_init):
        with tf.variable_scope(name, reuse = False) as scope:
            if use_bias:
                beta = tf.get_variable('beta', [n_out], initializer=tf.constant_initializer())
            else:
                beta = tf.zeros([n_out], name='beta')
            if use_scale:
                gamma = tf.get_variable('gamma', [n_out], initializer=gamma_init)
            else:
                gamma = tf.ones([n_out], name='gamma')
            # x * gamma + beta

            moving_mean = tf.get_variable('mean/EMA', [n_out],
                                          initializer=tf.constant_initializer(), trainable=False)
            moving_var = tf.get_variable('variance/EMA', [n_out],
                                         initializer=tf.constant_initializer(), trainable=False)
        return beta, gamma, moving_mean, moving_var


    def update_bn_ema(self, xn, batch_mean, batch_var, moving_mean, moving_var, decay):
        update_op1 = moving_averages.assign_moving_average(
            moving_mean, batch_mean, decay, zero_debias=False,
            name='mean_ema_op')
        update_op2 = moving_averages.assign_moving_average(
            moving_var, batch_var, decay, zero_debias=False,
            name='var_ema_op')
        # Only add to model var when we update them
        add_model_variable(moving_mean)
        add_model_variable(moving_var)

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op1)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op2)
        return xn

    def reshape_for_bn(self, param, ndims, chan, data_format):
        if ndims == 2:
            shape = [1, chan]
        else:
            shape = [1, 1, 1, chan] if data_format == 'NHWC' else [1, chan, 1, 1]
        return tf.reshape(param, shape)

    def batch_norm(self, x, name, train_phase=True, data_format = 'NHWC', epsilon = 1e-3):
        norm = tf.contrib.layers.batch_norm(
            x,
            scale = True,
            is_training = train_phase,
            scope = 'bn',
            reuse = False
        )
        return norm

    def batchnorm(self, x, name, train_phase=True, decay=0.9, epsilon=1e-5,
                  use_scale=True, use_bias=True,
                  gamma_init=tf.constant_initializer(1.0), data_format='NHWC'):
        """
        ref:
        https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/models/batch_norm.py

        """
        shape = x.get_shape().as_list()
        ndims = len(shape)
        assert ndims in [2, 4]
        if ndims == 2:
            data_format = 'NHWC'
        if data_format == 'NCHW':
            n_out = shape[1]
        else:
            n_out = shape[-1]  # channel
        assert n_out is not None, "Input to BatchNorm cannot have unknown channels!"
        beta, gamma, moving_mean, moving_var = self.get_bn_variables(name, n_out, use_scale, use_bias, gamma_init)

        if train_phase:
            if ndims == 2:
                x = tf.reshape(x, [-1, 1, 1, n_out])    # fused_bn only takes 4D input
                # fused_bn has error using NCHW? (see #190)

            xn, batch_mean, batch_var = tf.nn.fused_batch_norm(
                x, gamma, beta, epsilon=epsilon,
                is_training=True, data_format=data_format)

            if ndims == 2:
                xn = tf.squeeze(xn, [1, 2])
        else:
            # non-fused op is faster for inference
            if ndims == 4 and data_format == 'NCHW':
                [g, b, mm, mv] = [reshape_for_bn(_, ndims, n_out, data_format)
                                  for _ in [gamma, beta, moving_mean, moving_var]]
                xn = tf.nn.batch_normalization(x, mm, mv, b, g, epsilon)
            else:
                # avoid the reshape if possible (when channel is the last dimension)
                xn = tf.nn.batch_normalization(
                    x, moving_mean, moving_var, beta, gamma, epsilon)

        # maintain EMA only on one GPU is OK, even in replicated mode.
        # because training time doesn't use EMA

        # if index==0:
        #     ret = self.update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, decay)
        # else:
        #     ret = tf.identity(xn, name='output')
        ret = self.update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, decay)
        return ret

    def dropout_layer(self, x):
        return tf.nn.dropout(x, self.keep_prob)

    def fc_layer(self, x, name, prune = False, apply_relu = True, use_bias = False):
        with tf.variable_scope(name, reuse = True):
            with tf.device('/cpu:0'):
                w = tf.get_variable('w')
                if use_bias:
                    b = tf.get_variable('b')
            if prune:
                w = w * self.weights_masks[name]
            if use_bias:
                ret = tf.nn.xw_plus_b(x,w,b)
            else:
                ret = tf.matmul(x,w)
            if apply_relu:
                ret = tf.nn.relu(ret)
        return ret

    def conv_layer(self, x, name, padding = 'SAME', stride = 1,
        split = 1, data_format = 'NHWC', prune = False):

        channel_axis = 3 if data_format == 'NHWC' else 1
        with tf.variable_scope(name, reuse = True):
            with tf.device('/cpu:0'):
                w = tf.get_variable('w')
                # b = tf.get_variable('b')
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
            # ret = tf.nn.relu(tf.nn.bias_add(conv, b, data_format=data_format), name='output')
            ret = tf.nn.relu(conv, name='output')
        return ret

    def depth_separable_layer(self, x, name, padding = 'SAME', strides = 1, prune = True):
        with tf.variable_scope(name, reuse = True) as scope:
            with tf.device('/cpu:0'):
                w_dw = tf.get_variable('w_dw')
                w_pw = tf.get_variable('w_pw')
            # depthwise layer
            dw = tf.nn.depthwise_conv2d(x, w_dw, strides=[1, strides, strides, 1], padding="SAME")
            # dw_norm = self.batch_norm(dw, 'norm_dw', train_phase = self.isTrain)
            # dw_norm = self.batch_norm(dw, 'norm_dw', train_phase = self.isTrain)
        # with tf.variable_scope(name) as scope:
            # dw_norm = self.batchnorm(dw, 'dw')
            # # dw_norm = self.batch_norm(dw, 'norm_dw')
            # dw_relu = tf.nn.relu(dw_norm)
            dw_relu = tf.nn.relu(dw)
        # with tf.variable_scope(name, reuse = True) as scope:
            # point-wise layer
            pw = tf.nn.conv2d(dw_relu, w_pw, [1, 1, 1, 1], padding="SAME")
        # with tf.variable_scope(name) as scope:
            # pw_norm = self.batch_norm(pw, 'norm_pw')
            # pw_norm = self.batchnorm(pw, 'pw')
            # pw_relu = tf.nn.relu(pw_norm)
            pw_relu = tf.nn.relu(pw)
        return pw_relu

    def _get_variables(self, isload, weights_path = 'DEFAULT'):
        """
        Network architecture definition
        """
        self.keys = ['conv1', 'conv_ds_2', 'conv_ds_3',
                    'conv_ds_4', 'conv_ds_5', 'conv_ds_6', 'conv_ds_7',
                    'conv_ds_8', 'conv_ds_9', 'conv_ds_10', 'conv_ds_11',
                    'conv_ds_12', 'conv_ds_13', 'conv_ds_14', 'fc_16'
                    ]
        kernel_shapes = [
            [3, 3, 3, 32], #conv1
            [32, 64],#conv_ds_2
            [64, 128],
            [128, 128],
            [128, 256],
            [256, 256],#conv_ds_6
            [256, 512],
            [512, 512],
            [512, 512],
            [512, 512],
            [512, 512],
            [512, 512],
            [512, 1024],
            [1024, 1024],
            [1024, 1001]
        ]
        self.weight_shapes = kernel_shapes
        if isload:
            with open('mobilenet_vars.pkl', 'rb') as f:
                weights, biases = pickle.load(f)
            for i, key in enumerate(self.keys):
                self._init_layerwise_variables(w_shape = kernel_shapes[i],
                    b_shape = biase_shape[i],
                    name = key,
                    w_init = weights[key],
                    b_init = biases[key])
        else:
            for i,key in enumerate(self.keys):
                if (key == 'conv1' or key == 'fc_16'):
                    self._init_layerwise_variables(w_shape = kernel_shapes[i],
                        b_shape = None,
                        name = key)
                else:
                    self._init_depthseparable_variables(
                        name = key,
                        input_channel = kernel_shapes[i][0],
                        output_channel = kernel_shapes[i][1]
                    )

    def _init_layerwise_variables(self, w_shape, b_shape, name, w_init = None, b_init = None):
        with tf.variable_scope(name):
            with tf.device('/cpu:0'):
                if w_init is None:
                    w_init = tf.contrib.layers.variance_scaling_initializer()
                else:
                    w_init = tf.constant(w_init)
                if b_init is None:
                    b_init = tf.constant_initializer()
                else:
                    b_init = tf.constant(b_init)
                w = tf.get_variable('w', w_shape, initializer = w_init)
                # biases might not be necessary
                if b_shape == None:
                    pass
                else:
                    b = tf.get_variable('b', b_shape, initializer = b_init)

    def _init_depthseparable_variables(self, name, input_channel, output_channel, w_init = None):
        with tf.variable_scope(name):
            with tf.device('/cpu:0'):
                if w_init is None:
                    w_init = tf.contrib.layers.variance_scaling_initializer()
                else:
                    w_init = tf.constant(w_init)
                dw_shape = [3,3,input_channel,1]
                pw_shape = [1,1,input_channel,output_channel]
                dw = tf.get_variable('w_dw', dw_shape, initializer = w_init)
                pw = tf.get_variable('w_pw', pw_shape, initializer = w_init)

    def _init_weight_masks(self, is_load):
        names = self.keys
        # if is_load:
        #     with open(weights_path+'mask.npy', 'rb') as f:
        #         self.weights_masks = pickle.load(f)
        # else:
        self.weights_masks = {}
        # self.biases_masks = {}
        for i, key in enumerate(names):
            self.weights_masks[key] = np.ones(self.weight_shapes[i])
            # self.biases_masks[key] = np.ones(self.biase_shapes[i])

    def _apply_a_mask(self, mask, var):
        return (var * mask)

def save_model(sess, weights_path = 'mobilenet_vars'):
    w_save = {}
    b_save = {}
    keys = ['conv1', 'conv_ds_2', 'conv_ds_3',
                    'conv_ds_4', 'conv_ds_5', 'conv_ds_6', 'conv_ds_7',
                    'conv_ds_8', 'conv_ds_9', 'conv_ds_10', 'conv_ds_11',
                    'conv_ds_12', 'conv_ds_13', 'conv_ds_14', 'fc_16'
                    ]
    for key in keys:
        with tf.variable_scope(key, reuse = True):
            with tf.device('/cpu:0'):
                if key == 'conv1' or key == 'fc_16':
                    w = tf.get_variable('w')
                else:
                    w_dw = tf.get_variable('w_dw')
                    w_pw = tf.get_variable('w_dw')
                # b = tf.get_variable('b')
        if key == 'conv1' or key == 'fc_16':
            w_save[key] = w.eval(session = sess)
        else:
            w_save[key] = [w_dw.eval(session = sess), w_pw.eval(session = sess)]
        # b_save[key] = b.eval(session = sess)
    with open(weights_path, 'wb') as f:
        pickle.dump([w_save, b_save], f)
    print('model saved')
