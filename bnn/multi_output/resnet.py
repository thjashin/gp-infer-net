#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.ops.parallel_for import batch_jacobian
from tensorflow.contrib import layers
import zhusuan as zs
import gpflowSlim as gpflow
import numpy as np


def conv_res_block(input_, out_channel, data_format, resize=False):
    if not resize:
        lz_x = layers.conv2d(input_, out_channel, 3, stride=1,
                             data_format=data_format)
        lz_x = layers.conv2d(lz_x, out_channel, 3, stride=1,
                             activation_fn=None, data_format=data_format)
        lz_x += input_
    else:
        lz_x = layers.conv2d(input_, out_channel, 3, stride=2,
                             data_format=data_format)
        lz_x = layers.conv2d(lz_x, out_channel, 3, stride=1,
                             activation_fn=None, data_format=data_format)
        residual = layers.conv2d(input_, out_channel, 3, stride=2,
                                 activation_fn=None, data_format=data_format)
        lz_x += residual
    lz_x = tf.nn.relu(lz_x)
    return lz_x


def headless_network(h, block_sizes, block_strides, data_format):
    if data_format == "NCHW":
        data_format_v2 = "channels_first"
    else:
        data_format_v2 = "channels_last"
    h = tf.layers.conv2d(h, 64, 3, padding="same", data_format=data_format_v2)

    nf = 64
    for i, num_blocks in enumerate(block_sizes):
        print("block_layer_{}".format(i))
        with tf.variable_scope("block_layer_{}".format(i + 1)):
            with tf.variable_scope("block_0"):
                # h = block_v2(h, True, block_strides[i], data_format, dtype, nf)
                resize = True if block_strides[i] == 2 else False
                h = conv_res_block(h, nf, data_format, resize=resize)
                print("-> h: {}".format(h.shape))
            print("block 0")
            for j in range(1, num_blocks):
                with tf.variable_scope("block_{}".format(j)):
                    # h = block_v2(h, False, 1, data_format, dtype, nf)
                    h = conv_res_block(h, nf, data_format)
                print("block {}:".format(j))
                print("-> h: {}".format(h.shape))
        nf *= 2

    # Dense layer
    flattened_dim = np.prod(h.shape[-3:].as_list())
    h = tf.reshape(h, [-1, flattened_dim])

    # # var: [h, 512]
    h = tf.layers.dense(h, 512)
    return h


def build_resnet(scope, n_cls, input_shape, block_sizes,
                 block_strides, data_format="NCHW", mvn=True, dtype=tf.float64,
                 net="tangent"):
    @zs.reuse(scope)
    def resnet(x, full_cov=True):
        # x: [batch_size, x_dim]
        # input_shape: [C, H, W]
        n = tf.shape(x)[0]
        x = tf.reshape(x, [n] + input_shape)

        # Define almost all the network
        # h: [bs, 512]
        h = headless_network(x, block_sizes, block_strides, data_format)
        h = tf.nn.relu(h)

        if net == "tangent":
            layer_sizes_ = [int(h.shape[-1]), 100, 1]
            h = tf.tile(h[None, ...], [n_cls, 1, 1])
            batch_size = tf.shape(x)[0]
            w_means, w_stds = [], []
            for i, (n_in, n_out) in enumerate(zip(layer_sizes_[:-1],
                                                  layer_sizes_[1:])):
                w_mean = tf.get_variable(
                    'w_mean_' + str(i),
                    shape=[n_cls, n_in, n_out],
                    dtype=dtype,
                    initializer=tf.truncated_normal_initializer(stddev=0.01))
                w_logstd = tf.get_variable(
                    'w_logstd_' + str(i),
                    shape=[n_cls, n_in, n_out],
                    dtype=dtype,
                    initializer=tf.constant_initializer(-2))

                b_mean = tf.get_variable(
                    'b_mean_' + str(i),
                    shape=[n_cls, 1, n_out],
                    dtype=dtype,
                    initializer=tf.zeros_initializer())
                b_logstd = tf.get_variable(
                    'b_logstd_' + str(i),
                    shape=[n_cls, 1, n_out],
                    dtype=dtype,
                    initializer=tf.constant_initializer(-2))

                w_means.extend([w_mean, b_mean])
                w_stds.extend([tf.exp(w_logstd), tf.exp(b_logstd)])
                h = tf.matmul(h, w_mean) + b_mean
                if i == len(layer_sizes_) - 2:
                    h = tf.nn.tanh(h)
                    # h = tf.nn.relu(h)
                elif i < len(layer_sizes_) - 2:
                    h = tf.nn.tanh(h)
            # f_mean: [n_cls, batch_size]
            f_mean = tf.squeeze(h, -1)
            # grad_w_means: [[n_output, batch_size, shape(w)], ...]
            grad_w_means = [batch_jacobian(f_mean, w_mean) for w_mean in w_means]
            # f_cov_half: [[n_output, batch_size, shape(w)], ...]
            f_cov_half = [tf.reshape(grad_w_mean * w_std[:, None, ...],
                                     [n_cls, batch_size, -1])
                          for (grad_w_mean, w_std) in zip(grad_w_means, w_stds)]
            f_cov = tf.add_n([tf.matmul(i, i, transpose_b=True)
                              for i in f_cov_half])
            f_var = tf.add_n([tf.reduce_sum(tf.square(i), -1)
                              for i in f_cov_half])
        elif net == "deep":
            layer_sizes_ = [100]
            h0 = tf.tile(h[None, ...], [n_cls, 1, 1])
            h = h0
            for n_units in layer_sizes_:
                # h: [n_cls, batch_size, n_units]
                h = tf.layers.dense(h, n_units, activation=tf.tanh)
            n_units = layer_sizes_[-1]
            w_mean = tf.get_variable(
                "w_mean", shape=[n_cls, n_units], dtype=tf.float64,
                initializer=tf.truncated_normal_initializer(
                    stddev=0.01, dtype=tf.float64))
            w_cov_raw = tf.get_variable(
                "w_cov", dtype=tf.float64,
                initializer=tf.tile(
                    tf.eye(n_units, dtype=tf.float64)[None, ...],
                    [n_cls, 1, 1]))
            w_cov_tril = tf.matrix_set_diag(
                tf.matrix_band_part(w_cov_raw, -1, 0),
                tf.nn.softplus(tf.matrix_diag_part(w_cov_raw)))
            # f_mean: [n_cls, batch_size]
            f_mean = tf.squeeze(tf.matmul(h, w_mean[..., None]), -1)
            if True:
                h_res = tf.layers.dense(h0, 100, activation=tf.tanh)
                f_mean_res = tf.squeeze(tf.layers.dense(h_res, 1), -1)
                f_mean += f_mean_res
            # f_cov: [n_cls, batch_size, batch_size]
            f_cov_half = tf.matmul(h, w_cov_tril)
            f_cov = tf.matmul(f_cov_half, f_cov_half, transpose_b=True)
            f_var = tf.reduce_sum(tf.square(f_cov_half), axis=-1)
        else:
            raise NotImplementedError()

        if full_cov:
            # f_cov: [n_cls, bs, bs]
            f_cov = f_cov + tf.eye(tf.shape(f_cov)[1], dtype=dtype) * \
                gpflow.settings.jitter
            if mvn:
                f_cov_tril = tf.cholesky(f_cov)
                f_dist = zs.distributions.MultivariateNormalCholesky(
                    f_mean, f_cov_tril)
                return f_dist
            else:
                return f_mean, f_cov
        else:
            # f_cov_diag: [n_cls, bs]
            f_var = f_var + gpflow.settings.jitter
            return f_mean, f_var
    return resnet
