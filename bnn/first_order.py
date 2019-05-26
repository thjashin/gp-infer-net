#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops.parallel_for import jacobian
import gpflowSlim as gpflow
import zhusuan as zs


def build_first_order(scope, layer_sizes, periodic=False, mvn=True,
                      activation=tf.nn.tanh):
    @zs.reuse(scope)
    def first_order(x, full_cov=True):
        # x: [batch_size, x_dim]
        if periodic:
            x = tf.concat([x, tf.sin(tf.layers.dense(x, x.shape[1]))], axis=-1)
        layer_sizes_ = [int(x.shape[1])] + layer_sizes + [1]
        h = x
        batch_size = tf.shape(x)[0]
        w_means, w_stds = [], []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes_[:-1],
                                              layer_sizes_[1:])):
            w_mean = tf.get_variable(
                'w_mean_' + str(i), shape=[n_in, n_out], dtype=tf.float64,
                initializer=tf.truncated_normal_initializer(stddev=0.01))
            w_logstd = tf.get_variable(
                'w_logstd_' + str(i), shape=[n_in, n_out], dtype=tf.float64,
                initializer=tf.constant_initializer(-2))

            b_mean = tf.get_variable(
                'b_mean_' + str(i), shape=[1, n_out], dtype=tf.float64,
                initializer=tf.zeros_initializer())
            b_logstd = tf.get_variable(
                'b_logstd_' + str(i), shape=[1, n_out], dtype=tf.float64,
                initializer=tf.constant_initializer(-2))

            w_means.extend([w_mean, b_mean])
            w_stds.extend([tf.exp(w_logstd), tf.exp(b_logstd)])
            h = tf.matmul(h, w_mean) + b_mean
            if i < len(layer_sizes_) - 2:
                h = activation(h)
        # f_mean: [batch_size]
        f_mean = tf.squeeze(h, -1)
        # grad_w_means: [[batch_size, shape(w)], ...]
        grad_w_means = jacobian(f_mean, w_means)
        # f_cov_half: [[batch_size, shape(w)], ...]
        f_cov_half = [tf.reshape(grad_w_mean * w_std, [batch_size, -1])
                      for (grad_w_mean, w_std) in zip(grad_w_means, w_stds)]
        if full_cov:
            # f_cov: [batch_size, batch_size]
            f_cov = tf.add_n([tf.matmul(i, i, transpose_b=True)
                              for i in f_cov_half])
            f_cov += gpflow.settings.jitter * tf.eye(
                batch_size, dtype=tf.float64)
            if mvn:
                f_cov_tril = tf.cholesky(f_cov)
                f_dist = zs.distributions.MultivariateNormalCholesky(
                    f_mean, f_cov_tril)
                return f_dist
            else:
                return f_mean, f_cov
        else:
            # f_var: [batch_size]
            f_var = tf.add_n([tf.reduce_sum(tf.square(i), -1)
                              for i in f_cov_half])
            f_var += gpflow.settings.jitter
            return f_mean, f_var
    return first_order
