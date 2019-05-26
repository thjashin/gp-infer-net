#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import zhusuan as zs
import gpflowSlim as gpflow


def build_deep_linear(scope, layer_sizes, mvn=True, activation=tf.nn.tanh):
    @zs.reuse(scope)
    def deep_linear(x, full_cov=True):
        # x: [batch_size, x_dim]
        h = x
        for n_units in layer_sizes:
            # h: [batch_size, n_units]
            h = tf.layers.dense(h, n_units, activation=activation)
        # w_mean: [n_units]
        n_units = layer_sizes[-1]
        w_mean = tf.get_variable(
            "w_mean", shape=[n_units], dtype=tf.float64,
            initializer=tf.truncated_normal_initializer(
                stddev=0.001, dtype=tf.float64))
        w_cov_raw = tf.get_variable(
            "w_cov", dtype=tf.float64,
            initializer=tf.eye(n_units, dtype=tf.float64))
        w_cov_tril = tf.matrix_set_diag(
            tf.matrix_band_part(w_cov_raw, -1, 0),
            tf.nn.softplus(tf.matrix_diag_part(w_cov_raw)))
        # f_mean: [batch_size]
        f_mean = tf.squeeze(tf.matmul(h, w_mean[:, None]), -1)
        # f_cov: [batch_size, batch_size]
        f_cov_half = tf.matmul(h, w_cov_tril)
        if full_cov:
            f_cov = tf.matmul(f_cov_half, f_cov_half, transpose_b=True)
            f_cov = f_cov + tf.eye(tf.shape(f_cov)[0], dtype=tf.float64) * \
                gpflow.settings.jitter
            if mvn:
                f_cov_tril = tf.cholesky(f_cov)
                f_dist = zs.distributions.MultivariateNormalCholesky(
                    f_mean, f_cov_tril)
                return f_dist
            else:
                return f_mean, f_cov
        else:
            # hw_cov: [batch_size, n_units]
            hw_cov = tf.matmul(f_cov_half, w_cov_tril, transpose_b=True)
            # f_cov_diag: [batch_size]
            f_var = tf.reduce_sum(hw_cov * h, axis=-1)
            f_var += gpflow.settings.jitter
            return f_mean, f_var
    return deep_linear
