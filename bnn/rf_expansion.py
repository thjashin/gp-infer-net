#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import zhusuan as zs
import gpflowSlim as gpflow
import numpy as np


def build_rf_expansion(scope, x_dim, n_units, kernel, mvn=True, fix_freq=False,
                       residual=False, fix_ls=False, activation=tf.tanh):
    fixed_freq = np.random.normal(size=(x_dim, n_units))
    @zs.reuse(scope)
    def rf_expansion(x, full_cov=True):
        # x: [batch_size, x_dim]
        h = x
        # freq: [x_dim, n_units]
        if fix_freq:
            freq = fixed_freq
        else:
            freq = tf.get_variable(
                "freq", dtype=tf.float64,
                initializer=tf.constant(fixed_freq, dtype=tf.float64))
        if fix_ls:
            lengthscales = kernel.lengthscales
        else:
            lengthscales = tf.get_variable(
                "ls", dtype=tf.float64, initializer=kernel.lengthscales)
        # h: [batch_size, n_units]
        h = tf.matmul(h, freq * 1. / lengthscales[..., None])
        # h: [batch_size, 2 * n_units]
        h = tf.sqrt(kernel.variance) / np.sqrt(n_units) * tf.concat(
            [tf.cos(h), tf.sin(h)], -1)
        # w_mean: [2 * n_units]
        w_mean = tf.get_variable(
            "w_mean", shape=[2 * n_units], dtype=tf.float64,
            initializer=tf.random_normal_initializer(
                stddev=0.001, dtype=tf.float64))
        w_cov_raw = tf.get_variable(
            "w_cov", dtype=tf.float64,
            initializer=tf.eye(2 * n_units, dtype=tf.float64))
        w_cov_tril = tf.matrix_set_diag(
            tf.matrix_band_part(w_cov_raw, -1, 0),
            tf.nn.softplus(tf.matrix_diag_part(w_cov_raw)))
        # f_mean: [batch_size]
        f_mean = tf.squeeze(tf.matmul(h, w_mean[:, None]), -1)
        # f_mean_res: [batch_size]
        if residual:
            h_res = tf.layers.dense(x, n_units, activation=activation)
            f_mean_res = tf.squeeze(tf.layers.dense(h_res, 1), -1)
            f_mean += f_mean_res
        # f_cov_half: [batch_size, 2 * n_units]
        f_cov_half = tf.matmul(h, w_cov_tril)
        if full_cov:
            # f_cov: [batch_size, batch_size]
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
            # f_cov_diag: [batch_size]
            f_var = tf.reduce_sum(tf.square(f_cov_half), axis=-1)
            f_var = f_var + gpflow.settings.jitter
            return f_mean, f_var
    return rf_expansion
