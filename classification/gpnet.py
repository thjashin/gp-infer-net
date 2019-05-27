#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import zhusuan as zs
import gpflowSlim as gpflow

from bnn.multi_output.resnet import build_resnet
from utils.mvn import (multivariate_normal_cross_entropy,
                       multivariate_normal_entropy,
                       multivariate_normal_kl)


FLAGS = tf.flags.FLAGS


def gpnet(logger, sess, data, kernel, dtype=tf.float64):
    train_x = data.train_x
    N, x_dim = train_x.shape
    _, n_cls = data.train_y.shape

    bnn_prev = build_resnet(
        "bnn_prev",
        n_cls,
        kernel.input_shape,
        kernel.block_sizes,
        kernel.block_strides,
        data_format="NCHW",
        mvn=False,
        dtype=dtype,
        net=FLAGS.net)
    bnn = build_resnet(
        "bnn",
        n_cls,
        kernel.input_shape,
        kernel.block_sizes,
        kernel.block_strides,
        data_format="NCHW",
        dtype=dtype,
        net=FLAGS.net)

    x, y = data.next_batch
    n = tf.shape(x)[0]
    # x_star: [bs_star, x_dim], x: [bs, x_dim]
    x_star = data.measure_batch
    beta = tf.placeholder(dtype, shape=[], name="beta")
    # xx: [bs + bs_star, x_dim]
    xx = tf.concat([x, x_star], axis=0)

    qf_star = bnn(x_star)
    # qff_mean_prev: [n_cls, bs + bs_star]
    # K_prox: [n_cls, bs + bs_star, bs + bs_star]
    qff_mean_prev, K_prox = bnn_prev(xx)
    # qf_mean: [n_cls, bs], qf_var: [n_cls, bs, bs]
    qf_mean, qf_var = bnn(x, full_cov=False)
    # y_pred: [bs]
    y_pred = tf.argmax(qf_mean, axis=0, output_type=tf.int32)
    # y_target: [bs]
    y_target = tf.argmax(y, axis=1, output_type=tf.int32)
    # acc: []
    acc = tf.reduce_mean(tf.to_float(tf.equal(y_pred, y_target)))

    bnn_ops = []
    bnn_prev_var = tf.trainable_variables(scope="bnn_prev")
    bnn_var = tf.trainable_variables(scope="bnn")
    for (prev, cur) in zip(bnn_prev_var, bnn_var):
        bnn_ops.append(prev.assign(cur))
    bnn_op = tf.group(bnn_ops)

    with tf.variable_scope("likelihood"):
        likelihood = gpflow.likelihoods.Gaussian(var=0.1)
    likelihood_var = likelihood.variance
    # K_prior: [bs + bs_star, bs + bs_star]
    K_prior = kernel.K(xx) + tf.eye(
        tf.shape(xx)[0], dtype=dtype) * gpflow.settings.jitter

    # K_sum_tril: [n_cls, bs + bs_star, bs + bs_star]
    K_sum_tril = tf.cholesky(K_prox * beta + K_prior * (1 - beta))
    # K_sum_tril_inv: [n_cls, bs + bs_star, bs + bs_star]
    K_sum_tril_inv = tf.matrix_triangular_solve(
        K_sum_tril,
        tf.tile(tf.eye(tf.shape(xx)[0], dtype=dtype)[None, ...],
                [n_cls, 1, 1]))
    # K_sum_inv: [n_cls, bs + bs_star, bs + bs_star]
    K_sum_inv = tf.matmul(K_sum_tril_inv, K_sum_tril_inv, transpose_a=True)
    # K_adapt: [n_cls, bs + bs_star, bs + bs_star]
    K_adapt = tf.matmul(tf.tile(K_prior[None, ...], [n_cls, 1, 1]),
                        tf.matmul(K_sum_inv, K_prox))
    # mean_adapt: [n_cls, bs + bs_star, 1]
    mean_adapt = (1 - beta) * tf.matmul(
        K_adapt, tf.matrix_solve(K_prox, qff_mean_prev[..., None]))
    # mean_n: [n_cls, bs, 1], mean_m: [n_cls, bs_star, 1]
    mean_n, mean_m = mean_adapt[:, :n, :], mean_adapt[:, n:, :]
    # Kn: [n_cls, bs, bs],
    # Knm: [n_cls, bs, bs_star],
    # Km: [n_cls, bs_star, bs_star]
    Kn, Knm, Km = K_adapt[:, :n, :n], K_adapt[:, :n, n:], K_adapt[:, n:, n:]

    # Ky: [n_cls, bs, bs]
    Ky = Kn + tf.eye(n, dtype=dtype) * likelihood_var / (
        N / tf.cast(n, dtype) * beta)
    # Ky_tril: [n_cls, bs, bs]
    Ky_tril = tf.cholesky(Ky)

    # y: [bs, n_cls]
    # yf: [n_cls, bs]
    yf = tf.cast(tf.matrix_transpose(y), dtype)
    # mean_target: [n_cls, bs_star, 1]
    mean_target = tf.matmul(
        Knm, tf.cholesky_solve(Ky_tril, yf[..., None] - mean_n),
        transpose_a=True) + mean_m
    # mean_target: [n_cls, bs_star]
    mean_target = tf.squeeze(mean_target, -1)
    # K_target: [n_cls, bs_star, bs_star]
    K_target = Km - tf.matmul(Knm, tf.cholesky_solve(Ky_tril, Knm),
                              transpose_a=True)
    K_target_tril = tf.cholesky(K_target)
    target_pf_star = zs.distributions.MultivariateNormalCholesky(
        mean_target, K_target_tril)

    kl_obj = tf.reduce_sum(
        multivariate_normal_kl(qf_star, target_pf_star, dtype=dtype))

    # hyper-parameter update
    Kn_prior = tf.tile(K_prior[None, :n, :n], [n_cls, 1, 1])
    pf = zs.distributions.MultivariateNormalCholesky(
        tf.zeros([n_cls, n], dtype=dtype), tf.cholesky(Kn_prior))
    Kn_prox = K_prox[:, :n, :n]
    qf_prev_mean = qff_mean_prev[:, :n]
    qf_prev_var = tf.matrix_diag_part(Kn_prox)
    qf_prev = zs.distributions.MultivariateNormalCholesky(
        qf_prev_mean, tf.cholesky(Kn_prox))
    hyper_obj = tf.reduce_sum(likelihood.variational_expectations(
        qf_prev_mean, qf_prev_var, yf)) - tf.reduce_sum(
        multivariate_normal_kl(qf_prev, pf))

    gpnet_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    infer_gpnet = gpnet_opt.minimize(kl_obj, var_list=bnn_var)

    hyper_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.hyper_rate)
    gp_var_list = (tf.trainable_variables(scope="likelihood"))
    if FLAGS.hyper_rate < 1e-8:
        hyper_op = tf.no_op()
    else:
        hyper_op = hyper_opt.minimize(-hyper_obj, var_list=gp_var_list)

    test_freq = 50
    likelihood_var_list = tf.trainable_variables(scope="likelihood")
    var_list = bnn_prev_var + bnn_var + likelihood_var_list
    sess.run(tf.variables_initializer(
        var_list=var_list + gpnet_opt.variables() + hyper_opt.variables()))
    logger.info("prior gp var: {}".format(sess.run(likelihood_var)))

    for t in range(1, FLAGS.n_iters + 1):
        beta_t = FLAGS.beta0 * 1. / (1. + FLAGS.gamma * np.sqrt(t - 1))
        _, _, train_obj, obs_var, train_acc = sess.run(
            [infer_gpnet, hyper_op, kl_obj, likelihood_var, acc],
            feed_dict={data.handle: data.train_handle,
                       beta: beta_t})
        sess.run(bnn_op)
        logger.info("Iter {}, kl_obj = {}, obs_var = {}, train_acc = {}"
                    .format(t, train_obj, obs_var, train_acc))

        if t % test_freq == 0:
            sess.run(data.test_iterator.initializer)
            test_stats = []
            while True:
                try:
                    test_stats.append(
                        sess.run(acc,
                                 feed_dict={data.handle: data.test_handle}))
                except tf.errors.OutOfRangeError:
                    break
            logger.info(">> Test acc = {:.4f}".format(np.mean(test_stats)))


def gpnet_nonconj(logger, sess, data, kernel, dtype=tf.float64):
    train_x = data.train_x
    N, x_dim = train_x.shape
    _, n_cls = data.train_y.shape

    bnn_prev = build_resnet(
        "bnn_prev",
        n_cls,
        kernel.input_shape,
        kernel.block_sizes,
        kernel.block_strides,
        data_format="NCHW",
        dtype=dtype,
        net=FLAGS.net)
    bnn = build_resnet(
        "bnn",
        n_cls,
        kernel.input_shape,
        kernel.block_sizes,
        kernel.block_strides,
        data_format="NCHW",
        dtype=dtype,
        net=FLAGS.net)

    x, y = data.next_batch
    n = tf.shape(x)[0]
    # x_star: [bs_star, x_dim], x: [bs, x_dim]
    x_star = data.measure_batch
    beta = tf.placeholder(dtype, shape=[], name="beta")
    n_particles = tf.placeholder(tf.int32, shape=[], name="n_particles")
    # xx: [bs + bs_star, x_dim]
    xx = tf.concat([x, x_star], axis=0)

    qff = bnn(xx)
    qff_prev = bnn_prev(xx)
    qf_mean, qf_var = bnn(x, full_cov=False)
    f_pred = qf_mean + tf.sqrt(qf_var) * tf.random_normal(tf.shape(qf_mean),
                                                          dtype=dtype)
    # y_pred: [bs]
    y_pred = tf.argmax(qf_mean, axis=0, output_type=tf.int32)
    # y_target: [bs]
    y_target = tf.argmax(y, axis=1, output_type=tf.int32)
    # acc: []
    acc = tf.reduce_mean(tf.to_float(tf.equal(y_pred, y_target)))

    bnn_ops = []
    bnn_prev_var = tf.trainable_variables(scope="bnn_prev")
    bnn_var = tf.trainable_variables(scope="bnn")
    for (prev, cur) in zip(bnn_prev_var, bnn_var):
        bnn_ops.append(prev.assign(cur))
    bnn_op = tf.group(bnn_ops)

    # K_prior: [bs + bs_star, bs + bs_star]
    K_prior = kernel.K(xx)
    K_prior_tril = tf.cholesky(
        K_prior + tf.eye(tf.shape(xx)[0], dtype=dtype) * gpflow.settings.jitter)
    pff = zs.distributions.MultivariateNormalCholesky(
        tf.zeros([n_cls, tf.shape(xx)[0]], dtype=dtype),
        tf.tile(K_prior_tril[None, ...], [n_cls, 1, 1]))

    # # likelihood term
    f_term = -tf.nn.softmax_cross_entropy_with_logits(
        labels=y,
        logits=tf.matrix_transpose(f_pred))
    f_term = tf.reduce_sum(f_term)
    f_term *= N / tf.cast(n, tf.float64) * beta

    # prior term
    prior_term = -beta * tf.reduce_sum(
        multivariate_normal_cross_entropy(qff, pff))

    # proximity term
    prox_term = -(1 - beta) * tf.reduce_sum(
        multivariate_normal_cross_entropy(qff, qff_prev))

    # entropy term
    entropy_term = tf.reduce_sum(multivariate_normal_entropy(qff))

    lower_bound = f_term + prior_term + prox_term + entropy_term

    gpnet_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    infer_gpnet = gpnet_opt.minimize(-lower_bound, var_list=bnn_var)
    print_freq = 1
    test_freq = 100
    var_list = bnn_prev_var + bnn_var
    sess.run(tf.variables_initializer(var_list=var_list + gpnet_opt.variables()))
    train_stats = []
    for t in range(1, FLAGS.n_iters + 1):
        beta_t = FLAGS.beta0 * 1. / (1. + FLAGS.gamma * np.sqrt(t - 1))
        _, train_ll, train_acc = sess.run(
            [infer_gpnet, lower_bound, acc],
            feed_dict={data.handle: data.train_handle,
                       beta: beta_t})
        train_stats.append((train_ll, train_acc))
        sess.run(bnn_op)

        if t % print_freq == 0:
            train_lls, train_accs = list(zip(*train_stats))
            logger.info("Iter {}, lower bound = {:.4f}, train acc = {:.4f}"
                        .format(t, np.mean(train_lls), np.mean(train_accs)))
            train_stats = []

        if t % test_freq == 0:
            sess.run(data.test_iterator.initializer)
            test_stats = []
            while True:
                try:
                    test_stats.append(
                        sess.run(acc,
                                 feed_dict={data.handle: data.test_handle}))
                except tf.errors.OutOfRangeError:
                    break
            logger.info(">> Test acc = {:.4f}".format(np.mean(test_stats)))
