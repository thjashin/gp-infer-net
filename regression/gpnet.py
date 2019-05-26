#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import zhusuan as zs
import gpflowSlim as gpflow

from .utils import evaluate
from utils.mvn import (multivariate_normal_cross_entropy,
                       multivariate_normal_entropy,
                       multivariate_normal_kl)
from bnn.first_order import build_first_order
from bnn.rf_expansion import build_rf_expansion
from bnn.deep_linear import build_deep_linear

FLAGS = tf.flags.FLAGS


def gpnet(logger, sess, data, prior_gp, ard=True, test_writer=None):
    train_x = data.orig.train_x
    N, x_dim = train_x.shape

    if FLAGS.net == "tangent":
        kernel = prior_gp.kern
        bnn_prev = build_first_order(
            "bnn_prev", layer_sizes=[FLAGS.n_hidden] * FLAGS.n_layer, mvn=False)
        bnn = build_first_order(
            "bnn", layer_sizes=[FLAGS.n_hidden] * FLAGS.n_layer)
        init_op = tf.no_op()
        kernel_op = tf.no_op()
        kernel_var = []
        kernel_prev_var = []
    elif FLAGS.net == "deep":
        kernel = prior_gp.kern
        bnn_prev = build_deep_linear(
            "bnn_prev", layer_sizes=[FLAGS.n_hidden] * FLAGS.n_layer, mvn=False)
        bnn = build_deep_linear(
            "bnn", layer_sizes=[FLAGS.n_hidden] * FLAGS.n_layer)
        init_op = tf.no_op()
        kernel_op = tf.no_op()
        kernel_var = []
        kernel_prev_var = []
    elif FLAGS.net == "rf":
        with tf.variable_scope("kernel"):
            kernel = gpflow.kernels.RBF(x_dim, ARD=ard, name="ard")
        with tf.variable_scope("kernel_prev"):
            kernel_prev = gpflow.kernels.RBF(x_dim, ARD=ard, name="ard")
        bnn_prev = build_rf_expansion(
            "bnn_prev", x_dim, FLAGS.n_hidden, kernel_prev, mvn=False,
            fix_ls=FLAGS.fix_rf_ls, residual=FLAGS.residual)
        bnn = build_rf_expansion(
            "bnn", x_dim, FLAGS.n_hidden, kernel,
            fix_ls=FLAGS.fix_rf_ls, residual=FLAGS.residual)
        init_ops = []
        kernel_var = tf.trainable_variables(scope="kernel/ard")
        gp_kernel_var = tf.trainable_variables(scope="exact_gp/ard")
        for (prev, cur) in zip(kernel_var, gp_kernel_var):
            init_ops.append(prev.assign(cur))
        init_op = tf.group(init_ops)

        kernel_ops = []
        kernel_prev_var = tf.trainable_variables(scope="kernel_prev/ard")
        for (prev, cur) in zip(kernel_prev_var, kernel_var):
            kernel_ops.append(prev.assign(cur))
        kernel_op = tf.group(kernel_ops)
    else:
        raise NotImplementedError("Unknown inference net.")

    x, y = data.next_batch
    n = tf.shape(x)[0]
    # x_star: [bs_star, x_dim], x: [bs, x_dim]
    y = tf.squeeze(y, axis=-1)
    x_star = data.measure_batch
    beta = tf.placeholder(tf.float64, shape=[], name="beta")
    xx = tf.concat([x, x_star], axis=0)

    qf_star = bnn(x_star)
    qff_mean_prev, K_prox = bnn_prev(xx)
    # qf_mean: [bs], qf_var: [bs]
    qf_mean, qf_var = bnn(x, full_cov=False)

    bnn_ops = []
    bnn_prev_var = tf.trainable_variables(scope="bnn_prev")
    bnn_var = tf.trainable_variables(scope="bnn")
    for (prev, cur) in zip(bnn_prev_var, bnn_var):
        bnn_ops.append(prev.assign(cur))
    bnn_op = tf.group(bnn_ops)

    likelihood = prior_gp.likelihood
    likelihood_var = likelihood.variance
    K_prior = kernel.K(xx, xx) + tf.eye(
        tf.shape(xx)[0], dtype=tf.float64) * gpflow.settings.jitter

    K_sum_tril = tf.cholesky(K_prox * beta + K_prior * (1 - beta))
    K_sum_tril_inv = tf.matrix_triangular_solve(
        K_sum_tril, tf.eye(tf.shape(xx)[0], dtype=tf.float64))
    K_sum_inv = tf.matmul(K_sum_tril_inv, K_sum_tril_inv, transpose_a=True)
    K_adapt = tf.matmul(K_prior, tf.matmul(K_sum_inv, K_prox))
    # mean_adapt: [bs + bs_star, 1]
    mean_adapt = (1 - beta) * tf.matmul(
        K_adapt, tf.matrix_solve(K_prox, qff_mean_prev[..., None]))
    mean_n, mean_m = mean_adapt[:n, :], mean_adapt[n:, :]
    Kn, Knm, Km = K_adapt[:n, :n], K_adapt[:n, n:], K_adapt[n:, n:]

    Ky = Kn + tf.eye(n, dtype=tf.float64) * likelihood_var / (N / n * beta)
    Ky_tril = tf.cholesky(Ky)

    mean_target = tf.matmul(
        Knm, tf.cholesky_solve(Ky_tril, y[..., None] - mean_n),
        transpose_a=True) + mean_m
    mean_target = tf.squeeze(mean_target, -1)
    K_target = Km - tf.matmul(Knm, tf.cholesky_solve(Ky_tril, Knm),
                              transpose_a=True)
    K_target_tril = tf.cholesky(K_target)
    target_pf_star = zs.distributions.MultivariateNormalCholesky(
        mean_target, K_target_tril)

    kl_obj = multivariate_normal_kl(qf_star, target_pf_star)

    if FLAGS.lr_anneal:
        gpnet_opt = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate * beta / FLAGS.beta0)
    else:
        gpnet_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    infer_gpnet = gpnet_opt.minimize(kl_obj, var_list=bnn_var)

    # hyper-parameter update
    Kn_prior = K_prior[:n, :n]
    pf = zs.distributions.MultivariateNormalCholesky(
        tf.zeros(n, dtype=tf.float64), tf.cholesky(Kn_prior))
    Kn_prox = K_prox[:n, :n]
    qf_prev_mean = qff_mean_prev[:n]
    qf_prev_var = tf.matrix_diag_part(Kn_prox)
    qf_prev = zs.distributions.MultivariateNormalCholesky(
        qf_prev_mean, tf.cholesky(Kn_prox))
    hyper_obj = tf.reduce_sum(likelihood.variational_expectations(
        qf_prev_mean, qf_prev_var, y)) - multivariate_normal_kl(qf_prev, pf)

    if FLAGS.hyper_anneal:
        hyper_opt = tf.train.AdamOptimizer(
            learning_rate=FLAGS.hyper_rate * beta / FLAGS.beta0)
    else:
        hyper_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.hyper_rate)
    gp_var_list = tf.trainable_variables(scope="exact_gp")
    hyper_op = hyper_opt.minimize(-hyper_obj, var_list=gp_var_list + kernel_var)

    y_mean = qf_mean[:, None]
    y_var = qf_var[:, None] + likelihood_var

    var_list = bnn_prev_var + bnn_var + kernel_var + kernel_prev_var
    sess.run(tf.variables_initializer(
        var_list=var_list + gpnet_opt.variables() + hyper_opt.variables()))
    sess.run(init_op)
    logger.info("prior gp var: {}".format(sess.run(likelihood_var)))
    for t in range(1, FLAGS.n_iters + 1):
        beta_t = FLAGS.beta0 * 1. / (1. + FLAGS.gamma * np.sqrt(t - 1))
        _, _, train_obj, obs_var = sess.run(
            [infer_gpnet, hyper_op, kl_obj, likelihood_var],
            feed_dict={data.handle: data.train_handle, beta: beta_t})
        sess.run([bnn_op, kernel_op])
        logger.info("Iter {}, kl_obj = {:.4f}, obs_var = {:.4f}"
                    .format(t, train_obj, obs_var))

        if t % FLAGS.test_freq == 0:
            if test_writer or (t == FLAGS.n_iters):
                test_stats = evaluate(logger, sess, data, t, y_mean, y_var,
                                      test_writer)
                if t == FLAGS.n_iters:
                    return test_stats


def gpnet_nonconj(logger, sess, data, prior_gp, ard=True, test_writer=None):
    train_x = data.orig.train_x
    N, x_dim = train_x.shape

    if FLAGS.net == "tangent":
        kernel = prior_gp.kern
        bnn_prev = build_first_order(
            "bnn_prev", layer_sizes=[FLAGS.n_hidden] * FLAGS.n_layer, mvn=False)
        bnn = build_first_order(
            "bnn", layer_sizes=[FLAGS.n_hidden] * FLAGS.n_layer)
        init_op = tf.no_op()
        kernel_op = tf.no_op()
        kernel_var = []
        kernel_prev_var = []
    elif FLAGS.net == "deep":
        kernel = prior_gp.kern
        bnn_prev = build_deep_linear(
            "bnn_prev", layer_sizes=[FLAGS.n_hidden] * FLAGS.n_layer, mvn=False)
        bnn = build_deep_linear(
            "bnn", layer_sizes=[FLAGS.n_hidden] * FLAGS.n_layer)
        init_op = tf.no_op()
        kernel_op = tf.no_op()
        kernel_var = []
        kernel_prev_var = []
    elif FLAGS.net == "rf":
        with tf.variable_scope("kernel"):
            kernel = gpflow.kernels.RBF(x_dim, ARD=ard, name="ard")
        with tf.variable_scope("kernel_prev"):
            kernel_prev = gpflow.kernels.RBF(x_dim, ARD=ard, name="ard")
        bnn_prev = build_rf_expansion(
            "bnn_prev", x_dim, FLAGS.n_hidden, kernel_prev, mvn=False,
            fix_ls=FLAGS.fix_rf_ls, residual=FLAGS.residual)
        bnn = build_rf_expansion(
            "bnn", x_dim, FLAGS.n_hidden, kernel,
            fix_ls=FLAGS.fix_rf_ls, residual=FLAGS.residual)
        init_ops = []
        kernel_var = tf.trainable_variables(scope="kernel/ard")
        gp_kernel_var = tf.trainable_variables(scope="exact_gp/ard")
        for (prev, cur) in zip(kernel_var, gp_kernel_var):
            init_ops.append(prev.assign(cur))
        init_op = tf.group(init_ops)

        kernel_ops = []
        kernel_prev_var = tf.trainable_variables(scope="kernel_prev/ard")
        for (prev, cur) in zip(kernel_prev_var, kernel_var):
            kernel_ops.append(prev.assign(cur))
        kernel_op = tf.group(kernel_ops)
    else:
        raise NotImplementedError("Unknown inference net.")

    x, y = data.next_batch
    n = tf.shape(x)[0]
    # x_star: [bs_star, x_dim], x: [bs, x_dim]
    y = tf.squeeze(y, axis=-1)
    x_star = data.measure_batch
    beta = tf.placeholder(tf.float64, shape=[], name="beta")
    xx = tf.concat([x, x_star], axis=0)

    qff = bnn(xx)
    qff_mean_prev, K_prox = bnn_prev(xx)
    # qf_mean: [bs], qf_var: [bs]
    qf_mean, qf_var = bnn(x, full_cov=False)

    bnn_ops = []
    bnn_prev_var = tf.trainable_variables(scope="bnn_prev")
    bnn_var = tf.trainable_variables(scope="bnn")
    for (prev, cur) in zip(bnn_prev_var, bnn_var):
        bnn_ops.append(prev.assign(cur))
    bnn_op = tf.group(bnn_ops)

    likelihood = prior_gp.likelihood
    likelihood_var = likelihood.variance
    K_prior = kernel.K(xx, xx) + tf.eye(
        tf.shape(xx)[0], dtype=tf.float64) * gpflow.settings.jitter
    K_prior_tril = tf.cholesky(K_prior)
    pff = zs.distributions.MultivariateNormalCholesky(
        tf.zeros(tf.shape(xx)[0], dtype=tf.float64), K_prior_tril)

    # likelihood term
    f_term = tf.reduce_sum(
        likelihood.variational_expectations(qf_mean, qf_var, y))
    f_term *= N / tf.cast(tf.shape(x)[0], tf.float64) * beta
    # f_term = tf.Print(f_term, [f_term], message="f_term:")

    # prior term
    prior_term = -beta * multivariate_normal_cross_entropy(qff, pff)

    # proximity term
    qff_prev = zs.distributions.MultivariateNormalCholesky(
        qff_mean_prev, tf.cholesky(K_prox))
    prox_term = -(1 - beta) * multivariate_normal_cross_entropy(qff, qff_prev)

    # entropy term
    entropy_term = multivariate_normal_entropy(qff)

    lower_bound = f_term + prior_term + prox_term + entropy_term

    if FLAGS.lr_anneal:
        gpnet_opt = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate * beta / FLAGS.beta0)
    else:
        gpnet_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    infer_gpnet = gpnet_opt.minimize(-lower_bound, var_list=bnn_var)

    # hyper-parameter update
    Kn_prior = K_prior[:n, :n]
    pf = zs.distributions.MultivariateNormalCholesky(
        tf.zeros(n, dtype=tf.float64), tf.cholesky(Kn_prior))
    Kn_prox = K_prox[:n, :n]
    qf_prev_mean = qff_mean_prev[:n]
    qf_prev_var = tf.matrix_diag_part(Kn_prox)
    qf_prev = zs.distributions.MultivariateNormalCholesky(
        qf_prev_mean, tf.cholesky(Kn_prox))
    hyper_obj = tf.reduce_sum(likelihood.variational_expectations(
        qf_prev_mean, qf_prev_var, y)) - multivariate_normal_kl(qf_prev, pf)

    if FLAGS.hyper_anneal:
        hyper_opt = tf.train.AdamOptimizer(
            learning_rate=FLAGS.hyper_rate * beta / FLAGS.beta0)
    else:
        hyper_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.hyper_rate)
    gp_var_list = tf.trainable_variables(scope="exact_gp")
    hyper_op = hyper_opt.minimize(-hyper_obj, var_list=gp_var_list + kernel_var)

    y_mean = qf_mean[:, None]
    y_var = qf_var[:, None] + likelihood_var

    var_list = bnn_prev_var + bnn_var + kernel_var + kernel_prev_var
    sess.run(tf.variables_initializer(
        var_list=var_list + gpnet_opt.variables() + hyper_opt.variables()))
    sess.run(init_op)
    logger.info("prior gp var: {}".format(sess.run(likelihood_var)))
    for t in range(1, FLAGS.n_iters + 1):
        beta_t = FLAGS.beta0 * 1. / (1. + FLAGS.gamma * np.sqrt(t - 1))
        _, _, train_ll, obs_var = sess.run(
            [infer_gpnet, hyper_op, lower_bound, likelihood_var],
            feed_dict={data.handle: data.train_handle, beta: beta_t})
        sess.run([bnn_op, kernel_op])
        logger.info("Iter {}, lower bound = {:.4f}, obs_var = {:.4f}"
                    .format(t, train_ll, obs_var))

        if t % FLAGS.test_freq == 0:
            if test_writer or (t == FLAGS.n_iters):
                test_stats = evaluate(logger, sess, data, t, y_mean, y_var,
                                      test_writer)
                if t == FLAGS.n_iters:
                    return test_stats
