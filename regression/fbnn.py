#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import zhusuan as zs
import gpflowSlim as gpflow

from .utils import evaluate
from utils.mvn import multivariate_normal_kl
from bnn.deep_linear import build_deep_linear
from bnn.first_order import build_first_order
from bnn.rf_expansion import build_rf_expansion

FLAGS = tf.flags.FLAGS


def fbnn(logger, sess, data, prior_gp, ard=True, test_writer=None):
    train_x = data.orig.train_x
    N, x_dim = train_x.shape

    if FLAGS.net == "tangent":
        kernel = prior_gp.kern
        bnn = build_first_order(
            "bnn", layer_sizes=[FLAGS.n_hidden] * FLAGS.n_layer, mvn=False)
        init_op = tf.no_op()
        kernel_var = []
    elif FLAGS.net == "deep":
        kernel = prior_gp.kern
        bnn = build_deep_linear(
            "bnn", layer_sizes=[FLAGS.n_hidden] * FLAGS.n_layer, mvn=False)
        init_op = tf.no_op()
        kernel_var = []
    elif FLAGS.net == "rf":
        with tf.variable_scope("kernel"):
            kernel = gpflow.kernels.RBF(x_dim, ARD=ard, name="ard")
        bnn = build_rf_expansion(
            "bnn", x_dim, FLAGS.n_hidden, kernel, mvn=False,
            fix_ls=FLAGS.fix_rf_ls, residual=FLAGS.residual)
        init_ops = []
        kernel_var = tf.trainable_variables(scope="kernel/ard")
        gp_kernel_var = tf.trainable_variables(scope="exact_gp/ard")
        for (prev, cur) in zip(kernel_var, gp_kernel_var):
            init_ops.append(prev.assign(cur))
        init_op = tf.group(init_ops)
    else:
        raise NotImplementedError("Unknown inference net.")

    x, y = data.next_batch
    n = tf.shape(x)[0]
    # x_star: [bs_star, x_dim], x: [bs, x_dim]
    y = tf.squeeze(y, axis=-1)
    x_star = data.measure_batch
    xx = tf.concat([x, x_star], axis=0)

    qff_mean, qff_cov = bnn(xx)
    qff_cov_tril = tf.cholesky(qff_cov)
    qff = zs.distributions.MultivariateNormalCholesky(qff_mean, qff_cov_tril)
    qf_mean, qf_cov = bnn(x)
    qf_var = tf.matrix_diag_part(qf_cov)

    likelihood = prior_gp.likelihood
    likelihood_var = likelihood.variance
    K_prior = kernel.K(xx, xx) + tf.eye(
        tf.shape(xx)[0], dtype=tf.float64) * gpflow.settings.jitter
    K_prior_tril = tf.cholesky(K_prior)

    # likelihood term
    f_term = tf.reduce_sum(
        likelihood.variational_expectations(qf_mean, qf_var, y))

    # kl term
    pff = zs.distributions.MultivariateNormalCholesky(
        tf.zeros(tf.shape(xx)[0], dtype=tf.float64), K_prior_tril)
    # kl_uf_star: []
    kl_ff = multivariate_normal_kl(qff, pff)

    lower_bound = f_term - kl_ff

    fbnn_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    bnn_var = tf.trainable_variables(scope="bnn")
    infer_fbnn = fbnn_opt.minimize(-lower_bound, var_list=bnn_var)

    # hyper-parameter update
    Kn_prior = K_prior[:n, :n]
    pf = zs.distributions.MultivariateNormalCholesky(
        tf.zeros(n, dtype=tf.float64), tf.cholesky(Kn_prior))
    qf = zs.distributions.MultivariateNormalCholesky(
        qf_mean, tf.cholesky(qf_cov))
    hyper_obj = f_term - multivariate_normal_kl(qf, pf)

    hyper_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.hyper_rate)
    gp_var_list = tf.trainable_variables(scope="exact_gp")
    hyper_op = hyper_opt.minimize(-hyper_obj, var_list=gp_var_list + kernel_var)

    y_mean = qf_mean[:, None]
    y_var = qf_var[:, None] + likelihood_var

    var_list = bnn_var + kernel_var
    sess.run(tf.variables_initializer(
        var_list=var_list + fbnn_opt.variables() + hyper_opt.variables()))
    sess.run(init_op)
    logger.info("prior gp var: {}".format(sess.run(likelihood_var)))
    for t in range(1, FLAGS.n_iters + 1):
        _, _, train_ll, obs_var = sess.run(
            [infer_fbnn, hyper_op, lower_bound, likelihood_var],
            feed_dict={data.handle: data.train_handle})
        logger.info("Iter {}, lower bound = {:.4f}, obs_var = {:.4f}"
                    .format(t, train_ll, obs_var))

        if t % FLAGS.test_freq == 0:
            if test_writer or (t == FLAGS.n_iters):
                test_stats = evaluate(logger, sess, data, t, y_mean, y_var,
                                      test_writer)
                if t == FLAGS.n_iters:
                    return test_stats
