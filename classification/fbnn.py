#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import zhusuan as zs
import gpflowSlim as gpflow

from bnn.multi_output.resnet import build_resnet
from utils.mvn import multivariate_normal_kl


FLAGS = tf.flags.FLAGS


def fbnn(logger, sess, data, kernel, dtype=tf.float64):
    train_x = data.train_x
    N, x_dim = train_x.shape
    _, n_cls = data.train_y.shape

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
    # x_star: [bs_star, x_dim], x: [bs, x_dim]
    x_star = data.measure_batch
    # xx: [bs + bs_star, x_dim]
    xx = tf.concat([x, x_star], axis=0)

    qff = bnn(xx)
    # qf_mean: [n_cls, bs], qf_var: [n_cls, bs], f_pred: [n_cls, bs]
    qf_mean, qf_var = bnn(x, full_cov=False)
    f_pred = qf_mean + tf.sqrt(qf_var) * tf.random_normal(tf.shape(qf_mean),
                                                          dtype=dtype)
    # y_pred: [bs]
    y_pred = tf.argmax(qf_mean, axis=0, output_type=tf.int32)
    # y_target: [bs]
    y_target = tf.argmax(y, axis=1, output_type=tf.int32)
    # acc: []
    acc = tf.reduce_mean(tf.to_float(tf.equal(y_pred, y_target)))

    # K_prior: [bs + bs_star, bs + bs_star]
    K_prior = kernel.K(xx)
    K_prior_tril = tf.cholesky(
        K_prior + tf.eye(tf.shape(xx)[0], dtype=dtype) * gpflow.settings.jitter)
    pff = zs.distributions.MultivariateNormalCholesky(
        tf.zeros([n_cls, tf.shape(xx)[0]], dtype=dtype),
        tf.tile(K_prior_tril[None, ...], [n_cls, 1, 1]))

    # likelihood term
    f_term = -tf.nn.softmax_cross_entropy_with_logits(
        labels=y,
        logits=tf.matrix_transpose(f_pred))
    f_term = tf.reduce_sum(f_term)

    # kl term
    kl_term = tf.reduce_sum(multivariate_normal_kl(qff, pff))

    lower_bound = f_term - kl_term

    fbnn_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    bnn_var = tf.trainable_variables(scope="bnn")
    infer_fbnn = fbnn_opt.minimize(-lower_bound, var_list=bnn_var)
    print_freq = 1
    test_freq = 100
    sess.run(tf.variables_initializer(var_list=bnn_var + fbnn_opt.variables()))
    train_stats = []
    for t in range(1, FLAGS.n_iters + 1):
        _, train_ll, train_acc = sess.run(
            [infer_fbnn, lower_bound, acc],
            feed_dict={data.handle: data.train_handle})
        train_stats.append((train_ll, train_acc))

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
