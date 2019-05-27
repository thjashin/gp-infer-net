#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import gpflowSlim as gpflow
import numpy as np


FLAGS = tf.flags.FLAGS


def svgp(logger, sess, data, resnet_kern):
    train_x = data.train_x
    N, x_dim = train_x.shape
    _, n_cls = data.train_y.shape
    x_batch, y_batch = data.next_batch
    y_target = tf.argmax(y_batch, axis=1, output_type=tf.int32)
    with tf.variable_scope("svgp"):
        likelihood = gpflow.likelihoods.MultiClass(n_cls)
        # inducing_points, _ = kmeans2(train_x, FLAGS.n_inducing, minit="points")
        idx = np.random.permutation(N)
        inducing_points = train_x[idx[:FLAGS.n_inducing]]
        svgp = gpflow.models.SVGP(
            x_batch, tf.cast(y_target[:, None], tf.float64), resnet_kern, likelihood,
            Z=inducing_points, num_latent=n_cls, num_data=N)
        obj = svgp.objective

    y_mean, y_var = svgp.predict_y(x_batch)
    y_pred = tf.argmax(y_mean, axis=1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.to_float(tf.equal(y_pred, y_target)))

    svgp_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    var_list = tf.trainable_variables(scope="svgp/GPModel")
    infer_svgp = svgp_opt.minimize(obj, var_list=var_list)

    test_freq = 50
    sess.run(tf.global_variables_initializer())
    for t in range(1, FLAGS.n_iters + 1):
        _, train_ll = sess.run(
            [infer_svgp, obj],
            feed_dict={data.handle: data.train_handle})
        logger.info("Iter {}, lower bound = {:.4f}".format(t, -train_ll))

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
