#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import gpflowSlim as gpflow
from scipy.cluster.vq import kmeans2

from .utils import evaluate

FLAGS = tf.flags.FLAGS


def svgp(logger, sess, data, test_writer=None, kernel=None, ard=True):
    train_x = data.orig.train_x
    N, x_dim = train_x.shape
    x_batch, y_batch = data.next_batch
    with tf.variable_scope("svgp"):
        if kernel is None:
            kernel = gpflow.kernels.RBF(x_dim, ARD=ard, name="ard")
        likelihood = gpflow.likelihoods.Gaussian(var=0.1)
        inducing_points, _ = kmeans2(train_x, FLAGS.n_inducing, minit="points")
        # inducing_points = train_x[:FLAGS.n_inducing]
        svgp = gpflow.models.SVGP(x_batch, y_batch, kernel, likelihood,
                                  Z=inducing_points, num_latent=1, num_data=N)
        obj = svgp.objective
        likelihood_var = likelihood.variance
    y_mean, y_var = svgp.predict_y(x_batch)
    svgp_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    infer_svgp = svgp_opt.minimize(obj)

    sess.run(tf.global_variables_initializer())
    for t in range(1, FLAGS.n_iters + 1):
        _, train_ll, obs_var = sess.run(
            [infer_svgp, obj, likelihood_var],
            feed_dict={data.handle: data.train_handle})
        logger.info("Iter {}, lower bound = {:.4f}, obs_var = {:.4f}"
                    .format(t, -train_ll, obs_var))

        if t % FLAGS.test_freq == 0:
            if test_writer or (t == FLAGS.n_iters):
                test_stats = evaluate(logger, sess, data, t, y_mean, y_var,
                                      test_writer)
                if t == FLAGS.n_iters:
                    return test_stats
