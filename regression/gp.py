#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import gpflowSlim as gpflow
import numpy as np

from .utils import compute_rmse_and_ll, TestStats


def exact_gp(logger, sess, data, n_epochs, kernel=None, ard=True, eval=True):
    N, x_dim = data.orig.train_x.shape
    train_x, train_y = data.orig.train_x, data.orig.train_y
    if N > 1000:
        indices = np.random.permutation(N)
        sub_ind = indices[:1000]
        train_x = train_x[sub_ind]
        train_y = train_y[sub_ind]
    with tf.variable_scope("exact_gp"):
        if kernel is None:
            kernel = gpflow.kernels.RBF(x_dim, ARD=ard, name="ard")
        gp = gpflow.models.GPR(train_x, train_y, kernel)
        obj = gp.objective
        likelihood_var = gp.likelihood.variance
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    infer_gp = optimizer.minimize(obj)
    sess.run(tf.global_variables_initializer())
    for t in range(1, n_epochs + 1):
        _, train_ll, obs_var = sess.run([infer_gp, obj, likelihood_var])
        logger.info("Iter {}, log marginal = {:.4f}, obs_var = {:.4f}"
                    .format(t, -train_ll, obs_var))
    test_stats = TestStats(None, None)
    if N <= 1000:
        test_y_mean, test_y_var = gp.predict_y(data.orig.test_x)
        test_y_mean_, test_y_var_ = sess.run([test_y_mean, test_y_var])
        if eval:
            rmse, ll = compute_rmse_and_ll(data, test_y_mean_, test_y_var_)
            logger.info(">> Test rmse = {:.4f}, log likelihood = {:.4f}".
                        format(rmse, ll))
        test_stats = test_stats._replace(test_y_mean=test_y_mean_,
                                         test_y_var=test_y_var_)
    return gp, test_stats
