#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from collections import namedtuple

import tensorflow as tf
import numpy as np
from scipy import stats


TestStats = namedtuple("TestStats", ["test_y_mean", "test_y_var"])


def compute_rmse_and_ll(data, yn_mean, yn_var):
    rmse = np.sqrt(
        np.mean(np.square(data.orig.test_y - yn_mean))) * np.squeeze(
        data.orig.std_y)
    lls = stats.norm.logpdf(
        data.orig.test_y, yn_mean, np.sqrt(yn_var)) - np.log(data.orig.std_y)
    ll = np.mean(lls)
    return rmse, ll


def evaluate(logger, sess, data, t, y_mean, y_var, test_writer=None):
    sess.run(data.test_iterator.initializer)
    test_stats = []
    while True:
        try:
            test_stats.append(
                sess.run([y_mean, y_var],
                         feed_dict={data.handle: data.test_handle}))
        except tf.errors.OutOfRangeError:
            break
    test_y_means, test_y_vars = list(zip(*test_stats))
    test_y_mean = np.vstack(test_y_means)
    test_y_var = np.vstack(test_y_vars)
    if test_writer:
        rmse, ll = compute_rmse_and_ll(data, test_y_mean, test_y_var)
        write_summary(test_writer, rmse, ll, t)
        logger.info(">> Test rmse = {:.4f}, log likelihood = {:.4f}"
                    .format(rmse, ll))
    return TestStats(test_y_mean, test_y_var)


def write_summary(test_writer, rmse, ll, epoch):
    summary = tf.Summary()
    summary.value.add(tag="rmse", simple_value=rmse)
    summary.value.add(tag="ll", simple_value=ll)
    test_writer.add_summary(summary, epoch)
    test_writer.flush()
