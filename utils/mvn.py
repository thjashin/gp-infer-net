#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf


def kl_normal_normal(mean1, logstd1, mean2, logstd2):
    return logstd2 - logstd1 + (tf.exp(2 * logstd1) + (mean1 - mean2) ** 2) / \
        (2 * tf.exp(2 * logstd2)) - 0.5


def multivariate_normal_kl(q, p, dtype=tf.float64):
    # return KL(q, p)
    kl = -0.5 * tf.cast(tf.shape(q.mean)[-1], dtype)
    kl -= tf.reduce_sum(tf.log(tf.matrix_diag_part(q.cov_tril)), axis=-1)
    kl += tf.reduce_sum(tf.log(tf.matrix_diag_part(p.cov_tril)), axis=-1)
    p_inv_q = tf.matrix_triangular_solve(p.cov_tril, q.cov_tril, lower=True)
    kl += 0.5 * tf.reduce_sum(tf.square(p_inv_q), axis=(-1, -2))
    Kinv_m = tf.cholesky_solve(p.cov_tril, tf.expand_dims(q.mean - p.mean, -1))
    kl += 0.5 * tf.reduce_sum((q.mean - p.mean) * tf.squeeze(Kinv_m, -1),
                              axis=-1)
    return kl


def multivariate_normal_entropy(p, dtype=tf.float64):
    # KL(p, q) = E_p \log p - E_p \log q
    # return -E_p \log p
    d = tf.cast(tf.shape(p.mean)[-1], dtype)
    ret = 0.5 * d * tf.log(2 * tf.constant(np.pi, dtype))
    ret += tf.reduce_sum(tf.log(tf.matrix_diag_part(p.cov_tril)), axis=-1)
    ret += 0.5 * d
    return ret


def multivariate_normal_cross_entropy(p, q, dtype=tf.float64):
    # return -E_p \log q
    d = tf.cast(tf.shape(p.mean)[-1], dtype)
    ret = 0.5 * d * tf.log(2 * tf.constant(np.pi, dtype))
    ret += tf.reduce_sum(tf.log(tf.matrix_diag_part(q.cov_tril)), axis=-1)
    q_inv_p = tf.matrix_triangular_solve(q.cov_tril, p.cov_tril, lower=True)
    ret += 0.5 * tf.reduce_sum(tf.square(q_inv_p), axis=(-1, -2))
    Kinv_m = tf.cholesky_solve(q.cov_tril, tf.expand_dims(p.mean - q.mean, -1))
    ret += 0.5 * tf.reduce_sum((p.mean - q.mean) * tf.squeeze(Kinv_m, -1),
                               axis=-1)
    return ret
