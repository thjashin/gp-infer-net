#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import os
from collections import namedtuple

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes()
sns.set_style("white")

from utils.data import load_snelson_data
from regression import exact_gp, svgp, fbnn, gpnet, gpnet_nonconj
from utils.log import setup_logger


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("dataset", "snelson", """Toy dataset (snelson).""")
tf.flags.DEFINE_string("method", "gpnet", """Inference algorithm.""")
tf.flags.DEFINE_integer("batch_size", 20, """Total batch size.""")
tf.flags.DEFINE_integer("m", 20, """Measurement set size.""")
tf.flags.DEFINE_float("learning_rate", 0.003, """Learning rate.""")
tf.flags.DEFINE_integer("n_hidden", 20, """Hidden layer size.""")
tf.flags.DEFINE_integer("n_layer", 1, """Hidden layer size.""")
tf.flags.DEFINE_integer("n_inducing", 20, """Number of inducing points.""")
tf.flags.DEFINE_integer("n_iters", 40000, """Number of training iterations.""")
tf.flags.DEFINE_integer("test_freq", 50, """Test frequency.""")
tf.flags.DEFINE_string("measure", "uniform", """Measurement set.""")
tf.flags.DEFINE_bool("pretrain", False, """Iters of pretraining GP priors.""")
tf.flags.DEFINE_string("net", "rf", """Inference network.""")
tf.flags.DEFINE_bool("residual", False, """Inference network.""")
tf.flags.DEFINE_float("beta0", 1, """Initial beta value.""")
tf.flags.DEFINE_float("gamma", 0.1, """Beta schedule.""")
tf.flags.DEFINE_float("hyper_rate", 0.003, """Hyperparameter update rate.""")
tf.flags.DEFINE_bool("hyper_anneal", False, """Hyper_rate annealed by beta""")
tf.flags.DEFINE_bool("lr_anneal", False, "learning rate annealed by beta")
tf.flags.DEFINE_bool("fix_rf_ls", False, "fix the lengthscales of rf as prior")
tf.flags.DEFINE_string("note", "", """Note for random experiments.""")


def set_up_figure(data, test_stats):
    plt.figure(figsize=(12, 8))
    plt.scatter(
        data.orig.train_x.squeeze(-1), data.orig.train_y.squeeze(-1), c="k")
    plot_ground_truth(data.orig.test_x.squeeze(-1),
                      test_stats.test_y_mean.squeeze(-1),
                      test_stats.test_y_var.squeeze(-1))


def plot_ground_truth(test_x, test_y_mean_, test_y_var_):
    plt.plot(test_x, test_y_mean_, c="k", linewidth=2)
    plt.plot(test_x, test_y_mean_ + 3. * np.sqrt(test_y_var_), '--',
             color="k", linewidth=2)
    plt.plot(test_x, test_y_mean_ - 3. * np.sqrt(test_y_var_), '--',
             color="k", linewidth=2)


def plot_method(data, test_stats, color):
    test_x = data.orig.test_x.squeeze(-1)
    test_y_mean = test_stats.test_y_mean.squeeze(-1)
    test_y_var = test_stats.test_y_var.squeeze(-1)
    plt.plot(test_x, test_y_mean, c=color, linewidth=2)
    plt.fill_between(test_x,
                     test_y_mean + 3. * np.sqrt(test_y_var),
                     test_y_mean - 3. * np.sqrt(test_y_var),
                     alpha=0.2, color=color)


def save_figure(name, test_x):
    plt.xlim(test_x.min(), test_x.max())
    # plt.xlim([-1, 7])
    # plt.ylim([-3, 2])
    plt.ylim([-5, 5])
    plt.savefig(name, bbox_inches='tight')


def main():
    result_path = os.path.join("results", "toy", FLAGS.dataset, FLAGS.method)
    logger = setup_logger("toy", __file__, result_path, filename="log")

    tf.set_random_seed(1234)
    np.random.seed(1234)

    toy = load_snelson_data(n=100)
    train_x, train_y = toy.train_x, toy.train_y
    test_x, test_y = toy.test_x, np.zeros_like(toy.test_x)

    train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    test = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    train = train.shuffle(buffer_size=100).batch(FLAGS.batch_size).repeat()
    test = test.batch(FLAGS.batch_size)

    if FLAGS.measure == "uniform":
        measure_batch = tf.random_uniform(
            [FLAGS.m], toy.x_min, toy.x_max, dtype=tf.float64)[:, None]
    else:
        measure = tf.data.Dataset.from_tensor_slices(train_x)
        measure = measure.shuffle(buffer_size=20).batch(FLAGS.m).repeat()
        measure_iterator = measure.make_one_shot_iterator()
        measure_batch = measure_iterator.get_next()

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train.output_types, train.output_shapes)
    next_batch = iterator.get_next()

    train_iterator = train.make_one_shot_iterator()
    test_iterator = test.make_initializable_iterator()

    sess = tf.Session()

    train_handle = sess.run(train_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())

    Data = namedtuple("Data", [
        "next_batch",
        "measure_batch",
        "handle",
        "train_handle",
        "test_handle",
        "test_iterator",
        "orig"])
    data = Data(next_batch, measure_batch, handle, train_handle, test_handle,
                test_iterator, orig=toy)

    if not FLAGS.pretrain:
        with tf.variable_scope("truth"):
            _, true_stats = exact_gp(logger, sess, data,
                                     n_epochs=100, ard=False, eval=False)
        gp, _ = exact_gp(logger, sess, data,
                         n_epochs=0, ard=False, eval=False)
    else:
        gp, true_stats = exact_gp(logger, sess, data,
                                  n_epochs=100, ard=False, eval=False)
        if FLAGS.measure == "noise":
            measure_batch += tf.random_normal(
                tf.shape(measure_batch),
                dtype=tf.float64) * (gp.kern.lengthscales / np.sqrt(2))

    set_up_figure(data, true_stats)

    if FLAGS.method == "svgp":
        test_stats = svgp(logger, sess, data, ard=False)
        plot_method(data, test_stats, "b")
        path = os.path.join(result_path, "svgp-{}.png").format(FLAGS.n_inducing)
        save_figure(path, test_x)

    if FLAGS.method == "fbnn":
        test_stats = fbnn(logger, sess, data, prior_gp=gp, ard=False)
        plot_method(data, test_stats, "r")
        path = os.path.join(result_path, "fbnn-{}-{}.png").format(
            FLAGS.m, FLAGS.net)
        save_figure(path, test_x)

    if FLAGS.method == "gpnet":
        test_stats = gpnet(logger, sess, data, prior_gp=gp, ard=False)
        plot_method(data, test_stats, "g")
        path = os.path.join(result_path, "gpnet-{}-{}.png").format(
            FLAGS.m, FLAGS.net)
        save_figure(path, test_x)

    if FLAGS.method == "gpnet_nonconj":
        test_stats = gpnet_nonconj(logger, sess, data, prior_gp=gp, ard=False)
        plot_method(data, test_stats, "g")
        path = os.path.join(result_path, "gpnet-nonconj-{}-{}.png").format(
            FLAGS.m, FLAGS.net)
        save_figure(path, test_x)

    sess.close()


if __name__ == "__main__":
    main()
