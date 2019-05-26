#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import os
from collections import namedtuple

import numpy as np
import tensorflow as tf

from utils.data import load_regression
from utils.log import setup_logger
from regression import exact_gp, svgp, fbnn, gpnet


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("dataset", "boston", """UCI Regression dataset.""")
tf.flags.DEFINE_integer("split", 0, """Data split [0, 20).""")
tf.flags.DEFINE_string("method", "gpnet", """Inference algorithm.""")
tf.flags.DEFINE_integer("batch_size", 500, """Total batch size.""")
tf.flags.DEFINE_integer("m", 500, """Measurement set size.""")
tf.flags.DEFINE_float("learning_rate", 0.003, """Learning rate.""")
tf.flags.DEFINE_integer("n_hidden", 1000, """Hidden layer size.""")
tf.flags.DEFINE_integer("n_layer", 1, """Hidden layer size.""")
tf.flags.DEFINE_integer("n_inducing", 100, """Number of inducing points.""")
tf.flags.DEFINE_integer("n_iters", 10000, """Number of training iterations.""")
tf.flags.DEFINE_integer("test_freq", 50, """Test frequency.""")
tf.flags.DEFINE_string("measure", "noise", "Measurement set.")
tf.flags.DEFINE_integer("pretrain", 0, "Iters of pretraining GP priors.")
tf.flags.DEFINE_string("net", "rf", "Inference network.")
tf.flags.DEFINE_bool("residual", False, """Inference network.""")
tf.flags.DEFINE_float("beta0", 1, """Initial beta value.""")
tf.flags.DEFINE_float("gamma", 1, """Beta schedule.""")
tf.flags.DEFINE_float("hyper_rate", 0.003, "Hyperparameter update rate.")
tf.flags.DEFINE_bool("hyper_anneal", False, """Hyper_rate annealed by beta""")
tf.flags.DEFINE_bool("lr_anneal", False, "learning rate annealed by beta")
tf.flags.DEFINE_bool("fix_rf_ls", False, "fix the lengthscales of rf as prior")
tf.flags.DEFINE_string("note", "", "Note for random experiments.")


def main():
    flag_values = [
        ("method", FLAGS.method),
        ("net", FLAGS.net),
        ("res", FLAGS.residual),
        ("nh", FLAGS.n_hidden),
        ("nl", FLAGS.n_layer),
        ("inducing", FLAGS.n_inducing),
        ("pre", FLAGS.pretrain),
        ("hyper_lr", FLAGS.hyper_rate),
        ("hyper_anneal", FLAGS.hyper_anneal),
        ("beta0", FLAGS.beta0),
        ("gamma", FLAGS.gamma),
        ("niter", FLAGS.n_iters),
        ("bs", FLAGS.batch_size),
        ("m", FLAGS.m),
        ("lr", FLAGS.learning_rate),
        ("measure", FLAGS.measure),
        ("lr_anneal", FLAGS.lr_anneal),
        ("fix_rf_ls", FLAGS.fix_rf_ls),
        ("note", FLAGS.note),
    ]
    flag_str = "$".join(["@".join([i[0], str(i[1])]) for i in flag_values])
    result_path = os.path.join(
        "results", "regression", FLAGS.dataset, flag_str,
        "run_{}".format(FLAGS.split))
    logger = setup_logger("regression", __file__, result_path, filename="log")
    test_writer = tf.summary.FileWriter(result_path)

    tf.set_random_seed(1234)
    np.random.seed(1234)
    uci = load_regression(FLAGS.dataset)
    if hasattr(uci, "load"):
        uci.load(split=FLAGS.split)
    train_x, train_y = uci.train_x, uci.train_y
    test_x, test_y = uci.test_x, uci.test_y
    n_train, _ = train_x.shape

    train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    test = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    train = train.shuffle(buffer_size=10000).batch(FLAGS.batch_size).repeat()
    test = test.batch(FLAGS.batch_size)

    if FLAGS.measure == "uniform":
        raise NotImplementedError()
    else:
        if FLAGS.measure == "test":
            measure = tf.data.Dataset.from_tensor_slices(test_x)
        else:
            measure = tf.data.Dataset.from_tensor_slices(train_x)
        measure = measure.shuffle(buffer_size=10000).batch(FLAGS.m).repeat()
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
    data = Data(next_batch, None, handle, train_handle, test_handle,
                test_iterator, orig=uci)

    # SVGP
    if FLAGS.method == "svgp":
        svgp(logger, sess, data, test_writer=test_writer)
    else:
        gp, _ = exact_gp(logger, sess, data, n_epochs=FLAGS.pretrain)
        if (FLAGS.measure == "noise") and (FLAGS.pretrain > 0):
            logger.info(sess.run(gp.kern.lengthscales))
            measure_batch += tf.random_normal(
                tf.shape(measure_batch),
                dtype=tf.float64) * (gp.kern.lengthscales / np.sqrt(2))
        data = data._replace(measure_batch=measure_batch)
        # fBNN
        if FLAGS.method == "fbnn":
            fbnn(logger, sess, data, prior_gp=gp, test_writer=test_writer)
        # GPIN
        if FLAGS.method == "gpnet":
            gpnet(logger, sess, data, prior_gp=gp, test_writer=test_writer)
        # weight-space RFE
        if FLAGS.method == "rfe":
            rfe(logger, sess, data, prior_gp=gp, test_writer=test_writer)

    sess.close()


if __name__ == "__main__":
    main()
