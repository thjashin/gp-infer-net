#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import os
from collections import namedtuple

import tensorflow as tf
import numpy as np

from kernel.resnet import ResnetKernel
from kernel.elementwise import ReLUKernel
from utils.log import setup_logger
from utils.data import load_mnist_realval, load_cifar10
from classification import svgp, gpnet, gpnet_nonconj, fbnn


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("dataset", "mnist", "Dataset.")
tf.flags.DEFINE_string("method", "gpnet_nonconj", """Inference method.""")
tf.flags.DEFINE_string("net", "tangent", "Inference network.")
tf.flags.DEFINE_integer("batch_size", 128, """Total batch size.""")
tf.flags.DEFINE_float("learning_rate", 3e-4, """Learning rate.""")
tf.flags.DEFINE_integer("n_inducing", 100, """Number of inducing points.""")
tf.flags.DEFINE_string("measure", "train", "Measurement set.")
tf.flags.DEFINE_float("hyper_rate", 0, "Hyperparameter update rate.")
tf.flags.DEFINE_integer("block_size", 2, "number of blocks for each size.")
tf.flags.DEFINE_float("beta0", 0.01, """Initial beta value.""")
tf.flags.DEFINE_float("gamma", 0.1, """Beta schedule.""")
tf.flags.DEFINE_integer("n_iters", 10000, """Number of training iterations.""")
tf.flags.DEFINE_string("note", "", "Note for random experiments.")


def main():
    flag_values = [
        ("method", FLAGS.method),
        ("net", FLAGS.net),
        ("inducing", FLAGS.n_inducing),
        ("beta0", FLAGS.beta0),
        ("gamma", FLAGS.gamma),
        ("niter", FLAGS.n_iters),
        ("bs", FLAGS.batch_size // 2),
        ("m", FLAGS.batch_size // 2),
        ("lr", FLAGS.learning_rate),
        ("measure", FLAGS.measure),
        ("hyper_rate", FLAGS.hyper_rate),
        ("block", FLAGS.block_size),
        ("note", FLAGS.note),
    ]
    flag_str = "$".join(["@".join([i[0], str(i[1])]) for i in flag_values])
    result_path = os.path.join(
        "results", "classification", FLAGS.dataset, flag_str)
    logger = setup_logger("classification", __file__, result_path,
                          filename="log")

    np.random.seed(1234)
    tf.set_random_seed(1234)

    # Load MNIST
    if FLAGS.dataset == "mnist":
        train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist_realval(
            dtype=np.float64)
        train_x = np.vstack([train_x, valid_x])
        train_y = np.vstack([train_y, valid_y])
        input_shape = [1, 28, 28]
    elif FLAGS.dataset == "cifar10":
        train_x, train_y, test_x, test_y = load_cifar10(
            dtype=np.float64)
        input_shape = [3, 32, 32]
    else:
        raise NotImplementedError()

    train_x = 2 * train_x - 1
    test_x = 2 * test_x - 1

    train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    test = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    train = train.shuffle(buffer_size=1000).batch(
        FLAGS.batch_size // 2).repeat()
    test = test.batch(FLAGS.batch_size * 4)

    if FLAGS.measure == "test_x":
        measure = tf.data.Dataset.from_tensor_slices(test_x)
    else:
        measure = tf.data.Dataset.from_tensor_slices(train_x)
    measure = measure.shuffle(buffer_size=1000).batch(
        FLAGS.batch_size // 2).repeat()
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
        "train_x",
        "train_y"])
    data = Data(next_batch, measure_batch, handle, train_handle, test_handle,
                test_iterator, train_x, train_y)

    block_sizes = [FLAGS.block_size] * 3
    block_strides = [1, 2, 2]
    with tf.variable_scope("prior"):
        resnet_kern = ResnetKernel(
            input_shape=input_shape,
            block_sizes=block_sizes,
            block_strides=block_strides,
            kernel_size=3,
            recurse_kern=ReLUKernel(),
            var_weight=1.,
            var_bias=0.,
            conv_stride=1,
            data_format="NCHW",
            dtype=tf.float64,
        )

    sess.run(tf.variables_initializer(tf.trainable_variables("prior")))

    # SVGP
    if FLAGS.method == "svgp":
        svgp(logger, sess, data, resnet_kern)
    elif FLAGS.method == "gpnet":
        gpnet(logger, sess, data, resnet_kern, dtype=tf.float64)
    elif FLAGS.method == "gpnet_nonconj":
        gpnet_nonconj(logger, sess, data, resnet_kern, dtype=tf.float64)
    elif FLAGS.method == "fbnn":
        fbnn(logger, sess, data, resnet_kern, dtype=tf.float64)


if __name__ == "__main__":
    main()
