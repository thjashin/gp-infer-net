# Copyright 2019 Jiaxin Shi
# Copyright 2017 https://github.com/rhaps0dy/convnets-as-gps
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from __future__ import division

import gpflowSlim as gpflow
import tensorflow as tf

from .conv import ConvKernelBase


@gpflow.decors.name_scope()
def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in].
        kernel_size: The kernel to be used in the conv2d or max_pool2d
                     operation. Should be a positive integer.
    Returns:
        A tensor with the same format as the input with the data either intact
        (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    if pad_total == 0:
        return inputs
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if data_format == "NCHW":
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, var, kernel_size, strides,
                         data_format, name='conv2d_fixed_padding'):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    with tf.name_scope(name):
        if strides > 1:
            inputs = fixed_padding(inputs, kernel_size, data_format)
        chan_idx = data_format.index("C")
        try:
            C = int(inputs.shape[chan_idx])
        except TypeError:
            C = tf.shape(inputs)[chan_idx]
        fan_in = C * (kernel_size * kernel_size)
        W = tf.fill([kernel_size, kernel_size, C, 1], var/fan_in)
        if data_format == "NCHW":
            strides_shape = [1, 1, strides, strides]
        else:
            strides_shape = [1, strides, strides, 1]
        return tf.nn.conv2d(
            input=inputs, filter=W, strides=strides_shape,
            padding=('SAME' if strides == 1 else 'VALID'),
            data_format=data_format)


class ResnetKernel(ConvKernelBase):
    "Kernel equivalent to Resnet V2 (tensorflow/models/official/resnet)"
    @gpflow.decors.name_scope()
    def headless_network(self, inputs, apply_recurse_kern):
        """
        Apply the network that this kernel defines, except the last dense layer.
        The last dense layer is different for K and Kdiag.
        """
        # Copy from resnet_model.py
        inputs = conv2d_fixed_padding(
            inputs=inputs, var=self.var_weight,
            kernel_size=self.kernel_size,
            strides=self.conv_stride,
            data_format=self.data_format,
            name='initial_conv')

        for i, num_blocks in enumerate(self.block_sizes):
            with tf.name_scope("block_layer_{}".format(i+1)):
                # Only the first block per block_layer uses strides
                # and strides
                inputs = self.block_v2(inputs, True, self.block_strides[i],
                                       apply_recurse_kern)
                print("First layer of block {}:".format(i), inputs)
                for j in range(1, num_blocks):
                    inputs = self.block_v2(inputs, False, 1, apply_recurse_kern)
                    print("{}th layer of block {}:".format(j, i), inputs)
        # Dense layer
        inputs = tf.reduce_mean(inputs, axis=(1, 2, 3))
        return self.var_bias + self.var_weight * inputs

    @gpflow.decors.name_scope()
    def block_v2(self, inputs, projection_shortcut, strides, apply_recurse_kern):
        shortcut = inputs
        inputs = apply_recurse_kern(inputs)
        if projection_shortcut:
            # Need to project the inputs to a smaller space and also apply ReLU
            del shortcut
            shortcut = conv2d_fixed_padding(
                inputs=inputs, var=self.var_weight, kernel_size=1,
                strides=strides, data_format=self.data_format,
                name='projection_shortcut')

        inputs = conv2d_fixed_padding(
            inputs=inputs, var=self.var_weight, kernel_size=3, strides=strides,
            data_format=self.data_format)
        inputs = apply_recurse_kern(inputs)
        inputs = conv2d_fixed_padding(
            inputs=inputs, var=self.var_weight, kernel_size=3, strides=1,
            data_format=self.data_format)
        return inputs + shortcut
