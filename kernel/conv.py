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
from typing import List
import numpy as np
import tensorflow as tf
import abc

from .elementwise import ElementwiseKernel


class ConvKernelBase(gpflow.kernels.Kernel, metaclass=abc.ABCMeta):
    "General kernel for deep networks"
    def __init__(self,
                 input_shape: List[int],
                 block_sizes: List[int],
                 block_strides: List[int],
                 kernel_size: int,
                 recurse_kern: ElementwiseKernel,
                 var_weight: float = 1.0,
                 var_bias: float = 1.0,
                 conv_stride: int = 1,
                 active_dims: slice = None,
                 data_format: str = "NCHW",
                 input_type = None,
                 name: str = None,
                 dtype = tf.float32):
        input_dim = np.prod(input_shape)
        super(ConvKernelBase, self).__init__(input_dim, active_dims, name=name)

        self.input_shape = list(np.copy(input_shape))
        self.block_sizes = np.copy(block_sizes).astype(np.int32)
        self.block_strides = np.copy(block_strides).astype(np.int32)
        self.kernel_size = kernel_size
        self.recurse_kern = recurse_kern
        self.conv_stride = conv_stride
        self.data_format = data_format
        if input_type is None:
            input_type = dtype
        self.input_type = input_type
        self.dtype = dtype

        self._var_weight = gpflow.params.Parameter(
            var_weight, gpflow.transforms.positive, dtype=self.input_type,
            name="var_weight")
        self._var_bias = gpflow.params.Parameter(
            var_bias, gpflow.transforms.positive, dtype=self.input_type,
            name="var_bias")

    @property
    def var_weight(self):
        return self._var_weight.value

    @property
    def var_bias(self):
        return self._var_bias.value

    @gpflow.decors.name_scope()
    def K(self, X, X2=None):
        # Concatenate the covariance between X and X2 and their respective
        # variances. Only 1 variance is needed if X2 is None.
        if X.dtype != self.input_type or (
                X2 is not None and X2.dtype != self.input_type):
            raise TypeError("Input dtypes are wrong: {} or {} are not {}"
                            .format(X.dtype, X2.dtype, self.input_type))
        if X2 is None:
            N = N2 = tf.shape(X)[0]
            var_z_list = [
                tf.reshape(tf.square(X), [N] + self.input_shape),
                tf.reshape(X[:, None, :] * X, [N*N] + self.input_shape)]

            @gpflow.decors.name_scope("apply_recurse_kern_X_X")
            def apply_recurse_kern(var_a_all, concat_outputs=True):
                var_a_1 = var_a_all[:N]
                var_a_cross = var_a_all[N:]
                vz = [self.recurse_kern.Kdiag(var_a_1),
                      self.recurse_kern.K(var_a_cross, var_a_1, None)]
                if concat_outputs:
                    return tf.concat(vz, axis=0)
                return vz

        else:
            N, N2 = tf.shape(X)[0], tf.shape(X2)[0]
            var_z_list = [
                tf.reshape(tf.square(X), [N] + self.input_shape),
                tf.reshape(tf.square(X2), [N2] + self.input_shape),
                tf.reshape(X[:, None, :] * X2, [N*N2] + self.input_shape)]
            cross_start = N + N2

            @gpflow.decors.name_scope("apply_recurse_kern_X_X2")
            def apply_recurse_kern(var_a_all, concat_outputs=True):
                var_a_1 = var_a_all[:N]
                var_a_2 = var_a_all[N:cross_start]
                var_a_cross = var_a_all[cross_start:]
                vz = [self.recurse_kern.Kdiag(var_a_1),
                      self.recurse_kern.Kdiag(var_a_2),
                      self.recurse_kern.K(var_a_cross, var_a_1, var_a_2)]
                if concat_outputs:
                    return tf.concat(vz, axis=0)
                return vz
        inputs = tf.concat(var_z_list, axis=0)
        if self.data_format == "NHWC":
            # Transpose NCHW -> NHWC
            inputs = tf.transpose(inputs, [0, 2, 3, 1])

        if len(self.block_sizes) > 0:
            # Define almost all the network
            inputs = self.headless_network(inputs, apply_recurse_kern)
            # Last nonlinearity before final dense layer
            var_z_list = apply_recurse_kern(inputs, concat_outputs=False)
        # averaging for the final dense layer
        var_z_cross = tf.reshape(var_z_list[-1], [N, N2, -1])
        var_z_cross_last = tf.reduce_mean(var_z_cross, axis=2)
        result = self.var_bias + self.var_weight * var_z_cross_last
        return result

    @gpflow.decors.name_scope()
    def Kdiag(self, X):
        if X.dtype != self.input_type:
            raise TypeError("Input dtype is wrong: {} is not {}"
                            .format(X.dtype, self.input_type))
        inputs = tf.reshape(tf.square(X), [-1] + self.input_shape)
        if len(self.block_sizes) > 0:
            inputs = self.headless_network(inputs, self.recurse_kern.Kdiag)
            # Last dense layer
            inputs = self.recurse_kern.Kdiag(inputs)

        all_except_first = np.arange(1, len(inputs.shape))
        var_z_last = tf.reduce_mean(inputs, axis=all_except_first)
        result = self.var_bias + self.var_weight * var_z_last
        return result

    @abc.abstractmethod
    def headless_network(self, inputs, apply_recurse_kern):
        """
        Apply the network that this kernel defines, except the last dense layer.
        The last dense layer is different for K and Kdiag.
        """
        raise NotImplementedError
