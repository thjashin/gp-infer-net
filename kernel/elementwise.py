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

import tensorflow as tf
import numpy as np


class ElementwiseKernel(object):
    def K(self, cov, var1, var2=None):
        raise NotImplementedError

    def Kdiag(self, var):
        raise NotImplementedError

    def nlin(self, x):
        """
        The nonlinearity that this is computing the expected inner product of.
        Used for testing.
        """
        raise NotImplementedError


class ReLUKernel(ElementwiseKernel):
    def __init__(self, name=None):
        super(ReLUKernel, self).__init__()

    def K(self, cov, var1, var2=None):
        if var2 is None:
            sqrt1 = sqrt2 = tf.sqrt(var1)
        else:
            sqrt1, sqrt2 = tf.sqrt(var1), tf.sqrt(var2)

        norms_prod = sqrt1[:, None, ...] * sqrt2
        norms_prod = tf.reshape(norms_prod, tf.shape(cov))

        cos_theta = tf.clip_by_value(cov / norms_prod, -1., 1.)
        theta = tf.acos(cos_theta)  # angle wrt the previous RKHS

        sin_theta = tf.sqrt(1. - cos_theta**2)
        J = sin_theta + (np.pi - theta) * cos_theta
        div = 2*np.pi
        return norms_prod / div * J

    def Kdiag(self, var):
        return var/2

    def nlin(self, x):
        return tf.nn.relu(x)
