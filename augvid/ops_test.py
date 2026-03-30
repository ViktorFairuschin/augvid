# Copyright (c) 2025 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import tensorflow as tf


from augvid.ops import random_apply


class TestRandomApply(tf.test.TestCase):

    def setUp(self):
        super().setUp()

        self.x = tf.constant([1.0, 2.0, 3.0])
        self.f = lambda x: x ** 2

    def test_p_is_one(self):
        y = random_apply(self.f, self.x, p=1.0)
        self.assertAllClose(y, self.x ** 2)

    def test_p_is_zero(self):
        y = random_apply(self.f, self.x, p=0.0)
        self.assertAllClose(y, self.x)