# Copyright (c) 2025 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import typing
import tensorflow as tf


def random_apply(func: typing.Callable, x: tf.Tensor, p: float):
    """ Applies `func` to `x` with probability `p`. """
    random_var = tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32)
    pred = tf.less(random_var, tf.cast(p, tf.float32))
    return tf.cond(pred, lambda: func(x), lambda: x)

