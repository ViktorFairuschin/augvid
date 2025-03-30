# Copyright (c) 2025 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import typing
import tensorflow as tf


def adjust_video(video: tf.Tensor, func: typing.Callable, **kwargs) -> tf.Tensor:
    """ Applies `func` to video. """
    t, h, w, c = video.shape.as_list()

    video = func(tf.reshape(video, [t * h, w, c]), **kwargs)
    return tf.reshape(video, [t, h, w, c])

