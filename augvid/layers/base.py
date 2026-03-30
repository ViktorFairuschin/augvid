# Copyright (c) 2025 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import typing

import tensorflow as tf


class BaseAugmentationLayer(tf.keras.layers.Layer):
    """ Base class for video augmentation layer. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = tf.keras.layers.InputSpec(ndim=5, axes={4: 3})

    @staticmethod
    def _apply_to_video(video: tf.Tensor, op: typing.Callable, **kwargs) -> tf.Tensor:
        """ Applies image `op` to video. """
        t, h, w, c = video.shape.as_list()

        video = op(tf.reshape(video, [t * h, w, c]), **kwargs)
        return tf.reshape(video, [t, h, w, c])

