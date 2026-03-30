# Copyright (c) 2025 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import tensorflow as tf

from .base import BaseAugmentationLayer


class RandomHorizontalVideoFlip(BaseAugmentationLayer):
    """
    Randomly flips videos horizontally.
    """

    def call(self, inputs, training=False):
        def adjust(video):
            fn = lambda x: self._apply_to_video(x, tf.image.random_flip_left_right)
            return tf.map_fn(fn, video)

        if training:
            outputs = adjust(inputs)
            outputs.set_shape(inputs.shape)
            return outputs

        return inputs


class RandomVerticalVideoFlip(BaseAugmentationLayer):
    """
    Randomly flips videos vertically.
    """

    def call(self, inputs, training=False):
        def adjust(video):
            fn = lambda x: self._apply_to_video(x, tf.image.random_flip_up_down)
            return tf.map_fn(fn, video)

        if training:
            outputs = adjust(inputs)
            outputs.set_shape(inputs.shape)
            return outputs

        return inputs

