# Copyright (c) 2025 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import tensorflow as tf

from .base import BaseAugmentationLayer
from ..ops import random_apply


class RandomVideoBrightness(BaseAugmentationLayer):
    """
    Adjusts the brightness of videos by a random factor.

    :param max_delta: This parameter controls the maximum relative
        change in brightness (must be non-negative).
    """

    def __init__(self, max_delta: float, **kwargs):
        super().__init__(**kwargs)
        self.max_delta = max_delta

    def call(self, inputs, training=False):
        def adjust(video):
            fn = lambda x: tf.image.random_brightness(x, max_delta=self.max_delta)
            return tf.map_fn(fn, video)

        if training:
            outputs = adjust(inputs)
            outputs.set_shape(inputs.shape)
            return outputs

        return inputs


class RandomVideoContrast(BaseAugmentationLayer):
    """
    Adjusts the contrast of videos by a random factor.

    :param lower: Lower bound for the random contrast factor.
    :param upper: Upper bound for the random contrast factor.
    """

    def __init__(self, lower: float, upper: float, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def call(self, inputs, training=False):
        def adjust(video):
            fn = lambda x: tf.image.random_contrast(x, lower=self.lower, upper=self.upper)
            return tf.map_fn(fn, video)

        if training:
            outputs = adjust(inputs)
            outputs.set_shape(inputs.shape)
            return outputs

        return inputs


class RandomVideoHue(BaseAugmentationLayer):
    """
    Adjusts the hue of RGB videos by a random factor.

    :param max_delta: The maximum value for the random delta.
    """

    def __init__(self, max_delta: float, **kwargs):
        super().__init__(**kwargs)
        self.max_delta = max_delta

    def call(self, inputs, training=False):
        def adjust(video):
            fn = lambda x: tf.image.random_hue(x, max_delta=self.max_delta)
            return tf.map_fn(fn, video)

        if training:
            outputs = adjust(inputs)
            outputs.set_shape(inputs.shape)
            return outputs

        return inputs


class RandomVideoSaturation(BaseAugmentationLayer):
    """
    Adjusts the saturation of videos by a random factor.

    :param lower: Lower bound for the random saturation factor.
    :param upper: Upper bound for the random saturation factor.
    """

    def __init__(self, lower: float, upper: float,  **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def call(self, inputs, training=False):
        def adjust(video):
            fn = lambda x: tf.image.random_saturation(x, lower=self.lower, upper=self.upper)
            return tf.map_fn(fn, video)

        if training:
            outputs = adjust(inputs)
            outputs.set_shape(inputs.shape)
            return outputs

        return inputs


class RandomGrayscale(BaseAugmentationLayer):
    """ Randomly converts videos to grayscale. """

    def call(self, inputs, training=False):
        def adjust(video):
            fn = lambda x: random_apply(self._to_grayscale, x, p=0.5)
            return tf.map_fn(fn, video)

        if training:
            outputs = adjust(inputs)
            outputs.set_shape(inputs.shape)
            return outputs

        return inputs

    @staticmethod
    def _to_grayscale(video: tf.Tensor) -> tf.Tensor:
        """ Applies grayscale conversion to video. """
        video = tf.image.rgb_to_grayscale(video)
        # video = tf.tile(video, [1, 1, 1, 3])
        video = tf.image.grayscale_to_rgb(video)
        return video

