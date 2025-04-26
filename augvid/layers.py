# Copyright (c) 2025 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import typing
import keras
import tensorflow as tf

from .common import random_apply


class BaseAugmentationLayer(keras.layers.Layer):
    """
    Base class for video augmentation layer.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = keras.layers.InputSpec(ndim=5, axes={4: 3})

    @staticmethod
    def _apply_to_video(video: tf.Tensor, func: typing.Callable, **kwargs) -> tf.Tensor:
        """ Applies image op `func` to video. """
        t, h, w, c = video.shape.as_list()

        video = func(tf.reshape(video, [t * h, w, c]), **kwargs)
        return tf.reshape(video, [t, h, w, c])


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


class RandomGrayscale(BaseAugmentationLayer):
    """
    Randomly converts videos to grayscale.
    """

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


class RandomBlur(BaseAugmentationLayer):
    """
    Randomly applies Gaussian blur videos to videos.

    :param max_factor: Controls the extent to which the video is blurred.
    :param kernel_size: The size of the blur kernel.
    """

    def __init__(self, max_factor: float, filter_size: int, **kwargs):
        super().__init__(**kwargs)
        self.max_factor = max_factor
        self.filter_size = filter_size

    def call(self, inputs, training=False):
        def adjust(video):
            fn = lambda x: random_apply(self._apply_filter, x, p=0.5)
            return tf.map_fn(fn, video)

        if training:
            outputs = adjust(inputs)
            outputs.set_shape(inputs.shape)
            return outputs

        return inputs

    def _apply_filter(self, video: tf.Tensor) -> tf.Tensor:
        """ Applies convolution filter to video. """
        dtype = video.dtype
        video = tf.cast(video, dtype=tf.float32)

        factor = tf.random.uniform(shape=(), minval=0, maxval=self.max_factor)
        blur_h = self._get_filter(factor=factor, filter_size=self.filter_size)
        blur_v = self._get_filter(factor=factor, filter_size=self.filter_size)

        blurred = tf.nn.depthwise_conv2d(video, blur_h, strides=[1, 1, 1, 1], padding='SAME')
        blurred = tf.nn.depthwise_conv2d(blurred, blur_v, strides=[1, 1, 1, 1], padding='SAME')
        return tf.cast(blurred, dtype=dtype)

    @staticmethod
    def _get_filter(factor, filter_size):
        """ Creates convolution filter. """
        x = tf.cast(tf.range(- filter_size // 2 + 1, filter_size // 2 + 1), dtype=tf.float32)
        blur_filter = tf.exp(-tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(factor, dtype=tf.float32), 2.0)))
        blur_filter /= tf.reduce_sum(blur_filter)
        blur_filter = tf.reshape(blur_filter, [1, filter_size, 1, 1])
        blur_filter = tf.cast(tf.tile(blur_filter, [1, 1, 3, 1]), dtype=tf.float32)
        return blur_filter

