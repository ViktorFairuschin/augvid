# Copyright (c) 2025 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import tensorflow as tf

from .base import BaseAugmentationLayer


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

