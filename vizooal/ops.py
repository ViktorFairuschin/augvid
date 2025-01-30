# Copyright (c) 2025 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import typing
import tensorflow as tf


def random_apply(func: typing.Callable, x: tf.Tensor, p: float) -> typing.Any:
    """ Apply func to x with probability p. """
    random_var = tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32)
    pred = tf.less(random_var, tf.cast(p, tf.float32))
    return tf.cond(pred, lambda: func(x), lambda: x)


def random_brightness(image: tf.Tensor, max_delta: float) -> tf.Tensor:
    """ Adjust the brightness of the images by a random factor. """
    image = tf.image.random_brightness(image, max_delta)
    return tf.clip_by_value(image, 0, 255)


def random_contrast(image: tf.Tensor, lower: float, upper: float) -> tf.Tensor:
    """ Adjust the contrast of the images by a random factor. """
    image = tf.image.random_contrast(image, lower, upper)
    return tf.clip_by_value(image, 0, 255)


def random_hue(image: tf.Tensor, max_delta: float) -> tf.Tensor:
    """ Adjust the hue of the images by a random factor. """
    image = tf.image.random_hue(image, max_delta)
    return tf.clip_by_value(image, 0, 255)


def random_saturation(image: tf.Tensor, lower: float, upper: float) -> tf.Tensor:
    """ Adjust the hue of the images by a random factor. """
    image = tf.image.random_saturation(image, lower, upper)
    return tf.clip_by_value(image, 0, 255)


def random_flip_left_right(image: tf.Tensor) -> tf.Tensor:
    """ Randomly flip the image horizontally. """
    return tf.image.random_flip_left_right(image)


def random_flip_up_down(image: tf.Tensor) -> tf.Tensor:
    """ Randomly flip the image vertically. """
    return tf.image.random_flip_up_down(image)


def add_gaussian_noise(image: tf.Tensor, stddev: float) -> tf.Tensor:
    """ Add Gaussian noise to image. """
    with tf.name_scope('add_gaussian_noise'):
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=stddev, dtype=tf.float32)
        return tf.clip_by_value(tf.add(image, noise), 0, 255)


def adjust_video(video: tf.Tensor, func: typing.Callable, **kwargs):
    """ Apply func to each frame of the video. """
    t, h, w, c = video.shape.as_list()

    video = func(tf.reshape(video, [t * h, w, c]), **kwargs)
    return tf.reshape(video, [t, h, w, c])

