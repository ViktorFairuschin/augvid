# Copyright (c) 2025 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import keras

from vizooal.ops import *


class BaseAugmentationLayer(keras.layers.Layer):
    """
    Base class for augmentation layers.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = keras.layers.InputSpec(ndim=5, axes={4: 3})  # removed dtype=tf.uint8


class RandomVideoBrightness(BaseAugmentationLayer):
    """
    Apply random brightness adjustment to video.

    Expects inputs to be in [0, 255] range.

    :param max_delta: Adjustment strength, must be non-negative.
    :param p: Probability of adjustment, must be in [0, 1] range.
    """

    def __init__(self, max_delta: float = 0.1, p: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.max_delta = max_delta
        self.p = p

    def call(self, inputs, training=False, **kwargs):
        def adjust(video):
            fn = lambda x: adjust_video(x, random_brightness, max_delta=self.max_delta)
            return tf.map_fn(fn, video)

        if training:
            outputs = random_apply(adjust, inputs, p=self.p)
            outputs.set_shape(inputs.shape)
            return outputs

        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_delta": self.max_delta,
            "p": self.p,
        })
        return config


class RandomVideoContrast(BaseAugmentationLayer):
    """
    Apply random contrast adjustment to video.

    Expects inputs to be in [0, 255] range.

    :param lower: Lower bound for adjustment strength, must be non-negative.
    :param upper: Upper bound for adjustment strength, must be non-negative.
    :param p: Probability of adjustment, must be in [0, 1] range.
    """

    def __init__(self, lower: float = 0.2, upper: float = 2.0, p: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.p = p

    def call(self, inputs, training=False, **kwargs):
        def adjust(video):
            fn = lambda x: adjust_video(x, random_contrast, lower=self.lower, upper=self.upper)
            return tf.map_fn(fn, video)

        if training:
            outputs = random_apply(adjust, inputs, p=self.p)
            outputs.set_shape(inputs.shape)
            return outputs

        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "lower": self.lower,
            "upper": self.upper,
            "p": self.p,
        })
        return config


class RandomVideoHue(BaseAugmentationLayer):
    """
    Apply random hue adjustment to video.

    Expects inputs to be in [0, 255] range.

    :param max_delta: Adjustment strength, must be non-negative.
    :param p: Probability of adjustment, must be in [0, 1] range.
    """

    def __init__(self, max_delta: float = 0.3, p: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.max_delta = max_delta
        self.p = p

    def call(self, inputs, training=False, **kwargs):
        def adjust(video):
            fn = lambda x: adjust_video(x, random_hue, max_delta=self.max_delta)
            return tf.map_fn(fn, video)

        if training:
            outputs = random_apply(adjust, inputs, p=self.p)
            outputs.set_shape(inputs.shape)
            return outputs

        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_delta": self.max_delta,
            "p": self.p,
        })
        return config


class RandomVideoSaturation(BaseAugmentationLayer):
    """
    Apply random saturation adjustment to video.

    Expects inputs to be in [0, 255] range.

    :param lower: Lower bound for adjustment strength, must be non-negative.
    :param upper: Upper bound for adjustment strength, must be non-negative.
    :param p: Probability of adjustment, must be in [0, 1] range.
    """

    def __init__(self, lower: float = 0.0, upper: float = 3.0, p: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.p = p

    def call(self, inputs, training=False, **kwargs):
        def adjust(video):
            fn = lambda x: adjust_video(x, random_saturation, lower=self.lower, upper=self.upper)
            return tf.map_fn(fn, video)

        if training:
            outputs = random_apply(adjust, inputs, p=self.p)
            outputs.set_shape(inputs.shape)
            return outputs

        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "lower": self.lower,
            "upper": self.upper,
            "p": self.p,
        })
        return config


class RandomHorizontalVideoFlip(BaseAugmentationLayer):
    """
    Apply random horizontal flip to video.

    :param p: Probability of adjustment, must be in [0, 1] range.
    """

    def __init__(self, p: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def call(self, inputs, training=False, **kwargs):
        def adjust(video):
            fn = lambda x: adjust_video(x, random_flip_left_right)
            return tf.map_fn(fn, video)

        if training:
            outputs = random_apply(adjust, inputs, p=self.p)
            outputs.set_shape(inputs.shape)
            return outputs

        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "p": self.p,
        })
        return config


class RandomVerticalVideoFlip(BaseAugmentationLayer):
    """
    Apply random vertical flip to video.

    :param p: Probability of adjustment, must be in [0, 1] range.
    """

    def __init__(self, p: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def call(self, inputs, training=False, **kwargs):
        def adjust(video):
            fn = lambda x: adjust_video(x, random_flip_up_down)
            return tf.map_fn(fn, video)

        if training:
            outputs = random_apply(adjust, inputs, p=self.p)
            outputs.set_shape(inputs.shape)
            return outputs

        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "p": self.p,
        })
        return config
