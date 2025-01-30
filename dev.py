# Copyright (c) 2025 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import keras
import numpy as np

from vizooal.utils import load_video, show_video
from vizooal.layers import *


model = keras.Sequential([
    RandomVideoBrightness(),
    RandomVideoContrast(),
    RandomVideoSaturation(),
    RandomVideoHue(),
    RandomHorizontalVideoFlip(),
    RandomVerticalVideoFlip(),
])


video = load_video('assets/countdown.mp4', width=224, height=224, num_frames=100)
videos = np.stack([video] * 3, axis=0)

videos = model(videos, training=True)
for video in videos:
    show_video(video.numpy())

