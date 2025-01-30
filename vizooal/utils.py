# Copyright (c) 2025 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import typing

import cv2
from decord import VideoReader, cpu


def load_video(
        filepath: typing.Union[str, os.PathLike],
        height: int = -1,
        width: int = -1,
        num_frames: int = -1
):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(filepath)

    vr = VideoReader(filepath, height=height, width=width, num_threads=-1, ctx=cpu(0))

    if num_frames == -1:
        return vr[:].asnumpy()

    return vr.get_batch(list(range(num_frames))).asnumpy()


def show_video(video):
    cv2.destroyAllWindows()

    for frame in video:
        cv2.imshow('video', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    cv2.destroyAllWindows()


