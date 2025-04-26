# Copyright (c) 2025 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse

import cv2
import imageio
import numpy as np

from decord import VideoReader, cpu

from augvid import (
    RandomVideoBrightness,
    RandomVideoContrast,
    RandomVideoHue,
    RandomVideoSaturation,
    RandomHorizontalVideoFlip,
    RandomVerticalVideoFlip
)


HEIGHT, WIDTH, NUM_FRAMES = 480, 640, 10


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to video.')
    return parser


def main(args: argparse.Namespace):
    vr = VideoReader(args.video, height=HEIGHT, width=WIDTH, num_threads=-1, ctx=cpu(0))
    video = vr.get_batch(indices=range(NUM_FRAMES)).asnumpy()

    layers = [
        RandomVideoBrightness(max_delta=0.3),
        RandomVideoContrast(lower=0.5, upper=1.5),
        RandomVideoHue(max_delta=0.2),
        RandomVideoSaturation(lower=0.5, upper=1.5),
        RandomVerticalVideoFlip(),
        RandomHorizontalVideoFlip(),
    ]

    frames = []
    for layer in layers:
        aug_video = layer(np.expand_dims(video, axis=0), training=True)
        aug_video = aug_video.numpy().astype(np.uint8)[0]
        frames.extend(aug_video)

        for frame in aug_video:
            cv2.imshow('video', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    cv2.destroyAllWindows()

    # imageio.mimsave('assets/demo.gif', frames)


if __name__ == "__main__":
    main(create_parser().parse_args())
