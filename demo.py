# Copyright (c) 2025 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse

import cv2
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


HEIGHT, WIDTH, BATCH_SIZE = 480, 640, 30


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to video.')
    return parser


def main(args: argparse.Namespace):
    layers = [
        RandomVideoBrightness(max_delta=0.3),
        RandomVideoContrast(lower=0.5, upper=1.5),
        RandomVideoHue(max_delta=0.2),
        RandomVideoSaturation(lower=0.5, upper=1.5),
        RandomVerticalVideoFlip(),
        RandomHorizontalVideoFlip(),
    ]

    reader = VideoReader(args.video, height=HEIGHT, width=WIDTH, num_threads=-1, ctx=cpu(0))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('assets/demo.mp4', fourcc, 30, (WIDTH, HEIGHT))

    for i, layer in enumerate(layers):
        video = reader.get_batch(indices=range(i * BATCH_SIZE, (i + 1) * BATCH_SIZE)).asnumpy()
        video = np.expand_dims(video, axis=0)
        video = layer(video, training=True)
        video = video.numpy().astype(np.uint8)[0]

        for frame in video:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
            cv2.imshow('demo', frame)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    writer.release()


if __name__ == '__main__':
    main(create_parser().parse_args())

