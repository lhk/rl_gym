import lycon
import numpy as np

import algorithms.dqn.params as params


def preprocess_frame(frame):
    downsampled = lycon.resize(frame, width=params.FRAME_SIZE[0], height=params.FRAME_SIZE[1],
                               interpolation=lycon.Interpolation.NEAREST)
    grayscale = downsampled.mean(axis=-1).astype(np.uint8)
    return grayscale
