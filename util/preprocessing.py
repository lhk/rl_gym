import lycon
import numpy as np


def preprocess_frame(frame, FRAME_SIZE):
    downsampled = lycon.resize(frame, width=FRAME_SIZE[0], height=FRAME_SIZE[1],
                               interpolation=lycon.Interpolation.NEAREST)
    grayscale = downsampled.mean(axis=-1).astype(np.uint8)
    return grayscale
