import numpy as np
import tensorflow as tf


def huber_loss(y_true, y_pred):
    # huber loss
    delta = 1
    diff = y_true - y_pred
    mask = tf.abs(diff) < delta
    mask = tf.cast(mask, tf.float32)

    huber = 1 / 2 * diff ** 2 * mask + delta * (tf.abs(diff) - 1 / 2 * delta) * (1 - mask)
    return huber