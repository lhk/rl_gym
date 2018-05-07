import tensorflow as tf


def huber_loss(y_true, y_pred):
    # huber loss
    delta = 1
    error = tf.abs(y_true - y_pred)
    mask = tf.abs(error) < delta
    mask = tf.cast(mask, tf.float32)

    square = 1 / 2 * error ** 2 * mask
    linear = 1 / 2 * error * (1 - mask)

    huber = square + linear
    return huber
