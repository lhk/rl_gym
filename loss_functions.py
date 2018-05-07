import tensorflow as tf


def huber_loss(y_true, y_pred):
    # huber loss
    error = tf.abs(y_true - y_pred)
    mask = tf.abs(error) < 1

    square = 1 / 2 * error**2
    linear = 1 / 2 * error

    huber = tf.select(mask, square, linear)
    return huber
