import numpy as np
import tensorflow as tf

def geometric_augmentation(x, img_height, img_width, img_channels, data_type):
    x_augmented = np.zeros((len(x) * 6, img_height, img_width, img_channels), dtype = data_type)

    for n in range(len(x_augmented)):
        # Store the original image
        x_augmented[n * 6] = x[n]

        # Store the original image rotated 90 degrees
        x_augmented[n * 6 + 1] = tf.image.rot90(x_augmented[n * 6])

        # Store the original image rotated 180 degrees
        x_augmented[n * 6 + 2] = tf.image.rot90(x_augmented[n * 6 + 1])

        # Store the original image rotated 270 degrees
        x_augmented[n * 6 + 3] = tf.image.rot90(x_augmented[n * 6 + 2])

        # Store the original image flipped around y-axis
        x_augmented[n * 6 + 4] = tf.image.flip_left_right(x_augmented[n * 6])

        # Store the original image flipped around x-axis
        x_augmented[n * 6 + 5] = tf.image.flip_up_down(x_augmented[n * 6])

    return x_augmented


def x_axis_reflection(x, img_height, img_width, img_channels, data_type):
    x_augmented = np.zeros((len(x) * 2, img_height, img_width, img_channels), dtype = data_type)

    for n in range(len(x_augmented)):
        # Store the original image
        x_augmented[n * 2] = x[n]

        # Store the original image flipped around x-axis
        x_augmented[n * 2 + 1] = tf.image.flip_up_down(x_augmented[n * 2])

    return x_augmented