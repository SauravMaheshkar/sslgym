"""Common Augmentations"""

import random
from typing import Any, Callable, Tuple

import tensorflow as tf


def shuffle_zipped_output(*args) -> Tuple[Any, ...]:
    """Shuffle the given inputs"""
    listify = [*args]
    random.shuffle(listify)
    return tuple(listify)


@tf.function
def scale_image(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Convert all images to float32"""
    image = tf.image.convert_image_dtype(image, tf.float32)
    return (image, label)


@tf.function
def solarize(image: tf.Tensor, threshold: int = 128) -> tf.Tensor:
    """Solarize the input image"""
    return tf.where(image < threshold, image, 255 - image, name="solarize")


@tf.function
def color_drop(image: tf.Tensor) -> tf.Tensor:
    """Randomly convert the input image to GrayScale"""
    image = tf.image.rgb_to_grayscale(image)
    image = tf.tile(image, [1, 1, 3])
    return image


@tf.function
def gaussian_blur(
    image: tf.Tensor, kernel_size: int = 23, padding: str = "SAME"
) -> tf.Tensor:
    """
    Randomly apply Gaussian Blur to the input image

    Reference: https://github.com/google-research/simclr/blob/master/data_util.py
    """

    sigma = tf.random.uniform((1,)) * 1.9 + 0.1
    radius = tf.cast(kernel_size / 2, tf.int32)
    kernel_size = radius * 2 + 1
    x = tf.cast(tf.range(-radius, radius + 1), tf.float32)
    blur_filter = tf.exp(
        -tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, tf.float32), 2.0))
    )
    blur_filter /= tf.reduce_sum(blur_filter)

    # One vertical and one horizontal filter.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
        image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(
        image, blur_h, strides=[1, 1, 1, 1], padding=padding
    )
    blurred = tf.nn.depthwise_conv2d(
        blurred, blur_v, strides=[1, 1, 1, 1], padding=padding
    )
    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)
    return blurred


@tf.function
def color_jitter(image: tf.Tensor, s: float = 0.5) -> tf.Tensor:
    """Randomly apply Color Jittering to the input image"""
    x = tf.image.random_brightness(image, max_delta=0.8 * s)
    x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_hue(x, max_delta=0.2 * s)
    x = tf.clip_by_value(x, 0, 1)
    return x


@tf.function
def random_resize_crop(
    image: tf.Tensor, min_scale: float, max_scale: float, crop_size: int
) -> tf.Tensor:
    """Randomly resize and crop the input image"""
    if crop_size == 224:
        image_shape = 260
        image = tf.image.resize(image, (image_shape, image_shape))
    else:
        image_shape = 160
        image = tf.image.resize(image, (image_shape, image_shape))

    # Get the crop size for given min and max scale
    size = tf.random.uniform(
        shape=(1,),
        minval=min_scale * image_shape,
        maxval=max_scale * image_shape,
        dtype=tf.float32,
    )
    size = tf.cast(size, tf.int32)[0]

    # Get the crop from the image
    crop = tf.image.random_crop(image, (size, size, 3))
    crop_resize = tf.image.resize(crop, (crop_size, crop_size))

    return crop_resize


@tf.function
def random_apply(func: Callable, x: tf.Tensor, prob: float) -> tf.Tensor:
    """Randomly apply the desired func to the input image"""
    return tf.cond(
        tf.less(
            tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
            tf.cast(prob, tf.float32),
        ),
        lambda: func(x),
        lambda: x,
    )
