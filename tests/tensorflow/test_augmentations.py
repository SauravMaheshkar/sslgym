"""Test custom tensorflow augmentations."""

import pytest

import tensorflow as tf
from src.tensorflow.augmentations import (
    color_drop,
    color_jitter,
    gaussian_blur,
    random_apply,
    scale_image,
    shuffle_zipped_output,
    solarize,
)


@pytest.mark.tensorflow
def test_scale_image() -> None:
    """Test scale_image."""
    image = tf.random.uniform((32, 32, 3))
    label = tf.random.uniform((1,))
    output = scale_image(image, label)
    assert output[0].dtype == tf.float32
    assert output[1].dtype == tf.float32
    assert output[0].shape == image.shape
    assert output[1].shape == label.shape


@pytest.mark.tensorflow
@pytest.mark.parametrize("min_val, max_val", [pytest.param(128, 255)])
def test_solarize(min_val: int, max_val: int) -> None:
    """Test solarize."""
    image = tf.random.uniform((32, 32, 3), minval=min_val, maxval=max_val)
    # convert to float32
    image = scale_image(image, None)[0]
    output = solarize(image)
    assert output.dtype == tf.float32
    assert output.shape == image.shape
    # check if the image is solarized
    assert tf.reduce_all(tf.equal(output, max_val - image))


@pytest.mark.tensorflow
def test_color_drop() -> None:
    """Test color_drop."""
    image = tf.random.uniform((32, 32, 3))
    # convert to float32
    image = scale_image(image, None)[0]
    output = color_drop(image)
    assert output.dtype == tf.float32
    assert output.shape == image.shape
    # check if the image is grayscale
    assert tf.reduce_all(tf.equal(output[:, :, 0], output[:, :, 1]))
    assert tf.reduce_all(tf.equal(output[:, :, 1], output[:, :, 2]))


@pytest.mark.tensorflow
def test_gaussian_blur() -> None:
    """Test gaussian_blur."""
    image = tf.random.uniform((32, 32, 3))
    # convert to float32
    image = scale_image(image, None)[0]
    output = gaussian_blur(image)
    assert output.dtype == tf.float32
    assert output.shape == image.shape


@pytest.mark.tensorflow
def test_color_jitter() -> None:
    """Test color_jitter."""
    image = tf.random.uniform((32, 32, 3))
    # convert to float32
    image = scale_image(image, None)[0]
    output = color_jitter(image)
    assert output.dtype == tf.float32
    assert output.shape == image.shape
    # check if the image is jittered
    assert not tf.reduce_all(tf.equal(output, image))


@pytest.mark.tensorflow
def test_random_apply() -> None:
    """Test random_apply."""
    image = tf.random.uniform((32, 32, 3))
    # convert to float32
    image = scale_image(image, None)[0]
    output = random_apply(color_drop, image, prob=1.0)
    assert output.dtype == tf.float32
    assert output.shape == image.shape
    # check if the image is jittered
    assert not tf.reduce_all(tf.equal(output, image))


def test_shuffle_zipped_output() -> None:
    """Test shuffle_zipped_output."""
    a, b, c, d, e = (
        tf.random.uniform((1,)),
        tf.random.uniform((1,)),
        tf.random.uniform((1,)),
        tf.random.uniform((1,)),
        tf.random.uniform((1,)),
    )

    output = shuffle_zipped_output(a, b, c, d, e)
    assert len(output) == 5
