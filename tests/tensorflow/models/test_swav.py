"""Test SwAV model."""
import pytest
import tensorflow as tf

from src.tensorflow.models.swav import SwAV


BATCH_SIZE = 2


@pytest.mark.tensorflow
@pytest.mark.tensorflow_model
def test_swav_encoder() -> None:
    """Test SwAV Encoder model."""
    model = SwAV()
    image = tf.random.uniform((BATCH_SIZE, 224, 224, 3))
    encoder_output = model.encoder(image)

    assert isinstance(encoder_output, tf.Tensor)
    assert encoder_output.shape == (BATCH_SIZE, 2048)


@pytest.mark.tensorflow
@pytest.mark.tensorflow_model
def test_swav_projection() -> None:
    """Test SwAV Projection model."""
    model = SwAV()
    image = tf.random.uniform((BATCH_SIZE, 2048))
    expander_output = model.projection(image)

    assert isinstance(expander_output[0], tf.Tensor)
    assert expander_output[0].shape == (BATCH_SIZE, 96)
