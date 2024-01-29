"""Test VICReg model."""

import pytest

import tensorflow as tf
from src.tensorflow.models.vicreg import VICReg

BATCH_SIZE = 2


@pytest.mark.tensorflow
@pytest.mark.tensorflow_model
def test_vicreg_encoder() -> None:
    """Test VICReg Encoder model."""
    model = VICReg(num_units=512, invar_coeff=25.0, var_coeff=25.0, cov_coeff=1.0)
    image = tf.random.uniform((BATCH_SIZE, 224, 224, 3))
    encoder_output = model.encoder(image)

    assert isinstance(encoder_output, tf.Tensor)
    assert encoder_output.shape == (BATCH_SIZE, 2048)


@pytest.mark.tensorflow
@pytest.mark.tensorflow_model
def test_vicreg_expander() -> None:
    """Test VICReg Expander model."""
    model = VICReg(num_units=512, invar_coeff=25.0, var_coeff=25.0, cov_coeff=1.0)
    image = tf.random.uniform((BATCH_SIZE, 2048))
    expander_output = model.expander(image)

    assert isinstance(expander_output, tf.Tensor)
    assert expander_output.shape == (BATCH_SIZE, 512)


@pytest.mark.tensorflow
@pytest.mark.tensorflow_model
def test_vicreg() -> None:
    """Test VICReg model."""
    tf.keras.backend.clear_session()
    model = VICReg(num_units=128, invar_coeff=25.0, var_coeff=25.0, cov_coeff=1.0)
    view_1 = tf.random.uniform((BATCH_SIZE, 224, 224, 3))
    view_2 = tf.random.uniform((BATCH_SIZE, 224, 224, 3))

    loss = model([view_1, view_2])
    loss_tensor = tf.constant(loss[0])
    assert loss_tensor.shape == (BATCH_SIZE, 128)
