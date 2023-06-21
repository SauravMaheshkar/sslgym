"""
Implementation of VICReg model.

Based on Variance-Invariance-Covariance Regularization For Self-Supervised Learning

Authors: Adrien Bardes, Jean Ponce and Yann LeCun
"""

from typing import Tuple

import tensorflow as tf


class VICReg(tf.keras.Model):
    """VICReg model class"""

    def __init__(
        self,
        num_units: int,
        invar_coeff: float,
        var_coeff: float,
        cov_coeff: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_units = num_units
        self.invar_coeff = invar_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff

        self.encoder = self.build_encoder()
        self.expander = self.build_expander(self.num_units)

        self.loss_tracker = tf.keras.metrics.Mean(name="vicreg_loss")
        self.invarloss_tracker = tf.keras.metrics.Mean(name="invariance_loss")
        self.varloss_tracker = tf.keras.metrics.Mean(name="variance_loss")
        self.covloss_tracker = tf.keras.metrics.Mean(name="covariance_loss")

    def get_config(self) -> dict:
        return {
            "invar_coeff": self.invar_coeff,
            "var_coeff": self.var_coeff,
            "cov_coeff": self.cov_coeff,
            "num_units": self.num_units,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None) -> "VICReg":
        return cls(**config)

    @property
    def metrics(self) -> list:
        return [
            self.loss_tracker,
            self.invarloss_tracker,
            self.varloss_tracker,
            self.covloss_tracker,
        ]

    def build_encoder(self) -> tf.keras.Model:
        """Build the encoder"""
        encoder_input = tf.keras.layers.Input((None, None, 3))
        base_model = tf.keras.applications.ResNet50(
            include_top=False, weights=None, input_shape=(None, None, 3)
        )
        base_model.trainable = True
        representations = base_model(encoder_input, training=True)
        encoder_output = tf.keras.layers.GlobalAveragePooling2D()(representations)
        encoder = tf.keras.models.Model(
            inputs=encoder_input, outputs=encoder_output, name="encoder"
        )
        return encoder

    def build_expander(self, num_units: int) -> tf.keras.Model:
        """Build the expander"""
        expander_input = tf.keras.layers.Input((2048,))

        projection_1 = tf.keras.layers.Dense(num_units)(expander_input)
        projection_1 = tf.keras.layers.BatchNormalization()(projection_1)
        projection_1 = tf.keras.layers.Activation("relu")(projection_1)

        projection_2 = tf.keras.layers.Dense(num_units)(projection_1)
        projection_2 = tf.keras.layers.BatchNormalization()(projection_2)
        projection_2 = tf.keras.layers.Activation("relu")(projection_2)

        expander_output = tf.keras.layers.Dense(num_units)(projection_2)

        expander = tf.keras.models.Model(
            inputs=expander_input, outputs=expander_output, name="expander"
        )

        return expander

    def save_weights(
        self,
        filepath: str = "artifacts/vicreg/",
        overwrite=True,
        save_format="h5",
        options=None,
    ) -> None:
        self.encoder.save_weights(filepath + "encoder.h5")
        self.expander.save_weights(filepath + "expander.h5")

    def call(self, inputs, training=None, mask=None) -> Tuple[tf.Tensor, tf.Tensor]:
        x, x_prime = inputs[0], inputs[1]
        # Get Representations (through encoder)
        y = self.encoder(x)
        y_prime = self.encoder(x_prime)

        # Get Embeddings (through expander)
        z = self.expander(y)
        z_prime = self.expander(y_prime)

        return z, z_prime

    def train_step(self, data) -> dict:
        x, x_prime = data[0][0], data[1][0]
        inputs = [x, x_prime]
        batch_size = inputs[0][0].shape[0]

        with tf.GradientTape() as tape:
            z, z_prime = self.call(inputs, training=True)

            # Calculate the Representation (Invariance) Loss
            invar_loss = tf.keras.metrics.mean_squared_error(z, z_prime)

            # Calculate var. and std. dev. of embeddings
            z = z - tf.reduce_mean(z, axis=0)
            z_prime = z_prime - tf.reduce_mean(z_prime, axis=0)
            std_z = tf.sqrt(tf.math.reduce_variance(z, axis=0) + 0.0001)
            std_z_prime = tf.sqrt(tf.math.reduce_variance(z_prime, axis=0) + 0.0001)

            # Calculate the Variance Loss (Hinge Function)
            var_loss = (
                tf.reduce_mean(tf.nn.relu(1 - std_z)) / 2
                + tf.reduce_mean(tf.nn.relu(1 - std_z_prime)) / 2
            )

            # Get Covariance Matrix
            cov_z = (z.T @ z) / (batch_size - 1)
            cov_z_prime = (z_prime.T @ z_prime) / (batch_size - 1)

            # Calculate the Covariance Loss
            cov_loss_z = tf.divide(tf.reduce_sum(tf.pow(off_diagonal(cov_z), 2)), 8192)
            cov_loss_z_prime = tf.divide(
                tf.reduce_sum(tf.pow(off_diagonal(cov_z_prime), 2)), 8192
            )
            cov_loss = cov_loss_z + cov_loss_z_prime

            # Weighted Avg. of Invariance, Variance and Covariance Loss
            loss = (
                self.invar_coeff * invar_loss
                + self.var_coeff * var_loss
                + self.cov_coeff * cov_loss
            )

        # Compute gradients
        variables = self.encoder.trainable_variables + self.expander.trainable_variables
        gradients = tape.gradient(loss, variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, variables))
        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.invarloss_tracker.update_state(invar_loss)
        self.varloss_tracker.update_state(var_loss)
        self.covloss_tracker.update_state(cov_loss)
        # Return a dict mapping metric names to current value
        return {
            "loss": self.loss_tracker.result(),
            "invariance_loss": self.invarloss_tracker.result(),
            "variance_loss": self.varloss_tracker.result(),
            "covariance_loss": self.covloss_tracker.result(),
        }


def off_diagonal(tensor: tf.Tensor) -> tf.Tensor:
    """
    Returns the off-diagonal elements of a square tensor.

    :param tensor: A square tensor
    :return: A vector containing the off-diagonal elements of the input tensor
    """
    n, m = tensor.shape[0], tensor.shape[1]
    assert n == m, f"Not a square tensor, dimensions found: {n} and {m}"

    flattened_tensor = tf.reshape(tensor, [-1])[:-1]
    elements = tf.reshape(flattened_tensor, [n - 1, n + 1])[:, 1:]
    return tf.reshape(elements, [-1])
