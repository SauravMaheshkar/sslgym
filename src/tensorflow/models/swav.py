"""
Implementation of SwAV model.

Based on Unsupervised Learning of Visual Features by Contrasting Cluster Assignments

Authors: Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal,
Piotr Bojanowski and Armand Joulin
"""

from itertools import groupby
from typing import Tuple

import numpy as np
import tensorflow as tf


class SwAV(tf.keras.Model):
    """SwAV model class"""

    def __init__(
        self,
        units: Tuple[int, int] = (1024, 96),
        projection_dim: int = 10,
        num_sinkhorn_iters: int = 3,
        CROPS_FOR_ASSIGN: Tuple[int, int] = (0, 1),
        NUM_CROPS: Tuple[int, int] = (2, 3),
        TEMPERATURE: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.projection_dim = projection_dim
        self.CROPS_FOR_ASSIGN = CROPS_FOR_ASSIGN
        self.NUM_CROPS = NUM_CROPS
        self.TEMPERATURE = TEMPERATURE
        self.num_sinkhorn_iters = num_sinkhorn_iters

        self.encoder = self.build_encoder()
        self.projection = self.build_projection(self.units, self.projection_dim)

        self.loss_tracker = tf.keras.metrics.Mean(name="swav_loss")

    def get_config(self) -> dict:
        return {
            "units": self.units,
            "projection_dim": self.projection_dim,
            "num_sinkhorn_iters": self.num_sinkhorn_iters,
            "CROPS_FOR_ASSIGN": self.CROPS_FOR_ASSIGN,
            "NUM_CROPS": self.NUM_CROPS,
            "TEMPERATURE": self.TEMPERATURE,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None) -> "SwAV":
        return cls(**config)

    @property
    def metrics(self) -> list:
        return [
            self.loss_tracker,
        ]

    def save_weights(
        self,
        filepath: str = "artifacts/swav/",
        overwrite=True,
        save_format="h5",
        options=None,
    ) -> None:
        self.encoder.save_weights("encoder.h5")
        self.projection.save_weights("projection.h5")

    def build_encoder(self) -> tf.keras.Model:
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

    def build_projection(self, units, projection_dim) -> tf.keras.Model:
        inputs = tf.keras.layers.Input((2048,))
        projection_1 = tf.keras.layers.Dense(units[0])(inputs)
        projection_1 = tf.keras.layers.BatchNormalization()(projection_1)
        projection_1 = tf.keras.layers.Activation("relu")(projection_1)

        projection_2 = tf.keras.layers.Dense(units[1])(projection_1)
        projection_2_normalize = tf.math.l2_normalize(
            projection_2, axis=1, name="projection"
        )

        prototype = tf.keras.layers.Dense(
            projection_dim, use_bias=False, name="prototype"
        )(projection_2_normalize)

        return tf.keras.models.Model(
            inputs=inputs, outputs=[projection_2_normalize, prototype]
        )

    def train_step(self, images: tf.Tensor) -> dict:
        """
        References:

        * https://github.com/facebookresearch/swav/blob/master/main_swav.py
        * https://github.com/facebookresearch/swav/issues/19
        * https://github.com/ayulockin/SwAV-TF
        """
        im1, im2, im3, im4, im5 = images
        inputs = [im1, im2, im3, im4, im5]
        batch_size = inputs[0].shape[0]

        # ============ create crop entries with same shape ... ============
        crop_sizes = [inp.shape[1] for inp in inputs]  # list of crop size of views
        unique_consecutive_count = [
            len([elem for elem in g]) for _, g in groupby(crop_sizes)
        ]  # equivalent to torch.unique_consecutive
        idx_crops = tf.cumsum(unique_consecutive_count)

        # ============ multi-res forward passes ... ============
        start_idx = 0
        with tf.GradientTape() as tape:
            for end_idx in idx_crops:
                concat_input = tf.stop_gradient(
                    tf.concat(inputs[start_idx:end_idx], axis=0)
                )
                _embedding = self.encoder(
                    concat_input
                )  # get embedding of same dim views together
                if start_idx == 0:
                    embeddings = _embedding  # for first iter
                else:
                    embeddings = tf.concat(
                        (embeddings, _embedding), axis=0
                    )  # concat all the embeddings from all the views
                start_idx = end_idx

            projection, prototype = self.projection(
                embeddings
            )  # get normalized projection and prototype
            projection = tf.stop_gradient(projection)

            # ============ swav loss ... ============
            loss = 0
            for i, crop_id in enumerate(self.CROPS_FOR_ASSIGN):
                with tape.stop_recording():
                    out = prototype[batch_size * crop_id : batch_size * (crop_id + 1)]

                    # get assignments
                    q = sinkhorn(
                        out, self.num_sinkhorn_iters
                    )  # sinkhorn is used for cluster assignment

                # cluster assignment prediction
                subloss = 0
                for v in np.delete(
                    np.arange(np.sum(self.NUM_CROPS)), crop_id
                ):  # (for rest of the portions compute p and take cross entropy with q)
                    p = tf.nn.softmax(
                        prototype[batch_size * v : batch_size * (v + 1)]
                        / self.TEMPERATURE
                    )
                    subloss -= tf.math.reduce_mean(
                        tf.math.reduce_sum(q * tf.math.log(p), axis=1)
                    )
                loss += subloss / tf.cast(
                    (tf.reduce_sum(self.NUM_CROPS) - 1), tf.float32
                )

            loss /= len(self.CROPS_FOR_ASSIGN)  # type: ignore

        # ============ backprop ... ============
        variables = (
            self.encoder.trainable_variables + self.projection.trainable_variables
        )
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)

        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result()}


def sinkhorn(
    sample_prototype_batch: tf.Tensor, num_sinkhorn_iters: int = 3
) -> tf.Tensor:
    """
    Perform sinkhorn normalization on the sample prototype batch
    """
    Q = tf.transpose(tf.exp(sample_prototype_batch / 0.05))
    Q /= tf.keras.backend.sum(Q)
    K, B = tf.shape(Q)

    u = tf.zeros_like(K, dtype=tf.float32)
    r = tf.ones_like(K, dtype=tf.float32) / K
    c = tf.ones_like(B, dtype=tf.float32) / B

    for _ in range(num_sinkhorn_iters):
        u = tf.keras.backend.sum(Q, axis=1)
        Q *= tf.expand_dims((r / u), axis=1)
        Q *= tf.expand_dims(c / tf.keras.backend.sum(Q, axis=0), 0)

    final_quantity = Q / tf.keras.backend.sum(Q, axis=0, keepdims=True)
    final_quantity = tf.transpose(final_quantity)

    return final_quantity
