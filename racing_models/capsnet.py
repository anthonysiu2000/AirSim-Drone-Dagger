import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class CapsuleLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_caps,
        num_caps_prev,
        depth,
        depth_prev,
        routing_iterations=3,
        use_bias=True,
        squash_last=True,
    ):
        super(CapsuleLayer, self).__init__()
        self.squash_last = squash_last
        self.num_caps = num_caps
        self.num_caps_prev = num_caps_prev
        self.depth = depth
        self.depth_prev = depth_prev
        self.routing_iterations = routing_iterations
        self.use_bias = use_bias
        x = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
        y = tf.zeros_initializer()
        if num_caps_prev is not None:
            self.weight_matrix = tf.Variable(
                x([1, num_caps_prev, num_caps, depth_prev, depth]), trainable=True
            )
            self.bias_matrix = tf.Variable(
                y([1, num_caps_prev, num_caps, depth]), trainable=True
            )
        else:
            self.weight_matrix = tf.Variable(
                x([1, 1, num_caps, depth_prev, depth]), trainable=True
            )
            self.bias_matrix = tf.Variable(y([1, 1, num_caps, depth]), trainable=True)

        # To access, W_ij = self.weight_matrices[0][i][j]

    def call(
        self, inp, **kwargs
    ):  # inp is of shape (batch_size, num_caps_prev, depth_prev)
        batch_size, num_caps_prev = tf.shape(inp)[0], tf.shape(inp)[1]
        bij = tf.zeros([batch_size, num_caps_prev, self.num_caps])
        inp = tf.reshape(inp, [batch_size, num_caps_prev, 1, 1, self.depth_prev])

        # Here, we multiply W_ij * u_i
        weighted_input = tf.matmul(
            inp, self.weight_matrix
        )  # (batch_size, num_caps_prev, num_caps, 1, depth)
        weighted_input = tf.squeeze(
            weighted_input, axis=-2
        )  # (batch_size, num_caps_prev, num_caps, depth)
        if self.use_bias:
            weighted_input += self.bias_matrix
        weighted_input_gradients_stopped = tf.stop_gradient(weighted_input)

        vj = None
        for iteration in range(self.routing_iterations):
            cij = tf.keras.activations.softmax(
                bij, axis=-1
            )  # (batch_size, num_caps_prev, num_caps).
            cij = tf.expand_dims(
                cij, -1
            )  # (batch_size, num_caps_prev, num_caps, 1) "Coupling coefficients"

            if iteration == self.routing_iterations - 1:
                vj = (
                    cij * weighted_input
                )  # (batch_size, num_caps_prev, num_caps, depth)
                vj = tf.reduce_sum(vj, axis=1)  # (batch_size, num_caps, depth)
                if self.squash_last:
                    vj = self.squash(vj)  # (batch_size, num_caps, depth)
            else:
                vj = (
                    cij * weighted_input_gradients_stopped
                )  # (batch_size, num_caps_prev, num_caps, depth)
                vj = tf.reduce_sum(vj, axis=1)  # (batch_size, num_caps, depth)
                vj = self.squash(vj)  # (batch_size, num_caps, depth)

                # updating logits
                bij += tf.reduce_sum(
                    tf.expand_dims(vj, 1) * weighted_input_gradients_stopped, axis=-1
                )  # bij += (batch_size, num_caps_prev, num_caps)

        return vj

    def squash(self, inp):
        squared_magnitude_inp = tf.reduce_sum(
            (inp**2), axis=-1
        )  # (batch_size, num_caps)
        squared_magnitude_inp = tf.expand_dims(
            squared_magnitude_inp, -1
        )  # (batch_size, num_caps, 1)
        scale = squared_magnitude_inp / (
            1 + squared_magnitude_inp
        )  # (batch_size, num_caps, 1)
        inp = (
            scale * inp / tf.math.sqrt(squared_magnitude_inp)
        )  # (batch_size, num_caps, depth)
        return inp


class CapsuleDronet(tf.keras.Model):
    def __init__(self, num_outputs, include_top=True):
        super(CapsuleDronet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            activation="relu",
            data_format="channels_last",
            padding="same",
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation="relu",
            data_format="channels_last",
            padding="same",
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=5,
            activation="relu",
            data_format="channels_last",
            padding="same",
        )
        self.caps1depth = 32
        self.caps2depth = 64
        self.caps1_ncaps = 16
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        self.caps1 = None
        self.caps2 = CapsuleLayer(
            1,
            self.caps1_ncaps,
            128,
            self.caps2depth,
            routing_iterations=3,
            squash_last=False,
        )
        self.dense0 = tf.keras.layers.Dense(units=64, activation="relu")
        self.dense1 = tf.keras.layers.Dense(units=32, activation="relu")
        self.dense2 = tf.keras.layers.Dense(units=num_outputs, activation="linear")

    def call(self, inp):
        if not self.caps1:
            x = self.conv1(inp)
            x = self.maxpool(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = tf.reshape(x, [x.shape[0], -1, self.caps1depth])
            self.caps1 = CapsuleLayer(
                self.caps1_ncaps,
                x.shape[1],
                self.caps2depth,
                self.caps1depth,
                routing_iterations=3,
                squash_last=True,
            )
        x = self.conv1(inp)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = tf.reshape(x, [x.shape[0], -1, self.caps1depth])
        x = self.caps1(x)
        x = self.caps2(x)  # (batch, 1, depth)
        x = tf.reshape(x, [x.shape[0], -1])

        x = self.dense0(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x
