
from hbconfig import Config
import tensorflow as tf



class Graph:

    def __init__(self, mode, dtype=tf.float32):
        self.mode = mode
        self.dtype = dtype

    def build(self, inputs, latent_z=None):
        # TODO: predict mode logic
        # latent_z -> generate

        batch_size = tf.shape(inputs)[0]

        # Regularization Parameter
        self.z_mean, self.z_stddev = self._build_encoder(inputs)

        # z = µ + σ * N (0, 1)
        with tf.variable_scope('sample-z'):
            sample_z = self.z_mean + self.z_stddev * tf.random_normal(
                [batch_size, Config.model.z_dim], mean=0, stddev=1, dtype=self.dtype)
        output = self._build_decoder(sample_z)

        return output

    def _build_encoder(self, inputs):
        with tf.variable_scope('Encoder'):
            h1 = tf.layers.dense(inputs, Config.model.encoder_h1, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, Config.model.encoder_h2, activation=tf.nn.relu)

            mean = tf.layers.dense(h2, Config.model.z_dim, name="z_mean")
            stddev = tf.layers.dense(h2, Config.model.z_dim, name="z_stddev")
        return mean, stddev

    def _build_decoder(self, sample_z):
        with tf.variable_scope('Decoder'):
            h1 = tf.layers.dense(sample_z, Config.model.decoder_h1, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, Config.model.decoder_h2, activation=tf.nn.relu)
            output = tf.layers.dense(h2, Config.model.n_output, activation=tf.nn.sigmoid)
        return output
