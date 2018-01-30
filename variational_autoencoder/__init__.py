
import tensorflow as tf



class Graph:

    def __init__(self, mode, dtype=tf.float32):
        self.mode = mode
        self.dtype = dtype

    def build(self, inputs):
        # TODO: predict mode logic
        # latent -> generate

        # Simple Auto Encoder

        # Encoder
        h1 = tf.layers.dense(inputs, 256, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, 128, activation=tf.nn.relu)
        # Decoder
        h3 = tf.layers.dense(h2, 256, activation=tf.nn.relu)
        output = tf.layers.dense(h3, 784, activation=tf.nn.sigmoid)
        return output

    def _build_encoder(self, inputs):
        pass

    def _build_decoder(self, latents):
        pass

