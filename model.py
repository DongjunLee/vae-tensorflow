from __future__ import print_function


from hbconfig import Config
import tensorflow as tf

import variational_autoencoder



class Model:

    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.dtype = tf.float32

        self.mode = mode
        self.params = params

        self.loss, self.train_op, self.metrics, self.predictions = None, None, None, None
        self._init_placeholder(features, labels)
        self.build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops=self.metrics,
            predictions={"prediction": self.predictions})

    def _init_placeholder(self, features, labels):
        if self.mode == tf.estimator.ModeKeys.PREDICT:
            self.inputs = features["latent_vector"]
        else:
            self.inputs = features
            if type(features) == dict:
                self.inputs = features["input_data"]
            self.targets = labels

    def build_graph(self):
        graph = variational_autoencoder.Graph(self.mode)
        output = graph.build(inputs=self.inputs)

        self._build_prediction(output)
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(output, graph.z_mean, graph.z_stddev)
            self._build_optimizer()
            self._build_metric()

    def _build_prediction(self, output):
        self.predictions = output

    def _build_loss(self, output, mean, stddev):
        with tf.variable_scope('loss'):
            reconstruction_error = -tf.reduce_sum(self.targets * tf.log(output) + (1 - self.targets) * tf.log(1 - output), axis=1)
            reconstruction_error = tf.reduce_mean(reconstruction_error, name="reconstruction_error")

            kl_divergence = -0.5 * tf.reduce_sum(1 + tf.log(tf.square(stddev)) - tf.square(mean) - tf.square(stddev), axis=1)
            kl_divergence = tf.reduce_mean(kl_divergence, name="kl_divergence")

            self.loss = tf.reduce_mean(kl_divergence + reconstruction_error)

        tf.summary.scalar("loss/reconstruction_error", reconstruction_error)
        tf.summary.scalar("loss/kl_divergence", kl_divergence)

    def _build_optimizer(self):
        self.train_op = tf.contrib.layers.optimize_loss(
            self.loss, tf.train.get_global_step(),
            optimizer=Config.train.get('optimizer', 'Adam'),
            learning_rate=Config.train.learning_rate,
            summaries=['gradients', 'learning_rate'],
            name="train_op")

    def _build_metric(self):
        self.metrics = {}
