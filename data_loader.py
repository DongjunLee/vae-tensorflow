# -*- coding: utf-8 -*-

import numpy as np
from hbconfig import Config
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data




def make_train_and_test_set(shuffle=True):
    print("make Training data and Test data Start....")

    mnist = input_data.read_data_sets("MNIST data", one_hot=True)

    # load train and test dataset
    train_X = mnist.train.images
    test_X = mnist.test.images

    print(f"train data count : {train_X.shape[0]}")
    print(f"test data count : {test_X.shape[0]}")

    if shuffle:
        print("shuffle dataset ...")
        train_p = np.random.permutation(train_X.shape[0])
        test_p = np.random.permutation(test_X.shape[0])

        return train_X[train_p], test_X[test_p]
    else:
        return train_X, test_X


def make_batch(X, buffer_size=10000, batch_size=64, scope="train"):

    class IteratorInitializerHook(tf.train.SessionRunHook):
        """Hook to initialise data iterator after Session is created."""

        def __init__(self):
            super(IteratorInitializerHook, self).__init__()
            self.iterator_initializer_func = None

        def after_create_session(self, session, coord):
            """Initialise the iterator after the session has been created."""
            self.iterator_initializer_func(session)


    iterator_initializer_hook = IteratorInitializerHook()

    def inputs():
        with tf.name_scope(scope):

            # Define placeholders
            input_placeholder = tf.placeholder(
                tf.float32, [None, 784], name="input_placeholder")
            target_placeholder = tf.placeholder(
                tf.float32, [None, 784], name="target_placeholder")

            # Build dataset iterator
            dataset = tf.data.Dataset.from_tensor_slices(
                (input_placeholder, target_placeholder))

            if scope == "train":
                dataset = dataset.repeat(None)  # Infinite iterations
            else:
                dataset = dataset.repeat(1)  # one Epoch

            dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_initializable_iterator()
            next_input, next_target = iterator.get_next()

            tf.identity(next_input[0], 'input_0')
            tf.identity(next_target[0], 'target_0')

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: X,
                               target_placeholder: X})

            # Return batched (features, labels)
            return next_input, next_target

    # Return function and hook
    return inputs, iterator_initializer_hook
