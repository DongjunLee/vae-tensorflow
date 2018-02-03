#-*- coding: utf-8 -*-

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt

from hbconfig import Config
import numpy as np
import tensorflow as tf

from model import Model



def generate(latent_vector):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"latent_vector": latent_vector},
        num_epochs=1,
        shuffle=False)

    estimator = _make_estimator()
    result = estimator.predict(input_fn=predict_input_fn)

    predictions = [image["prediction"] for image in list(result)]
    return predictions


def _make_estimator():
    params = tf.contrib.training.HParams(**Config.model.to_dict())
    # Using CPU
    run_config = tf.contrib.learn.RunConfig(
        model_dir=Config.train.model_dir,
        session_config=tf.ConfigProto(
            device_count={'GPU': 0}
        ))

    model = Model()
    return tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=Config.train.model_dir,
        params=params,
        config=run_config)


def main():
    # Sample noise vectors from N(0, 1)
    latent_vector = np.random.normal(size=[Config.model.batch_size, Config.model.z_dim]).astype(np.float32)
    generated_x = generate(latent_vector)
    generated_x = np.array(generated_x)

    n = np.sqrt(Config.model.batch_size).astype(np.int32)
    w = h = 28

    img = np.empty((h*n, w*n))
    for i in range(n):
        for j in range(n):
            img[i*h:(i+1)*h, j*w:(j+1)*w] = generated_x[i*n+j, :].reshape(28, 28)

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray_r')
    plt.savefig(f"generate_image_z_{Config.model.z_dim}_{Config.model.batch_size}.png")
    plt.close()
    print("success to generate image.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='set generate image count (default 20)')
    args = parser.parse_args()

    Config(args.config)
    Config.model.batch_size = args.batch_size

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    main()
