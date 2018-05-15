from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from put_all import fwb_net
import logging
import numpy as np

logging.getLogger().setLevel(logging.INFO)

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.97,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("min_learning_rate", 1e-4, "Minimum learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_steps", 100000, "Number of training steps.")

tf.app.flags.DEFINE_integer("patch_window_size", 8, "Size of glimpse patch window.")
tf.app.flags.DEFINE_integer("g_size", 128, "Size of theta_g^0.")
tf.app.flags.DEFINE_integer("l_size", 128, "Size of theta_g^1.")
tf.app.flags.DEFINE_integer("glimpse_output_size", 256, "Output size of Glimpse Network.")
tf.app.flags.DEFINE_integer("cell_size", 256, "Size of LSTM cell.")
tf.app.flags.DEFINE_integer("num_glimpses", 6, "Number of glimpses.")
tf.app.flags.DEFINE_float("variance", 0.22, "Gaussian variance for Location Network.")
tf.app.flags.DEFINE_integer("M", 10, "Monte Carlo sampling, see Eq(2).")

FLAGS = tf.app.flags.FLAGS



def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
    with tf.Session(config = run_config) as sess:
        net = fwb_net(sess, flags = FLAGS)
        net.train()

if __name__ == '__main__':
    tf.app.run()
