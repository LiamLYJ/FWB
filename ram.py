import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import rnn_decoder
from utils import *
from ops import *


class RetinaSensor(object):
    # one scale
    def __init__(self, img_channel, img_size, pth_size):
        self.img_size = img_size
        self.pth_size = pth_size
        self.img_channel = img_channel

    def __call__(self, img_ph, loc):
        img = tf.reshape(img_ph, [
            tf.shape(img_ph)[0],
            self.img_size,
            self.img_size,
            self.img_channel
        ])
        pth = tf.image.extract_glimpse(
            img,
            [self.pth_size, self.pth_size],
            loc)
        return tf.reshape(pth, [tf.shape(loc)[0], self.pth_size*self.pth_size*self.img_channel])


class GlimpseNetwork(object):
    def __init__(self, img_channel, img_size, pth_size, loc_dim, g_size, l_size, output_size):
        self.retina_sensor = RetinaSensor(img_channel, img_size, pth_size)

        self.img_channel = img_channel
        self.img_size = img_size
        self.pth_size = pth_size
        self.loc_dim = loc_dim
        self.g_size = g_size
        self.l_size = l_size
        self.output_size = output_size

        # layer 1
        self.g1_w = weight_variable((pth_size*pth_size*img_channel, g_size))
        self.g1_b = bias_variable((g_size,))
        self.l1_w = weight_variable((loc_dim, l_size))
        self.l1_b = bias_variable((l_size,))
        # layer 2
        self.g2_w = weight_variable((g_size, output_size))
        self.g2_b = bias_variable((output_size,))
        self.l2_w = weight_variable((l_size, output_size))
        self.l2_b = bias_variable((output_size,))

    def __call__(self, imgs_ph, locs):
        pths = self.retina_sensor(imgs_ph, locs)

        g = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(
            pths, self.g1_w, self.g1_b)), self.g2_w, self.g2_b)
        l = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(
            locs, self.l1_w, self.l1_b)), self.l2_w, self.l2_b)

        return tf.nn.relu(g + l)
