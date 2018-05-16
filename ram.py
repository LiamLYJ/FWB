import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import rnn_decoder
from tensorflow.python.ops.distributions.normal import Normal
from utils import *
from ops import *


def _weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.01)
    return tf.Variable(initial)


def _bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def _log_likelihood(loc_means, locs, variance):
    loc_means = tf.stack(loc_means)  # [timesteps, batch_sz, loc_dim]
    locs = tf.stack(locs)
    gaussian = Normal(loc_means, variance)
    logll = gaussian._log_prob(x=locs)  # [timesteps, batch_sz, loc_dim]
    logll = tf.reduce_sum(logll, 2)
    return tf.transpose(logll)      # [batch_sz, timesteps]


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
        self.g1_w = _weight_variable((pth_size*pth_size*img_channel, g_size))
        self.g1_b = _bias_variable((g_size,))
        self.l1_w = _weight_variable((loc_dim, l_size))
        self.l1_b = _bias_variable((l_size,))
        # layer 2
        self.g2_w = _weight_variable((g_size, output_size))
        self.g2_b = _bias_variable((output_size,))
        self.l2_w = _weight_variable((l_size, output_size))
        self.l2_b = _bias_variable((output_size,))

    def __call__(self, imgs_ph, locs):
        pths = self.retina_sensor(imgs_ph, locs)

        g = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(
            pths, self.g1_w, self.g1_b)), self.g2_w, self.g2_b)
        l = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(
            locs, self.l1_w, self.l1_b)), self.l2_w, self.l2_b)

        return tf.nn.relu(g + l)


class LocationNetwork(object):
    def __init__(self, loc_dim, rnn_output_size, variance=0.22, is_sampling=False):
        self.loc_dim = loc_dim
        self.variance = variance
        self.w = _weight_variable((rnn_output_size, loc_dim))
        self.b = _bias_variable((loc_dim,))

        self.is_sampling = is_sampling

    def __call__(self, cell_output):
        mean = tf.nn.xw_plus_b(cell_output, self.w, self.b)
        mean = tf.clip_by_value(mean, -1., 1.)
        mean = tf.stop_gradient(mean)

        if self.is_sampling:
            loc = mean + tf.random_normal(
                (tf.shape(cell_output)[0], self.loc_dim),
                stddev=self.variance)
            loc = tf.clip_by_value(loc, -1., 1.)
        else:
            loc = mean
        loc = tf.stop_gradient(loc)
        return loc, mean

class WhiteBalanceNetwork(object):
    def __init(self, rnn_output_size, output_channel=3):
        self.input_size = rnn_output_size
        self.w = _weight_variable((rnn_output_size, output_channel))
        self.b = _bias_variable((output_channel,))

    def __call__(self, rnn_output):
        est = tf.nn.xw_plus_b(rnn_output, self.w, self.b)
        est = tf.nn.l2_normalize9(est, axis =1 )
        est = est
        return est
