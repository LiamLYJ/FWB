import os
import sys
import tensorflow as tf
from ops import *
from utils import *

class LocationNetwork(object):
    def __init__(self, loc_dim, rnn_output_size, variance=0.22, is_sampling=False):
        self.loc_dim = loc_dim
        self.variance = variance
        self.w = weight_variable((rnn_output_size, loc_dim))
        self.b = bias_variable((loc_dim,))

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
    def __init__(self, rnn_output_size, output_dim):
        self.input_size = rnn_output_size
        self.output_dim = output_dim
        self.w = weight_variable((rnn_output_size, output_dim))
        self.b = bias_variable((output_dim,))

    def __call__(self, rnn_output):
        est = tf.nn.xw_plus_b(rnn_output, self.w, self.b)
        est = tf.nn.l2_normalize(est, axis =1 )
        est = est
        return est
