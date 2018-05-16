import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import rnn_decoder
from tensorflow.python.ops.distributions.normal import Normal
from utils import *
from ops import *
from ram import *
from sub_nets import *




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



class fwb_net(object):
    def __init__(self, img_channel, img_size, pth_size, g_size, l_size, glimpse_output_size,
                 loc_dim, variance,
                 cell_size, num_glimpses, num_classes,
                 learning_rate, learning_rate_decay_factor, min_learning_rate, training_steps_per_epoch,
                 max_gradient_norm,
                 is_training=False):

        self.img_ph = tf.placeholder(
            tf.float32, [None, img_size*img_size*img_channel])
        self.lbl_ph = tf.placeholder(tf.int64, [None])

        self.global_step = tf.Variable(0, trainable=False)

        self.learning_rate = tf.maximum(tf.train.exponential_decay(
            learning_rate, self.global_step,
            training_steps_per_epoch,
            learning_rate_decay_factor,
            staircase=True),
            min_learning_rate)

        cell = BasicLSTMCell(cell_size)

        with tf.variable_scope('GlimpseNetwork'):
            glimpse_network = GlimpseNetwork(
                img_channel, img_size, pth_size, loc_dim, g_size, l_size, glimpse_output_size)
        with tf.variable_scope('LocationNetwork'):
            location_network = LocationNetwork(
                loc_dim=loc_dim, rnn_output_size=cell.output_size, variance=variance, is_sampling=is_training)
        with tf.variable_scope('WhiteBalanceNetwork'):
            wb_network = WhiteBalanceNetwork(rnn_output_size = cell.output_size)


        # Core Network
        batch_size = tf.shape(self.img_ph)[0]
        init_loc = tf.random_uniform(
            (batch_size, loc_dim), minval=-1, maxval=1)
        init_state = cell.zero_state(batch_size, tf.float32)

        init_glimpse = glimpse_network(self.img_ph, init_loc)
        rnn_inputs = [init_glimpse]
        rnn_inputs.extend([0] * num_glimpses)

        locs, loc_means = [], []
        gains = []
        img_retouched = []
        def loop_function(prev, _):
            loc, loc_mean = location_network(prev)
            locs.append(loc)
            loc_means.append(loc_mean)
            gain = wb_network(prev)
            if img_retouched:
                img_retouched.append(_apply_gain(gain, loc, img_retouched[-1])
            else:
                img_retouched.append(_apply_gain(gain, loc, self.img_ph))
            gains.append(gain)
            glimpse = glimpse_network(self.img_ph, loc)
            return glimpse
        rnn_outputs, _ = rnn_decoder(
            rnn_inputs, init_state, cell, loop_function=loop_function)

        # Time independent baselines
        with tf.variable_scope('Baseline'):
            baseline_w = _weight_variable((cell.output_size, 1))
            baseline_b = _bias_variable((1,))
        baselines = []
        for output in rnn_outputs[1:]:
            baseline = tf.nn.xw_plus_b(output, baseline_w, baseline_b)
            baseline = tf.squeeze(baseline)
            baselines.append(baseline)
        baselines = tf.stack(baselines)        # [timesteps, batch_sz]
        baselines = tf.transpose(baselines)   # [batch_sz, timesteps]

        # Classification. Take the last step only.
        rnn_last_output = rnn_outputs[-1]
        with tf.variable_scope('Classification'):
            logit_w = _weight_variable((cell.output_size, num_classes))
            logit_b = _bias_variable((num_classes,))
        logits = tf.nn.xw_plus_b(rnn_last_output, logit_w, logit_b)
        self.prediction = tf.argmax(logits, 1)
        self.softmax = tf.nn.softmax(logits)

        if is_training:
            # classification loss
            self.xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.lbl_ph, logits=logits))
            # RL reward
            reward = tf.cast(
                tf.equal(self.prediction, self.lbl_ph), tf.float32)
            rewards = tf.expand_dims(reward, 1)             # [batch_sz, 1]
            rewards = tf.tile(rewards, (1, num_glimpses)
                              )   # [batch_sz, timesteps]
            advantages = rewards - tf.stop_gradient(baselines)
            self.advantage = tf.reduce_mean(advantages)
            logll = _log_likelihood(loc_means, locs, variance)
            logllratio = tf.reduce_mean(logll * advantages)
            self.reward = tf.reduce_mean(reward)
            # baseline loss
            self.baselines_mse = tf.reduce_mean(
                tf.square((rewards - baselines)))
            # hybrid loss
            self.loss = -logllratio + self.xent + self.baselines_mse
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(
                gradients, max_gradient_norm)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=99999999)
