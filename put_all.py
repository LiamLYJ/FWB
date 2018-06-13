import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import rnn_decoder
from tensorflow.python.ops.distributions.normal import Normal
from utils import *
from ops import *
from ram import *
from agent import *
from critic import *

FLAGS = tf.app.flags.FLAGS

# contain three subpart : GlimpseNetwork, Agent, critic. Connected by RNN
class fwb_net(object):
    def __init__(self, img_channel, img_size, pth_size, g_size, l_size, glimpse_output_size,
                 loc_dim, variance,
                 cell_size, num_glimpses, num_classes,
                 learning_rate, learning_rate_decay_factor, min_learning_rate, training_steps_per_epoch,
                 max_gradient_norm, fc1_size, base_channels, output_dim,
                 is_training=False):

        self.img_ph = tf.placeholder(
            tf.float32, [None, img_size*img_size*img_channel])
        self.lbl_ph = tf.placeholder(tf.float32, [None, output_dim])

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
        with tf.variable_scope('Agent'):
            # the agent is resposibale for select a windows and est a gain
            with tf.variable_scope('LocationNetwork'):
                location_network = LocationNetwork(
                    loc_dim=loc_dim, rnn_output_size=cell.output_size, variance=variance, is_sampling=is_training)
            with tf.variable_scope('WhiteBalanceNetwork'):
                wb_network = WhiteBalanceNetwork(rnn_output_size = cell.output_size,output_dim=output_dim)
        if FLAGS.USE_CRITIC:
            with tf.variable_scope('Critic'):
                critic_network = CriticNetwork(fc1_size, base_channels)


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

        def _apply_gain(ill, loc, img, patch_wise = False):
            if patch_wise:
                retina = RetinaSensor(img_channel, img_size, pth_size)
                pth = retina(img, loc, serial = False)
                img = tf.reshape(img, [
                    tf.shape(img)[0],
                    img_size,
                    img_size,
                    img_channel
                ])
                retouched_channel = []
                for i in range(3):
                    tmp = pth[:,:,:,i]
                    tmp = tf.reshape(tmp, [tf.shape(tmp)[0], -1])
                    tmp_ill = tf.reshape(ill[:,i]/ill[:,1],[tf.shape(img)[0], 1])
                    tmp_ill = tf.tile(tmp_ill, [1, pth_size * pth_size])
                    tmp *= tmp_ill
                    retouched_channel.append(tmp)
                retouched = tf.concat(retouched_channel, -1)
                img[:,
                    round(img_size*loc[0]) - pth_size : round(img_size*loc[0]) + pth_size,
                    round(img_size*loc[1]) - pth_size : round(img_size*loc[1]) + pth_size,
                    :] = retouched
            else:
                img = tf.reshape(img, [
                    tf.shape(img)[0], img_size, img_size, img_channel ] )
                retouched_channel = []
                for i in range(3):
                    tmp = img[:,:,:,i]
                    tmp = tf.reshape(tmp, [tf.shape(tmp)[0], -1])
                    tmp_ill = tf.reshape(ill[:,i] / ill[:,1],
                                         [tf.shape(img)[0], 1] )
                    tmp_ill = tf.tile(tmp_ill, [1, img_size * img_size])
                    tmp *= tmp_ill
                    retouched_channel.append(tmp)
                img = tf.concat(retouched_channel, -1)
            return img

        def _loop_function(prev, _):
            loc, loc_mean = location_network(prev)
            locs.append(loc)
            loc_means.append(loc_mean)
            gain = wb_network(prev)
            gains.append(gain)
            if img_retouched:
                img_retouched.append(_apply_gain(gain, loc, img_retouched[-1])
                                     )
                glimpse = glimpse_network(img_retouched[-1], loc)
            else:
                img_retouched.append(_apply_gain(gain, loc, self.img_ph))
                glimpse = glimpse_network(self.img_ph, loc)
            return glimpse


        rnn_outputs, _ = rnn_decoder(
            rnn_inputs, init_state, cell, loop_function=_loop_function)

        assert len(gains) == len(locs)
        # Time independent baselines
        with tf.variable_scope('Baseline'):
            baseline_w = weight_variable((cell.output_size, 1))
            baseline_b = bias_variable((1,))
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
            logit_w = weight_variable((cell.output_size, num_classes))
            logit_b = bias_variable((num_classes,))
        logits = tf.nn.xw_plus_b(rnn_last_output, logit_w, logit_b)
        # batch_size *3
        self.prediction = tf.nn.l2_normalize(logits, axis=1)
        self.locations = locs
        if is_training:
            # angular loss
            self.xent = get_angular_loss(self.prediction, self.lbl_ph)
            tf.summary.scalar('xent', self.xent)

            # RL reward
            # reward shape [batchsize, 1]
            if FLAGS.USE_CRITIC:

                img_critic = tf.reshape(self.img_ph, [
                    tf.shape(self.img_ph)[0], img_size, img_size, img_channel ] )
                img_real = apply_gain(img_critic, self.lbl_ph)
                img_real = tf.reshape(img_real, [tf.shape(img_real)[0], img_size, img_size, img_channel])
                img_fake = apply_gain(img_critic, self.prediction)
                img_fake = tf.reshape(img_fake, [tf.shape(img_fake)[0], img_size, img_size, img_channel])

                real_logit = critic_network(img_real, is_train = is_training, reuse = False)
                fake_logit = critic_network(img_fake, is_train = is_training, reuse = True)
                rnn_fake_logits = []
                for index_sequence in range(len(img_retouched)):
                    rnn_img_fake = tf.reshape(img_retouched[index_sequence], [
                        tf.shape(img_retouched[index_sequence])[0], img_size, img_size, img_channel ] )
                    rnn_fake_logit = critic_network(rnn_img_fake, is_train = is_training, reuse = True)
                    rnn_fake_logits.append(rnn_fake_logit)

                rewards = tf.stop_gradient(tf.convert_to_tensor(rnn_fake_logits)) # shape (timesteps, batch_sz, 1)
                rewards = tf.transpose(tf.squeeze(rewards,2)) # shape [batch_sz, timesteps]

                self.c_loss = tf.reduce_mean(fake_logit - real_logit)
                if FLAGS.grad_penalty < 0:
                    # use grad clip
                    gradients = tf.gradients(self.c_loss, theta_c)

                    clipped_gradients, norm = tf.clip_by_global_norm(
                        gradients, max_gradient_norm)
                    self.opt_c = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                        zip(clipped_gradients, params), global_step=self.global_step)

                else :
                        # Critic gradient norm and penalty
                    alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
                    alpha = alpha_dist.sample((batch_size, 1, 1, 1))
                    interpolated = img_real + alpha * (img_fake - img_real)

                    inte_logit = critic_network(
                        images=interpolated, is_train = is_training, reuse=True)

                    gradients = tf.gradients(inte_logit, [
                        interpolated,
                    ])[0]

                    gradient_norm = tf.sqrt(1e-6 + tf.reduce_sum(gradients**2, axis=[1, 2, 3]))
                    gradient_penalty = FLAGS.grad_penalty * tf.reduce_mean(
                        tf.maximum(gradient_norm - 1.0, 0.0)**2)

                    self.c_loss += gradient_penalty

                theta_c = tf.trainable_variables(scope = 'critic')

                gradients = tf.gradients(self.c_loss, theta_c)

                self.opt_c = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                    zip(gradients, theta_c), global_step=self.global_step)

            else :
                reward = tf.norm(self.prediction - self.lbl_ph, axis = 1)
                rewards= tf.expand_dims(reward,1)
                rewards = tf.tile(rewards, (1, num_glimpses))   # [batch_sz, timesteps]

            advantages = rewards - tf.stop_gradient(baselines)
            self.advantage = tf.reduce_mean(advantages)
            logll = log_likelihood(loc_means, locs, variance)
            logllratio = tf.reduce_mean(logll * advantages)
            self.reward = tf.reduce_mean(rewards)
            tf.summary.scalar('reward', self.reward)
            # baseline loss
            self.baselines_mse = tf.reduce_mean(
                tf.square((rewards - baselines)))
            # hybrid loss
            self.loss = -logllratio + self.xent + self.baselines_mse
            tf.summary.scalar('loss', self.loss)

            # exclude the variables in critic scope
            params_all = tf.trainable_variables()
            params = []
            for var in params_all:
                if not 'critic' in var.op.name:
                    params.append(var)
            gradients = tf.gradients(self.loss, params)

            clipped_gradients, norm = tf.clip_by_global_norm(
                gradients, max_gradient_norm)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

            img = tf.reshape(self.img_ph,
                [ tf.shape(self.img_ph)[0],
                img_size,
                img_size,
                img_channel] )
            tf.summary.image('input', img)
            tf.summary.image('gt', apply_gain(img, self.lbl_ph))
            tf.summary.image('est', apply_gain(img, self.prediction))
            self.sum_total = tf.summary.merge_all()

        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=99999999)
