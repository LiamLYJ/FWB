from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import sys
import tensorflow as tf
import cv2
from put_all import fwb_net
import logging
import numpy as np
import pickle
from tensorflow.examples.tutorials.mnist import input_data
from data_provider import *
from utils import *

logging.getLogger().setLevel(logging.INFO)

AUGMENTATION = False
# Rotation angle
AUGMENTATION_ANGLE = 60
# Patch scale
AUGMENTATION_SCALE = [0.1, 1.0]
# Random left-right flip?
AUGMENTATION_FLIP_LEFTRIGHT = True
# Random top-down flip?
AUGMENTATION_FLIP_TOPDOWN = False
# Color rescaling?
AUGMENTATION_COLOR = 0.10
# Cross-channel terms
AUGMENTATION_COLOR_OFFDIAG = 0.0
# Augment Gamma?
AUGMENTATION_GAMMA = 0.0
# Augment using a polynomial curve?
USE_CURVE = False
# Apply different gamma and curve to left/right halves?
SPATIALLY_VARIANT = False

tf.app.flags.DEFINE_boolean('DATA_SHUFFLE', 'True', 'if use data shuffle')
tf.app.flags.DEFINE_boolean('AUGMENTATION', 'True', 'if use data augmentation')
tf.app.flags.DEFINE_float('AUGMENTATION_ANGLE', '60', 'augmentation angle')
tf.app.flags.DEFINE_float('AUGMENTATION_SCALE_LOW', '0.1', 'augmentation scale low bound')
tf.app.flags.DEFINE_float('AUGMENTATION_SCALE_HIGH', '60', 'augmentation scale high bound')
tf.app.flags.DEFINE_boolean('AUGMENTATION_FLIP_TOPDOWN', 'False', 'augmentation todown')
tf.app.flags.DEFINE_boolean('AUGMENTATION_FLIP_LEFTRIGHT', 'True', 'augmentation todown')
tf.app.flags.DEFINE_float('AUGMENTATION_COLOR', '0.10', 'augmentation color')
tf.app.flags.DEFINE_float('AUGMENTATION_COLOR_OFFDIAG', '0.0', 'augmentation color offdiag')
tf.app.flags.DEFINE_float('AUGMENTATION_GAMMA', '0.0', 'augmentation gamma')
tf.app.flags.DEFINE_boolean('USE_CURVE', 'False', 'Use curve')
tf.app.flags.DEFINE_boolean('SPATIALLY_VARIANT', 'False', 'spatially_variant')

tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.97,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("min_learning_rate", 1e-4, "Minimum learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer(
    "batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_steps", 100000, "Number of training steps.")
tf.app.flags.DEFINE_integer("save_steps", 1000, "Number of saving steps.")
tf.app.flags.DEFINE_integer("test_steps", 20, "Number of test steps.")

tf.app.flags.DEFINE_integer("patch_window_size", 8,
                            "Size of glimpse patch window.")
tf.app.flags.DEFINE_integer("g_size", 128, "Size of theta_g^0.")
tf.app.flags.DEFINE_integer("l_size", 128, "Size of theta_g^1.")
tf.app.flags.DEFINE_integer("glimpse_output_size",
                            256, "Output size of Glimpse Network.")
tf.app.flags.DEFINE_integer("cell_size", 256, "Size of LSTM cell.")
tf.app.flags.DEFINE_integer("num_glimpses", 6, "Number of glimpses.")
tf.app.flags.DEFINE_float(
    "variance", 0.22, "Gaussian variance for Location Network.")
tf.app.flags.DEFINE_integer("M", 10, "Monte Carlo sampling, see Eq(2).")

tf.app.flags.DEFINE_integer("img_size", 64, "image size")
tf.app.flags.DEFINE_integer("img_channel", 3, "image channel")
tf.app.flags.DEFINE_integer("output_dim", 3, "est dims")

tf.app.flags.DEFINE_integer("fc1_size", 128, "final fc before the output of discriminator")
tf.app.flags.DEFINE_integer("base_channels", 32, "base channel of discriminator")

tf.app.flags.DEFINE_string('summary_dir', './log', 'summary file dir')
tf.app.flags.DEFINE_string('checkpoint_dir', './check_point', 'check point file dir')
tf.app.flags.DEFINE_string('model_name', 'RNN', 'model name')

TRAINING_FOLDS = ['s0']
FLAGS = tf.app.flags.FLAGS

if not os.path.exists(FLAGS.summary_dir):
    os.mkdir(FLAGS.summary_dir)
if not os.path.exists(FLAGS.checkpoint_dir):    
    os.mkdir(FLAGS.checkpoint_dir)

training_data_provider = DataProvider(True, TRAINING_FOLDS)
training_data_provider.set_batch_size(FLAGS.batch_size)
training_steps_per_epoch = training_data_provider.data_count // FLAGS.batch_size

ram = fwb_net(img_channel=FLAGS.img_channel,
              img_size=FLAGS.img_size,
              pth_size=FLAGS.patch_window_size,
              g_size=FLAGS.g_size,
              l_size=FLAGS.l_size,
              glimpse_output_size=FLAGS.glimpse_output_size,
              loc_dim=2,   # (x,y)
              variance=FLAGS.variance,
              cell_size=FLAGS.cell_size,
              num_glimpses=FLAGS.num_glimpses,
              num_classes=FLAGS.output_dim,
              learning_rate=FLAGS.learning_rate,
              learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
              min_learning_rate=FLAGS.min_learning_rate,
              training_steps_per_epoch=training_steps_per_epoch,
              max_gradient_norm=FLAGS.max_gradient_norm,
              fc1_size=FLAGS.fc1_size,
              base_channels=FLAGS.base_channels,
              output_dim = FLAGS.output_dim,
              is_training=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

    for step in range(FLAGS.num_steps):
        batch  = training_data_provider.get_batch()
        images, labels = batch[0], batch[2]
        # output from provider is BGR
        images = images[:,:,:,::-1]
        images = images / 65535.0
        images = (np.power(images, 1/2.2)*255.0)[:,:,:,::-1]
        images = images - np.array([104.0,
                                            117.0,123.0]).reshape([1,1,1,3])
        images = np.reshape(images, (FLAGS.batch_size, -1))

        images = np.tile(images, [FLAGS.M, 1])
        labels = np.tile(labels, [FLAGS.M, 1])

        output_feed = [ram.train_op, ram.loss, ram.xent, ram.reward,
                       ram.advantage, ram.baselines_mse, ram.learning_rate, ram.sum_total]
        _, loss, xent, reward, advantage, baselines_mse, learning_rate, sum_total = \
                                                    sess.run(output_feed,
                                                              feed_dict={
                                                                  ram.img_ph: images,
                                                                  ram.lbl_ph: labels
                                                              })

        if step and step % 10 == 0:
            writer.add_summary(sum_total, step)
            logging.info(
                'step {}: lr = {:3.6f}\tloss = {:3.4f}\txent = {:3.4f}\treward = {:3.4f}\tadvantage = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
                    step, learning_rate, loss, xent, reward, advantage, baselines_mse))

        if step and step % FLAGS.save_steps == 0:
            checkpoint_dir = FLAGS.checkpoint_dir
            model_name = FLAGS.model_name
            ram.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step = step)

        if step % FLAGS.test_steps == 0:
        # if step and step % FLAGS.test_steps == 0:
            f = open('./data/sony/image_pack.0.pkl', 'br')
            bb = pickle.load(f, encoding = "UFT-8")
            gts = []
            ests = []
            errors = []
            for i in range(len(bb)):
                gt = bb[i].illum
                gts.append(gt)
                image = bb[i].img
                image = image / 65535.0
                image = cv2.resize(image, (FLAGS.img_size, FLAGS.img_size))
                image = np.expand_dims(image, 0)
                alex_image = (np.power(image, 1/2.2)*255.0)[:,:,:,::-1]
                input_data = alex_image - np.array([104.0, 117.0,
                                                     123.0]).reshape([1,1,1,3])

                input_data = np.reshape(input_data, (1, -1))
                est = sess.run(ram.prediction, feed_dict = {ram.img_ph :
                                                   input_data})            
                est = np.squeeze(est)
                ests.append(est)
                error = angular_error(gt, est)
                errors.append(error)
            print_angular_errors(errors)



        # # Evaluation
        # if step and step % training_steps_per_epoch == 0:
        #     for dataset in [mnist.validation, mnist.test]:
        #         steps_per_epoch = dataset.num_examples // FLAGS.batch_size
        #         correct_cnt = 0
        #         num_samples = steps_per_epoch * FLAGS.batch_size
        #         for test_step in range(steps_per_epoch):
        #             images, labels = dataset.next_batch(FLAGS.batch_size)
        #             labels_bak = labels
        #             # Duplicate M times
        #             images = np.tile(images, [FLAGS.M, 1])
        #             labels = np.tile(labels, [FLAGS.M])
        #             softmax = sess.run(ram.softmax,
        #                                feed_dict={
        #                                    ram.img_ph: images,
        #                                    ram.lbl_ph: labels
        #                                })
        #             softmax = np.reshape(softmax, [FLAGS.M, -1, 10])
        #             softmax = np.mean(softmax, 0)
        #             prediction = np.argmax(softmax, 1).flatten()
        #             correct_cnt += np.sum(prediction == labels_bak)
        #         acc = correct_cnt / num_samples
        #         if dataset == mnist.validation:
        #             logging.info('valid accuracy = {}'.format(acc))
        #         else:
        #             logging.info('test accuracy = {}'.format(acc))


#
# def main(_):
#     run_config = tf.ConfigProto()
#     run_config.gpu_options.allow_growth=True
#     with tf.Session(config = run_config) as sess:
#         net = fwb_net(sess, flags = FLAGS)
#         net.train()
#
# if __name__ == '__main__':
#     tf.app.run()
