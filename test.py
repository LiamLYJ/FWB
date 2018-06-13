import os
import sys
import tensorflow as tf
import cv2
from put_all import fwb_net
import logging
import numpy as np
import pickle
from data_provider import *
from utils import *
import numpy as np
logging.getLogger().setLevel(logging.INFO)

from train import FLAGS

fwb = fwb_net(img_channel=FLAGS.img_channel,
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
              training_steps_per_epoch=0, # now its test mode
              max_gradient_norm=FLAGS.max_gradient_norm,
              fc1_size=FLAGS.fc1_size,
              base_channels=FLAGS.base_channels,
              output_dim=FLAGS.output_dim,
              is_training = False)

saver = tf.train.Saver()
check_point = './check_point/RNN-40'
f = open('./data/sony/image_pack.0.pkl', 'br')
if_draw = True
bb = pickle.load(f, encoding="UFT-8")
gts = []
ests = []
errors = []
location = []
with tf.Session() as sess:
    saver.restore(sess, check_point)
    for i in range(len(bb)):
        gt = bb[i].illum
        gts.append(gt)
        image = bb[i].img
        image = image / 65535.0
        image = cv2.resize(image, (FLAGS.img_size, FLAGS.img_size))
        image = np.expand_dims(image, 0)
        alex_image = (np.power(image, 1/2.2)*255.0)[:, :, :, ::-1]
        input_data = alex_image - np.array([104.0, 117.0,
                                            123.0]).reshape([1, 1, 1, 3])

        input_data = np.reshape(input_data, (1, -1))
        est, locations  = sess.run([fwb.prediction, fwb.locations], feed_dict={fwb.img_ph:
                                                  input_data})
        if if_draw:
            for j in range(len(locations)):
                y,x = locations[j][0]
                center_x = FLAGS.img_size / 2
                center_y = FLAGS.img_size / 2
                loc_x = center_x + abs(x)*center_x if x >= 0 else center_x - abs(x)*center_x
                loc_y = center_y + abs(y)*center_y if y >= 0 else center_y - abs(y)*center_y
                pt1 = (int(loc_x - FLAGS.patch_window_size), int(loc_y -
                                                                 FLAGS.patch_window_size))
                pt2 = (int(loc_x + FLAGS.patch_window_size), int(loc_y +
                                                                 FLAGS.patch_window_size))
                template = bb[i].img / 65535.0
                template = cv2.resize(template, (FLAGS.img_size, FLAGS.img_size))
                template = np.power(template, 1/2.2)
                cv2.rectangle(template, pt1, pt2,(0,255,0), 3 )
                cv2.imshow('image', template)
                cv2.waitKey(0)
        est = np.squeeze(est)
        ests.append(est)
        error = angular_error(gt, est)
        errors.append(error)
    print_angular_errors(errors)
