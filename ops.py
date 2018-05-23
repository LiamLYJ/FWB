import tensorflow as tf
import math
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.ops.distributions.normal import Normal

FLAGS = tf.app.flags.FLAGS

def apply_gain(img, ill):
    # rgb, same order img and ill
    after_apply = []
    for i in range(3):
        tmp = img[:,:,:,i]
        tmp = tf.reshape(tmp, [tf.shape(tmp)[0], -1])
        tmp_ill = tf.reshape(ill[:,i]/ill[:,1],[tf.shape(tmp)[0], 1])
        tmp_ill = tf.tile(tmp_ill, [1, FLAGS.img_size * FLAGS.img_size])
        one_channel = tmp_ill * tmp
        output = tf.reshape(one_channel, [tf.shape(tmp)[0], FLAGS.img_size,
                                          FLAGS.img_size, -1])
        after_apply.append(output)
    return tf.concat(after_apply, -1)


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def log_likelihood(loc_means, locs, variance):
    loc_means = tf.stack(loc_means)  # [timesteps, batch_sz, loc_dim]
    locs = tf.stack(locs)
    gaussian = Normal(loc_means, variance)
    logll = gaussian._log_prob(x=locs)  # [timesteps, batch_sz, loc_dim]
    logll = tf.reduce_sum(logll, 2)
    return tf.transpose(logll)      # [batch_sz, timesteps]



def get_angular_loss(vec1, vec2, length_regularization=0.0):
    with tf.name_scope('angular_error'):
        safe_v = 0.999999
        assert len(vec1.get_shape()) == 2
        illum_normalized = tf.nn.l2_normalize(vec1, 1)
        _illum_normalized = tf.nn.l2_normalize(vec2, 1)
        dot = tf.reduce_sum(illum_normalized * _illum_normalized, 1)
        dot = tf.clip_by_value(dot, -safe_v, safe_v)
        length_loss = tf.reduce_mean(
            tf.maximum(tf.log(tf.reduce_sum(vec1**2, axis=1) + 1e-7), 0))

        angle = tf.acos(dot) * (180 / math.pi)

        return tf.reduce_mean(angle) + length_loss * length_regularization



def leaky_relu(x):
    return tf.where(tf.greater(x,0),x,0.2*x)

def model_arg_scope(weight_decay=0.0005, ac_fn = tf.nn.relu):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
        activation_fn = ac_fn,
        weights_regularizer= slim.l2_regularizer(weight_decay),
        weights_initializer= tf.truncated_normal_initializer(stddev=0.01),
        biases_initializer= tf.zeros_initializer(),
        normalizer_fn = None):
        with slim.arg_scope([slim.conv2d], padding ='SAME') as sc:
            return sc



def pdf_sample(pdf, uniform_noise):
  pdf = pdf / (tf.reduce_sum(pdf, axis=1, keep_dims=True) + 1e-36)
  cdf = tf.cumsum(pdf, axis=1, exclusive=True)
  indices = tf.reduce_sum(
      tf.cast(tf.less(cdf, uniform_noise), tf.int32), axis=1) - 1
  return indices


def pdf_sample_2d(pdf, uniform_noise):
  height, width = pdf.get_shape()[1], pdf.get_shape()[2]
  pdf = tf.reshape(pdf, (int(pdf.get_shape()[0]), -1))
  indices_1d = pdf_sample(pdf, uniform_noise)
  indices = tf.stack(
      [tf.clip_by_value(indices_1d / width, 0, height - 1), indices_1d % width],
      axis=1)
  return indices


def test1():
  import cv2
  batch_size = 1024
  img = cv2.imread('data/doggy.jpg').mean(axis=2)

  pdf_batch = np.empty(
      shape=(batch_size, img.shape[0], img.shape[1]), dtype=np.float32)

  for i in range(batch_size):
    pdf_batch[i] = img

  pdf = tf.placeholder(tf.float32, (batch_size, img.shape[0], img.shape[1]))
  noise = tf.placeholder(tf.float32, (batch_size, 1))

  with tf.Session() as sess:
    indices = pdf_sample_2d(pdf, noise)
    image_buffer = np.zeros(
        shape=(img.shape[0], img.shape[1]), dtype=np.float32)

    while True:
      indices_out = sess.run(
          indices,
          feed_dict={pdf: pdf_batch,
                     noise: np.random.rand(batch_size, 1)})

      for ind in indices_out:
        image_buffer[ind[0]][ind[1]] += 1

      cv2.imshow('img', image_buffer / np.max(image_buffer))
      cv2.waitKey(30)


def test2():
  batch_size = 1024
  n = 3

  pdf_batch = [[2.0**i for i in range(1, n + 1)] for _ in range(batch_size)]

  pdf = tf.placeholder(tf.float32, (batch_size, n))
  noise = tf.placeholder(tf.float32, (batch_size, 1))

  counter = [0 for _ in range(n)]

  with tf.Session() as sess:
    indices = pdf_sample(pdf_batch, noise)

    for i in range(1000):
      indices_out = sess.run(
          indices,
          feed_dict={pdf: pdf_batch,
                     noise: np.random.rand(batch_size, 1)})
      for i in indices_out:
        counter[indices_out[i]] += 1

    for i in range(n):
      print(counter[i] * 1.0 / 100 / batch_size)


if __name__ == '__main__':
  test2()
  # test()
