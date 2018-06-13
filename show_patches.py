import pickle
import sys
import cv2
import os
from data_provider import ImageRecord

import numpy as np
from train import FLAGS

def show_patches():
  from data_provider import DataProvider
  # dp = DataProvider(False, ['s0'])
  dp = DataProvider(True, ['s0'])
  dp.set_batch_size(10)
  while True:
    batch = dp.get_batch()
    imgs = batch[0]
    illums = batch[2]
    for i in range(len(imgs)):
      #img = img / np.mean(img, axis=(0, 1))[None, None, :]
      img = imgs[i] / imgs[i].max()
      illum = illums[i]
      print ('illum: ', illum)
      cv2.imshow("Input", np.power(img, 1 / 2.2))
      cv2.waitKey(0)


if __name__ == '__main__':
  show_patches()
