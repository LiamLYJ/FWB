from process_data import get_image_pack_fn, ImageRecord
import pickle
import numpy as np
import random
import threading
import cv2
import math

from utils import rotate_and_crop
from condition import AsyncTaskManager
import process_data
from process_data import get_image_pack_fn
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def load_data(folds):
    records = []
    r = ImageRecord('', '', '', '', '')
    for fold in folds:
        fn = get_image_pack_fn(fold)
        print ('Loading image pack', fn)
        # cached
        if fn not in load_data.data:
            with open(fn, 'br') as f:
                load_data.data[fn] = pickle.load(f, encoding = 'UTF-8')
        records += load_data.data[fn]
    return records


load_data.data = {}


# returns a function that takes array(int, 0,..resolution - 1)
def create_lut(f, resolution):
    num_samples = resolution

    lut = np.array(
        [f(x) for x in np.linspace(0, 1, num_samples)], dtype=np.float32)

    return lambda x: np.take(lut, x.astype('int32'))


def augment(ldr, illum):
    angle = (random.random() - 0.5) * FLAGS.AUGMENTATION_ANGLE
    scale = math.exp(random.random() * math.log(
        FLAGS.AUGMENTATION_SCALE_HIGH / FLAGS.AUGMENTATION_SCALE_LOW)) * FLAGS.AUGMENTATION_SCALE_LOW
    s = int(round(min(ldr.shape[:2]) * scale))
    s = min(max(s, 10), min(ldr.shape[:2]))
    start_x = random.randrange(0, ldr.shape[0] - s + 1)
    start_y = random.randrange(0, ldr.shape[1] - s + 1)
    # Left-right flip?
    flip_lr = random.randint(0, 1)
    # Top-down flip?
    flip_td = random.randint(0, 1)
    color_aug = np.zeros(shape=(3, 3))
    for i in range(3):
        color_aug[i, i] = 1 + random.random(
        ) * FLAGS.AUGMENTATION_COLOR - 0.5 * FLAGS.AUGMENTATION_COLOR
        for j in range(3):
            if i != j:
                color_aug[i, j] = (random.random() - 0.5) * FLAGS.AUGMENTATION_COLOR_OFFDIAG

    def crop(img, illumination):
        if img is None:
            return None
        img = img[start_x:start_x + s, start_y:start_y + s]
        img = rotate_and_crop(img, angle)
        img = cv2.resize(img, (FLAGS.img_size, FLAGS.img_size))
        if FLAGS.AUGMENTATION_FLIP_LEFTRIGHT and flip_lr:
            img = img[:, ::-1]
        if FLAGS.AUGMENTATION_FLIP_TOPDOWN and flip_td:
            img = img[::-1, :]

        img = img.astype(np.float32)
        new_illum = np.zeros_like(illumination)
        # RGB -> BGR
        illumination = illumination[::-1]
        for i in range(3):
            for j in range(3):
                new_illum[i] += illumination[j] * color_aug[i, j]
        if FLAGS.AUGMENTATION_COLOR_OFFDIAG > 0:
            # Matrix mul, slower
            new_image = np.zeros_like(img)
            for i in range(3):
                for j in range(3):
                    new_image[:, :, i] += img[:, :, j] * color_aug[i, j]
        else:
            img *= np.array(
                [[[color_aug[0][0], color_aug[1][1], color_aug[2][2]]]],
                dtype=np.float32)
            new_image = img
        new_image = np.clip(new_image, 0, 65535)

        def apply_nonlinearity(image):
            if FLAGS.AUGMENTATION_GAMMA != 0 or FLAGS.USE_CURVE:
                res = 1024
                image = np.clip(image * (res * 1.0 / 65536), 0, res - 1)
                gamma = 1.0 + (random.random() - 0.5) * FLAGS.AUGMENTATION_GAMMA
                if USE_CURVE:
                    curve = get_random_curve()
                else:
                    curve = lambda x: x
                mapping = create_lut(lambda x: curve(x)**gamma * 65535.0, res)
                return mapping(image)
            else:
                return image

        if FLAGS.SPATIALLY_VARIANT:
            split = new_image.shape[1] / 2
            new_image[:, :split] = apply_nonlinearity(new_image[:, :split])
            new_image[:, split:] = apply_nonlinearity(new_image[:, split:])
        else:
            new_image = apply_nonlinearity(new_image)

        new_illum = np.clip(new_illum, 0.01, 100)

        return new_image, new_illum[::-1]

    return crop(ldr, illum)


def split(ldr, illum, ldr_, illum_):
    is_row = random.choice([0,1])
    h,w,c = ldr.shape
    ldr_mix = ldr_.copy()
    if is_row == 1:
        rand_row = random.uniform(0.2,0.8)
        end = round(w*rand_row)
        y, x = 0,end 
        ldr_mix[:,0:end,:] = ldr[:,0:end,:]
    else:
        rand_col = random.uniform(0.2,0.8)
        end = round(h*rand_col)
        y, x = end, 0
        ldr_mix[0:end,:,:] = ldr[0:end,:,:]
    illum_mix = [(y,x), illum, illum_]
    return ldr_mix, illum_mix


class DataProvider:

    def __init__(self, is_training, folds):

        self.cursor = 0
        records = load_data(folds)
        self.is_training = is_training
        self.records = records
        random.shuffle(self.records)
        self.data_count = len(self.records)
        print ('#records:', self.data_count, 'preprocessing...')
        self.preprocess()
        self.batch_size = None
        self.async_task = None

    def preprocess(self):
        images = []
        nrgbs = []
        illums = []
        for i in range(len(self.records)):
            images.append(self.records[i].img)
            nrgbs.append(None)
            illums.append(self.records[i].illum)
        # No same size...
        self.images, self.nrgbs, self.illums = images, nrgbs, np.vstack(illums)

    def set_batch_size(self, batch_size):
        assert self.batch_size is None
        self.batch_size = batch_size

    def shuffle(self):
        ind = range(self.data_count)
        random.shuffle(list(ind))
        images = [self.images[i] for i in ind]
        nrgbs = [self.nrgbs[i] for i in ind]
        illums = [self.illums[i] for i in ind]
        self.images = images
        self.nrgbs = nrgbs
        self.illums = illums

    def get_batch_(self):
        batch_size = self.batch_size
        indices = []
        while len(indices) < batch_size:
            s = min(self.data_count - self.cursor, batch_size - len(indices))
            indices += range(self.cursor, self.cursor + s)
            if self.cursor + s >= self.data_count:
                if self.is_training and FLAGS.DATA_SHUFFLE:
                    self.shuffle()
            self.cursor = (self.cursor + s) % self.data_count

        next_batch = [[], [], []]
        for i in indices:
            ldr, nrgb = self.images[i], self.nrgbs[i]
            illum = self.illums[i]
            # random choise another sample for splition
            i_ = random.choice(indices)
            ldr_, nrgb_ = self.images[i_], self.nrgbs[i_]
            illum_ = self.illums[i_]
            ldr = cv2.resize(ldr, (FLAGS.img_size, FLAGS.img_size))
            ldr_ = cv2.resize(ldr_, (FLAGS.img_size, FLAGS.img_size))
            if self.is_training :
                if FLAGS.AUGMENTATION:
                    ldr, illum = augment(ldr, illum)
                    ldr_, illum_ = augment(ldr_, illum_)
                if FLAGS.SPLITION:
                    ldr, illum = split(ldr, illum, ldr_, illum_)
            nrgb = None
            next_batch[0].append(ldr)
            next_batch[1].append(nrgb)
            next_batch[2].append(illum)

        next_batch = (np.stack(next_batch[0]), np.stack(next_batch[1]),
                      np.vstack(next_batch[2]))
        return next_batch

    def get_batch(self):
        if self.async_task is None:
            self.async_task = AsyncTaskManager(self.get_batch_)
        return self.async_task.get_next()

    def stop(self):
        self.async_task.stop()
