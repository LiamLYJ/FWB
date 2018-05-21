import cv2
import pickle
import os
import numpy as np
import sys
import random
import scipy.io
from glob import glob
import csv


SHOW_IMAGES = False
# we only use one split folder
FOLDS = 1
DATA_FRAGMENT = -1


def get_image_pack_fn(key):
    ds = key[0]
    if ds == 's':
        fold = int(key[1])
        return SonyDataSet().get_image_pack_fn(fold)
    if ds == 'd':
        fold = int(key[1])
        return DjiDataSet().get_image_pack_fn(fold)
    elif ds == 'm':
        assert False


class ImageRecord:
    def __init__(self, dataset, fn, illum, mcc_coord, img, extras=None):
        self.dataset = dataset
        self.fn = fn
        self.illum = illum
        self.mcc_coord = mcc_coord
        # BRG images
        self.img = img
        self.extras = extras

    def __repr__(self):
        return '[%s, %s, (%f, %f, %f)]' % (self.dataset, self.fn, self.illum[0],
                                           self.illum[1], self.illum[2])


class DataSet:

    def get_subset_name(self):
        return ''

    def get_directory(self):
        return 'data/' + self.get_name() + '/'

    def get_img_directory(self):
        return 'data/' + self.get_name() + '/'

    def get_meta_data_fn(self):
        return self.get_directory() + self.get_subset_name() + 'meta.pkl'

    def dump_meta_data(self, meta_data):
        print ('Dumping data =>', self.get_meta_data_fn())
        print ('  Total records:', sum(map(len, meta_data)))
        print ('  Slices:', map(len, meta_data))
        with open(self.get_meta_data_fn(), 'wb') as f:
            pickle.dump(meta_data, f, protocol=-1)
        print ('Dumped.')

    def load_meta_data(self):
        with open(self.get_meta_data_fn(),'rb') as f:
            return pickle.load(f)

    def get_image_pack_fn(self, fold):
        return self.get_directory() + self.get_subset_name(
        ) + 'image_pack.%d.pkl' % fold

    def dump_image_pack(self, image_pack, fold):
        with open(self.get_image_pack_fn(fold), 'wb') as f:
            pickle.dump(image_pack, f, protocol=-1)

    def load_image_pack(self, fold):
        with open(self.get_meta_data_fn()) as f:
            return pickle.load(f)

    def regenerate_image_pack(self, meta_data, fold):
        image_pack = []
        for i, r in enumerate(meta_data):
            print ('Processing %d/%d\r' % (i + 1, len(meta_data)),)
            sys.stdout.flush()
            r.img = self.load_image_without_mcc(r)

            if SHOW_IMAGES:
                cv2.imshow('img',
                           cv2.resize(
                               np.power(r.img / 65535., 1.0 / 3.2), (0, 0),
                               fx=0.25,
                               fy=0.25))
                il = r.illum
                print ('illum: ', il)
                cv2.waitKey(0)

            image_pack.append(r)
        print (self.dump_image_pack(image_pack, fold))

    def regenerate_image_packs(self):
        meta_data = self.load_meta_data()
        print ('Dumping image packs...')
        print ('%s folds found' % len(meta_data))
        for f, m in enumerate(meta_data):
            self.regenerate_image_pack(m, f)

    def get_folds(self):
        return FOLDS


class DjiDataSet(DataSet):

    def get_name(self):
        return 'dji'

    def regenerate_meta_data(self):
        meta_data_train = []
        meta_data_test = []
        print ("Loading and shuffle fn_and_illum[]")
        ground_truth = scipy.io.loadmat(self.get_directory() + 'ground_truth.mat')[
            'real_rgb']
        ground_truth /= np.linalg.norm(ground_truth, axis=1)[..., np.newaxis]
        filenames = sorted(os.listdir(self.get_directory() + 'images'))
        assert len(filenames) == len(ground_truth)
        #print filenames
        mcc_coord = None

        for i in range(len(filenames)):
            # if i >29:
            #     break
            fn = filenames[i]
            if i < 50:
                meta_data_test.append(
                    ImageRecord(
                        dataset=self.get_name(),
                        fn=fn,
                        illum=ground_truth[i],
                        mcc_coord=mcc_coord,
                        img=None))
            else:
                meta_data_train.append(
                    ImageRecord(
                        dataset=self.get_name(),
                        fn=fn,
                        illum=ground_truth[i],
                        mcc_coord=mcc_coord,
                        img=None))

        #for have two folder
        meta_data_folds = [[],[]]
        num_data_test = len(meta_data_test)
        num_data_train = len(meta_data_train)
        fold_train = np.random.permutation(np.arange(num_data_train))
        assert (len(filenames) == (len(meta_data_train) + len(meta_data_test)))
        fold_test = np.arange(num_data_test)
        for i in fold_train:
            meta_data_folds[0].append(meta_data_train[i])
        for i in fold_test:
            meta_data_folds[1].append(meta_data_test[i])
        print (sum(map(len, meta_data_folds)))
        self.dump_meta_data(meta_data_folds)



    def load_image(self, fn):
        file_path = self.get_img_directory() + '/images/' + fn
        raw = np.array(cv2.imread(file_path, -1), dtype='float32')
        return raw

    def load_image_without_mcc(self, r):
        raw = self.load_image(r.fn)
        img = (np.clip(raw / raw.max(), 0, 1) * 65535.0).astype(np.uint16)
        return img


    def re_make_gt(self, save_folder = './data/dji/',
                   source_folder = '../ffcc/data/DJI/preprocessed',
                   file_extend = '*.txt'):
        file_list = [os.path.basename(x) for x in glob(os.path.join(source_folder, file_extend))]
        print (len(file_list))
        file_list.sort()
        gts = []
        for file in file_list:
            with open(os.path.join(source_folder,file), 'r') as txt:
                content = txt.readlines()
                r = float(content[0])
                g = float(content[1])
                b = float(content[2])
            # I use the ground_truth for FFCC, FFCC use the groud-truth with iverse
            gt = [g/r, g/g ,g/b]
            gts.append(gt)

        real_rgb = {}
        real_rgb['real_rgb'] = np.array(gts)
        scipy.io.savemat(save_folder+'ground_truth.mat', real_rgb)




class SonyDataSet(DataSet):

    def get_name(self):
        return 'sony'

    def regenerate_meta_data(self):
        meta_data_train = []
        meta_data_test = []
        print ("Loading and shuffle fn_and_illum[]")
        ground_truth = scipy.io.loadmat(self.get_directory() + 'ground_truth.mat')[
            'real_rgb']
        ground_truth /= np.linalg.norm(ground_truth, axis=1)[..., np.newaxis]
        filenames = sorted(os.listdir(self.get_directory() + 'images'))
        assert len(filenames) == len(ground_truth)
        #print filenames
        mcc_coord = None

        for i in range(len(filenames)):
            # if i >29:
            #     break
            fn = filenames[i]
            if i < 50:
                meta_data_test.append(
                    ImageRecord(
                        dataset=self.get_name(),
                        fn=fn,
                        illum=ground_truth[i],
                        mcc_coord=mcc_coord,
                        img=None))
            else:
                meta_data_train.append(
                    ImageRecord(
                        dataset=self.get_name(),
                        fn=fn,
                        illum=ground_truth[i],
                        mcc_coord=mcc_coord,
                        img=None))

        #for have two folder
        meta_data_folds = [[],[]]
        num_data_test = len(meta_data_test)
        num_data_train = len(meta_data_train)
        assert (len(filenames) == (len(meta_data_train) + len(meta_data_test)))
        fold_train = np.random.permutation(np.arange(num_data_train))
        fold_test = np.arange(num_data_test)
        for i in fold_train:
            meta_data_folds[0].append(meta_data_train[i])
        for i in fold_test:
            meta_data_folds[1].append(meta_data_test[i])
        print (sum(map(len, meta_data_folds)))
        self.dump_meta_data(meta_data_folds)


    def load_image(self, fn):
        file_path = self.get_img_directory() + '/images/' + fn
        raw = np.array(cv2.imread(file_path, -1), dtype='float32')
        return raw

    def load_image_without_mcc(self, r):
        raw = self.load_image(r.fn)
        img = (np.clip(raw / raw.max(), 0, 1) * 65535.0).astype(np.uint16)
        return img


    def re_make_gt(self, save_folder = './data/sony/',
                   source_folder = '../ffcc/data/Sony/',
                   file_extend = '*_gt.csv'):
        file_list = [os.path.basename(x) for x in glob(os.path.join(source_folder, file_extend))]
        print (len(file_list))
        file_list.sort()
        gts = []
        for file in file_list:
            results = np.ones(6)
            with open(os.path.join(source_folder,file), 'r') as csv_file:
                reader = csv.reader(csv_file)
                for index, row in enumerate(reader):
                    results[index] = float(row[0])
            WB_RGB_level = [float(results[0]), float(results[1]), float(results[2])]
            gts.append(WB_RGB_level)
        real_rgb = {}
        real_rgb['real_rgb'] = np.array(gts)
        scipy.io.savemat(save_folder+'ground_truth.mat', real_rgb)


if __name__ == '__main__':
    sony = SonyDataSet()
    sony.re_make_gt()
    sony.regenerate_meta_data()
    sony.regenerate_image_packs()

    # dji = DjiDataSet()
    # dji.re_make_gt()
    # dji.regenerate_meta_data()
    # dji.regenerate_image_packs()
