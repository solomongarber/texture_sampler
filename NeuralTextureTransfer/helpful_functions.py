import constants
import os
import sys
import glob
import timeit
import cv2
import numpy
import random

def load_data():
    print ('Loading png images from ' + constants.IMAGE_DIRECTORY)
    labels, imgs = [], []
    dct = {}
    for category in constants.CATEGORIES:
        fnames = glob.glob(constants.IMAGE_DIRECTORY + category + '*.png')
        for fname in fnames:
            label = category
            if label not in dct.keys():
                ln = len(dct.keys())
                dct[label] = ln + 1
            idx = dct[label]

            im = cv2.imread(fname)
            imgs.append(im)
            labels.append(idx)

    imgs = numpy.array(imgs)
    labels = numpy.array(labels)
    return imgs, labels

def get_num_images(directory):
    total = 0
    for category in constants.CATEGORIES:
        fnames = glob.glob(directory + category + '*.png')
        total += len(fnames)
    return total

def get_num_categories():
    return len(constants.CATEGORIES)