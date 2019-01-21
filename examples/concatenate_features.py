from __future__ import print_function

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
from caffe.model_libs import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import subprocess
import numpy as np
import h5py
import time
import cv2
import re
import glob

pat = re.compile("(\d+)\D*$")
def key_func(x):
        mat=pat.search(os.path.split(x)[-1]) # match last group of digits
        if mat is None:
            return x
        return "{:>10}".format(mat.group(1)) # right align to 10 digits.

### Modify the following parameters accordingly ###
# The directory which contains the caffe code.
# We assume you are running the script at the CAFFE_ROOT.
caffe_root = os.getcwd()


read_dir = '/data/leejang/recorded_videos_on_0920_2016/scenario2/0204/'
save_dir = '/data/leejang/hdf5_files_4_regression_with_egohands_flow/'
save_f_name = 'two_stream_' + '090720_0204'

# input images
images = read_dir + 'frame_*.jpg'

num_of_images = len(glob.glob(images))

# HDF5
# hdf5 output file (read and write mode: 'a')
hdf_raw_f_name = save_dir + save_f_name + '_raw.h5'
hdf_raw_f = h5py.File(hdf_raw_f_name, 'a')

# concatenate feature maps
# 5 feature maps (256 -> 1280)
hdf_f_name = save_dir + save_f_name + '_con5.h5'
hdf_f = h5py.File(hdf_f_name, 'a')
hdf_f.create_dataset('data', (num_of_images - 30,256*5,25,25), dtype='f')
hdf_f.create_dataset('label', (num_of_images - 30,256,25,25), dtype='f')

# 10 feature maps (256 -> 2560)
"""
hdf_f_name = save_dir + save_f_name + '_con10.h5'
hdf_f = h5py.File(hdf_f_name, 'a')
hdf_f.create_dataset('data', (num_of_images - 30,256*10,25,25), dtype='f')
hdf_f.create_dataset('label', (num_of_images - 30,256,25,25), dtype='f')
"""

# Step 1: Adjust data to have correct labels
idx_cnt = 0
for test_img in sorted(glob.glob(images), key=key_func):

    if idx_cnt < (num_of_images - 30):
        file_name = os.path.basename(test_img)
        print (file_name, idx_cnt)

        # processing time check
        t = time.time()

        # Read Raw HDF5 file
        read_data = hdf_raw_f.get('data')[idx_cnt]
        read_label = hdf_raw_f.get('label')[idx_cnt + 30]

        # Write a new HDF5 file
        #hdf_f['data'][idx_cnt] = read_data

        # concatenate five feature maps
        hdf_f['data'][idx_cnt] = np.concatenate((hdf_raw_f.get('data')[idx_cnt],
                                                hdf_raw_f.get('data')[idx_cnt + 1],
                                                hdf_raw_f.get('data')[idx_cnt + 2],
                                                hdf_raw_f.get('data')[idx_cnt + 3],
                                                hdf_raw_f.get('data')[idx_cnt + 4]), axis=0)

        # concatenate ten feature maps
        """
        hdf_f['data'][idx_cnt] = np.concatenate((hdf_raw_f.get('data')[idx_cnt],
                                                hdf_raw_f.get('data')[idx_cnt + 1],
                                                hdf_raw_f.get('data')[idx_cnt + 2],
                                                hdf_raw_f.get('data')[idx_cnt + 3],
                                                hdf_raw_f.get('data')[idx_cnt + 4],
                                                hdf_raw_f.get('data')[idx_cnt + 5],
                                                hdf_raw_f.get('data')[idx_cnt + 6],
                                                hdf_raw_f.get('data')[idx_cnt + 7],
                                                hdf_raw_f.get('data')[idx_cnt + 8],
                                                hdf_raw_f.get('data')[idx_cnt + 9]), axis=0)
        """

        hdf_f['label'][idx_cnt] = read_label
        print("adjust feature vector {:.3f} seconds.".format(time.time() - t))

    else:
        break

    idx_cnt += 1

print("========================================")
print("SSD-EgoHandsFlow: Making ground truth labels -> done!!")
print("========================================")

hdf_f.close()
hdf_raw_f.close()

