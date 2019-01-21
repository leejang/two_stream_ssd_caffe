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

# Use GPUs
caffe.set_device(0)
caffe.set_mode_gpu()

read_dir = '/home/leejang/ros_ws/src/baxter_learning_from_egocentric_video/new_robot_video/0210/'
save_dir = '/fast-data/leejang/hdf5_files_with_new_robot_videos/'
save_f_name = 'two_stream_robot_' + '0210'

# input images
images = read_dir + '*.jpg'

num_of_images = len(glob.glob(images))

# HDF5
# hdf5 output file (read and write mode: 'a')
hdf_raw_f_name = save_dir + save_f_name + '_raw.h5'
hdf_raw_f = h5py.File(hdf_raw_f_name, 'a')
hdf_raw_f.create_dataset('data', (num_of_images,256,25,25), dtype='f')
hdf_raw_f.create_dataset('label', (num_of_images,256,25,25), dtype='f')

# Split daat to avoid size limit of HDF5(should be less than 2GB) for Caffe
hdf_f_name = save_dir + save_f_name + '.h5'
hdf_f = h5py.File(hdf_f_name, 'a')
hdf_f.create_dataset('data', (num_of_images - 30,256,25,25), dtype='f')
hdf_f.create_dataset('label', (num_of_images - 30,256,25,25), dtype='f')

# concatenate feature maps
# 5 feature maps (256 -> 1280)
#hdf_f_name = save_dir + save_f_name + '_con5.h5'
#hdf_f = h5py.File(hdf_f_name, 'a')
#hdf_f.create_dataset('data', (num_of_images - 30,256*5,25,25), dtype='f')
#hdf_f.create_dataset('label', (num_of_images - 30,256,25,25), dtype='f')

# 10 feature maps (256 -> 2560)
#hdf_f_name = save_dir + save_f_name + '_con10.h5'
#hdf_f = h5py.File(hdf_f_name, 'a')
#hdf_f.create_dataset('data', (num_of_images - 30,256*10,25,25), dtype='f')
#hdf_f.create_dataset('label', (num_of_images - 30,256,25,25), dtype='f')

# Caffe model and weights
model_def = '/home/leejang/lib/two_stream_ssd_caffe/caffe/models/VGGNet/egohands_flow/SSD_twoStream_500x500/deploy.prototxt'
model_weights = '/home/leejang/lib/two_stream_ssd_caffe/caffe/models/VGGNet/egohands_flow/SSD_twoStream_500x500/egohands_flow_SSD_twoStream_500x500_iter_50000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# for optical flow
transformer_flow = caffe.io.Transformer({'data': net.blobs['flowx'].data.shape})
transformer_flow.set_transpose('data', (2, 0, 1))
transformer_flow.set_mean('data', np.array([128])) # mean pixel
transformer_flow.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]

# for SSD 500
image_resize = 500
net.blobs['data'].reshape(1,3,image_resize,image_resize)
net.blobs['flowx'].reshape(1, 1, image_resize, image_resize)
net.blobs['flowy'].reshape(1, 1, image_resize, image_resize)

# layers to extract features
extract_layer = 'fc_e6'
if extract_layer not in net.blobs:
    raise TypeError("Invalid layer name: " + extract_layer)


################################
# main loop
#

# Step 0: Extract Features for every frame
idx_cnt = 0
for test_img in sorted(glob.glob(images), key=key_func):

    #print (read_dir)
    #print (hdf_raw_f_name)
    #print (hdf_f_name)

    file_name = os.path.basename(test_img)
    #print (file_name)

    base_name = os.path.splitext(file_name)[0]

    if base_name.isdigit():
      flow_x_f = read_dir + 'flow_x_' + base_name + '.jpg'
      flow_y_f = read_dir +  'flow_y_' + base_name + '.jpg'

      print (test_img)
      #print (flow_x_f)
      #print (flow_y_f)

      # processing time check
      t = time.time()
      image = caffe.io.load_image(test_img)

      transformed_image = transformer.preprocess('data', image)
      net.blobs['data'].data[...] = transformed_image

      # Forward pass starting from data
      detections = net.forward()['detection_out']

      # Extract feature vectors
      # data and label are same at this point
      extract_features = net.blobs[extract_layer].data
      hdf_raw_f['data'][idx_cnt] = extract_features
      hdf_raw_f['label'][idx_cnt] = extract_features

      print("get feature vector {:.3f} seconds.".format(time.time() - t))

      idx_cnt += 1

print("========================================")
print("SSD-EgoHandsFlow: Feature Extraction -> done!!")
print("========================================")

# Step 1: Adjust data to have correct labels
idx_cnt = 0
for test_img in sorted(glob.glob(images), key=key_func):

    file_name = os.path.basename(test_img)
    #print (file_name)

    base_name = os.path.splitext(file_name)[0]

    if base_name.isdigit():
      if idx_cnt < (num_of_images - 30):
        file_name = os.path.basename(test_img)
        print (file_name, idx_cnt)

        # processing time check
        t = time.time()

        # Read Raw HDF5 file
        read_data = hdf_raw_f.get('data')[idx_cnt]
        read_label = hdf_raw_f.get('label')[idx_cnt + 30]

        # Write a new HDF5 file
        hdf_f['data'][idx_cnt] = read_data

        """
        # concatenate five feature maps
        hdf_f['data'][idx_cnt] = np.concatenate((hdf_raw_f.get('data')[idx_cnt],
                                                hdf_raw_f.get('data')[idx_cnt + 1],
                                                hdf_raw_f.get('data')[idx_cnt + 2],
                                                hdf_raw_f.get('data')[idx_cnt + 3],
                                                hdf_raw_f.get('data')[idx_cnt + 4]), axis=0)
        """

        """
        # concatenate ten feature maps
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

