import sys
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
import random
#import json
import pickle

######
# future 1s, 5s
# contype: 1)one frame, 2)five frame, 3)ten frames
######
#future = 1
future = 5
#contype = 1
contype = 3


root_path = '/data/chenyou/caffe_SSD/examples/robot/procedures/dpn/data/ADLdataset'
caffe_root = '/data/chenyou/caffe_SSD'
wk_dir = caffe_root + '/examples/robot/procedures/dpn'
data_dir = wk_dir + '/data/ADLdataset'
model_dir = wk_dir + '/models/adl/SSD_oneStream_500x500'

test_sets = ['P_16', 'P_17', 'P_18', 'P_19', 'P_20']

with open(root_path + '/data_dict.pkl', 'r') as f:
    data_dict = pickle.load(f)

# Use GPUs
caffe.set_device(1)
caffe.set_mode_gpu()

# Caffe model and weights
model_def = model_dir + '/deploy.prototxt'
model_weights = model_dir + '/adl_SSD_oneStream_500x500_iter_32293.caffemodel'

net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)  # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

# image_size
image_resize = 500
net.blobs['data'].reshape(1, 3, image_resize, image_resize)

# layers to extract features
extract_layer = 'fc_e6'
if extract_layer not in net.blobs:
    raise contypeError("Invalid layer name: " + extract_layer)

if contype == 1:
    tList = 'con0_test_%ds.txt'%(future,)
    trainList = 'con0_train_list_h5_%ds.txt'%(future,)
    testList = 'con0_test_list_h5_%ds.txt'%(future,)
    shift = 1
elif contype == 2:
    tList = 'con5_test_%ds.txt'%(future,)
    trainList = 'con5_train_list_h5_%ds.txt'%(future,)
    testList = 'con5_test_list_h5_%ds.txt'%(future,)
    shift = 5
elif contype == 3:
    tList = 'con10_test_%ds.txt'%(future,)
    trainList = 'con10_train_list_h5_%ds.txt'%(future,)
    testList = 'con10_test_list_h5_%ds.txt'%(future,)
    shift = 10

index = []
for j in range(-shift+1,1):
    index.append(j)
index.append(future*30)


#f3 = open(data_dir + '/' + tList, 'w')

num_test = 0

h5_root_path = os.path.join(root_path, 'hdf5s_ssd_%ds' % (future,))
f1 = open(h5_root_path + '/' + trainList, 'w')
f2 = open(h5_root_path + '/' + testList, 'w')


for dname in sorted(data_dict.keys()):
    if contype == 1:
        h5_name = '%s/%s_con0.h5' % (h5_root_path, dname)
    elif contype == 2:
        h5_name = '%s/%s_con5.h5' % (h5_root_path, dname)
    elif contype == 3:
        h5_name = '%s/%s_con10.h5' % (h5_root_path, dname)

    if dname in test_sets:
        f2.write(h5_name + '\n')
    else:
        f1.write(h5_name + '\n')

for dname in sorted(data_dict.keys()):
    
    if dname<'P_14':
        continue
    
    if contype == 1:
        h5_name = '%s/%s_con0.h5' % (h5_root_path, dname)
    elif contype == 2:
        h5_name = '%s/%s_con5.h5' % (h5_root_path, dname)
    elif contype == 3:
        h5_name = '%s/%s_con10.h5' % (h5_root_path, dname)

    cnt = 0
    print dname,

    fnos = []
    for fns in sorted(data_dict[dname].keys()):
        #print fns
        blob = data_dict[dname][fns]
        img_path,flx_path,fly_path = blob[2],blob[4],blob[5]
        assert os.path.isfile(img_path) and os.path.isfile(flx_path) and os.path.isfile(fly_path)
        #print data_dict[dname].keys()
        if fns+future*30 in data_dict[dname].keys():
            fnos.append(fns)

    t = len(fnos)
    step = max(1, int(t / 400))
    fnos = fnos[::step]
    num_image = len(fnos)
    print 'has',t,'reduce to',num_image
    if dname in test_sets:
        num_test+=num_image

    hdf_f = h5py.File(h5_name, 'w')
    if contype == 1:
        hdf_f.create_dataset('data', (num_image, 256, 25, 25), dtype='f')
    elif contype == 2:
        hdf_f.create_dataset('data', (num_image, 256 * 5, 25, 25), dtype='f')
    elif contype == 3:
        hdf_f.create_dataset('data', (num_image, 256 * 10, 25, 25), dtype='f')
    hdf_f.create_dataset('label', (num_image, 256, 25, 25), dtype='f')


    net.blobs['data'].reshape(shift+1,3,500,500)

    for cnt,fns in enumerate(fnos):
        print cnt,'/',len(fnos)
        blob = data_dict[dname][fns+j]
        img_path,flx_path,fly_path = blob[2],blob[4],blob[5]
        print img_path
        # /data/chenyou/caffe_SSD/examples/robot/procedures/dpn/data/ADLdataset/ADL_images_flows/P_01/frame_53775.jpg
        segs = img_path.split('/')
        img_base_name = segs[-1]
        segs2 = img_base_name.split('_')
        assert len(segs2)==2
        img_fno = int(segs2[-1][:-4])
        
        for ix,j in enumerate(index):
            
            img_path_j = '/'.join(segs[:-1]) + '/' + segs2[0] + '_' + '%05d.jpg' % (img_fno+j)
            #print img_fno+j,
            img = caffe.io.load_image(img_path)
            net.blobs['data'].data[ix,...] = transformer.preprocess('data', img)
        #print
        
        # Forward pass starting from data
        net.forward()['detection_out']
        tmp = np.array(net.blobs[extract_layer].data)
        print tmp.shape
        tmp1 = tmp[:-1]
        tmp2 = tmp[-1]
        print tmp.shape,tmp1.shape,tmp2.shape
        
        if contype==1:
            hdf_f['data'][cnt] = tmp1 #np.concatenate(tmp2[:-1], axis=0)
        else:
            hdf_f['data'][cnt] = np.reshape(tmp1,(1,-1,25,25))
        hdf_f['label'][cnt] = tmp2



    hdf_f.close()


print 'num testing image: ', num_test
