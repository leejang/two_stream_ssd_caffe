import sys

import numpy as np
#import matplotlib.pyplot as plt
import cv2
import glob
import time

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
#caffe.set_mode_cpu()

import h5py

from google.protobuf import text_format
from caffe.proto import caffe_pb2

import re
# load PASCAL VOC labels
labelmap_file = 'data/egohands/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

pat = re.compile("(\d+)\D*$")
def key_func(x):
        mat=pat.search(os.path.split(x)[-1]) # match last group of digits
        if mat is None:
            return x
        return "{:>10}".format(mat.group(1)) # right align to 10 digits.

# SSD 500 with Egohands
model_def = '/home/leejang/lib/ssd_caffe/caffe/models/VGGNet/egohands/SSD_500x500/deploy.prototxt'
model_weights = '/home/leejang/lib/ssd_caffe/caffe/models/VGGNet/egohands/SSD_500x500/egohands_SSD_500x500_iter_50000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# for SSD 500
image_resize = 500
net.blobs['data'].reshape(1,3,image_resize,image_resize)

# input images
images = "/home/leejang/data/recorded_videos_on_0830_2016/scenario1/0105/*.jpg"
num_of_images = len(glob.glob(images))

# HDF5
# hdf5 output file (read and write mode: 'a')
hdf_raw_f_name = '/home/leejang/lib/ssd_caffe/caffe/examples/hand_only_083016_0105_raw.h5'
hdf_raw_f = h5py.File(hdf_raw_f_name, 'a')
hdf_raw_f.create_dataset('data', (num_of_images,4,6), dtype='i')
hdf_raw_f.create_dataset('label', (num_of_images,4,6), dtype='i')

hdf_f_name = '/home/leejang/lib/ssd_caffe/caffe/examples/hand_only_083016_0105.h5'
hdf_f = h5py.File(hdf_f_name, 'a')
hdf_f.create_dataset('data', (num_of_images - 30,4,6), dtype='i')
hdf_f.create_dataset('label', (num_of_images - 30,4,6), dtype='i')

################################
# main loop
#

# Step 0: Extract Features for every frame
idx_cnt = 0
for test_img in sorted(glob.glob(images), key=key_func):

    file_name = os.path.basename(test_img)
    print (file_name)

    # processing time check
    t = time.time()
    image = caffe.io.load_image(test_img)

    # check input image
    img = cv2.imread(test_img)

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    # Forward pass starting from data
    detections = net.forward()['detection_out']

    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = [(255,0,0), (0,255,0), (0,0,255), (255,102,0), (102,0,204), (204,102,0), (255,102,204), (0,255,255)]

    # to store detected results
    new_entry = np.zeros((4,6))

    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = top_labels[i]
        print(" %s: %.2f" %(label_name, score))
        text = ("%s: %.2f" %(label_name, score))
        coords = xmin, ymin, xmax-xmin+1, ymax-ymin+1
        centers = (xmin + xmax)/2, (ymin + ymax)/2
        color = colors[i]

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)
        if ymin < 10:
            cv2.putText(img, text, (xmin, ymin + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
        else:
            cv2.putText(img, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)

        if label_name == 'my_left':
            new_entry[0,:] = centers + coords         
        elif label_name == 'my_right':
            new_entry[1,:] = centers + coords         
        elif label_name == 'your_left':
            new_entry[2,:] = centers + coords         
        else:
            new_entry[3,:] = centers + coords         

    print("Proesssed in {:.3f} seconds.".format(time.time() - t))


    hdf_raw_f['data'][idx_cnt] = new_entry
    hdf_raw_f['label'][idx_cnt] = new_entry

    # write image
    sv_image = '/home/leejang/data/temp/'+str(file_name)+'.result.jpg'
    cv2.imwrite(sv_image, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    idx_cnt += 1

print("========================================")
print("ssd-hand-only-baseline: hand detector -> done!!")
print("========================================")

# Step 1: Adjust data to have correct labels
idx_cnt = 0
for test_img in sorted(glob.glob(images), key=key_func):

    if idx_cnt < (num_of_images - 30):
        file_name = os.path.basename(test_img)
        print (file_name, idx_cnt)

        # Read Raw HDF5 file
        read_data = hdf_raw_f.get('data')[idx_cnt]
        read_label = hdf_raw_f.get('label')[idx_cnt + 30]

        # Write a new HDF5 file
        hdf_f['data'][idx_cnt] = read_data
        hdf_f['label'][idx_cnt] = read_label

    else:
        break

    idx_cnt += 1

print("========================================")
print("ssd-hand-only-baseline: ground truth labels -> done!!")
print("========================================")


# close hdf5 file handlers
hdf_f.close()
hdf_raw_f.close()
