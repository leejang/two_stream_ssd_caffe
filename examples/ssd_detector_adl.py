import sys

import numpy as np
#import matplotlib.pyplot as plt
import cv2
import glob
import time

#plt.rcParams['figure.figsize'] = (10, 10)
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(1)
caffe.set_mode_gpu()
#caffe.set_mode_cpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

import re

# load ADL labels
labelmap_file = 'data/ADLdataset/labelmap_voc.prototxt'
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
model_def = '/home/leejang/lib/ssd_caffe/caffe/models/VGGNet/adl/SSD_2_500x500/deploy.prototxt'
model_weights = '/home/leejang/lib/ssd_caffe/caffe/models/VGGNet/adl/SSD_2_500x500/adl_SSD_2_500x500_iter_25000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

image_resize = 500
net.blobs['data'].reshape(1,3,image_resize,image_resize)

int_cnt = 1
for test_img in sorted(glob.glob("/fast-data/leejang/ADLdataset/Test/JPEGImages/*.jpg"), key=key_func):

    file_name = os.path.basename(test_img)
    print file_name
    print int_cnt
    t = time.time()

    image = caffe.io.load_image(test_img)

    img = cv2.imread(test_img)

    #t = time.time()
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    # Forward pass.
    detections = net.forward()['detection_out']

    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.65.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.2]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,102,0), (102,0,204), (204,102,0), (255,102,204), (0,255,255)]

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
        color = colors[0]

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)

        if ymin < 30:
            cv2.putText(img, text, (xmin, ymin + 20), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        else:
            cv2.putText(img, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

    print("Proesssed in {:.3f} seconds.".format(time.time() - t))

    if int_cnt > 1000:
        break

    sv_image = '/fast-data/leejang/test_iccv17/'+str(file_name)+'.result.jpg'
    cv2.imwrite(sv_image, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    int_cnt += 1

print("done!!")
