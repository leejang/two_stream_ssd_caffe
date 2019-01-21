import sys

import numpy as np
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

from google.protobuf import text_format
from caffe.proto import caffe_pb2

import re
# load PASCAL VOC labels
labelmap_file = 'data/hdi_detection/labelmap_voc.prototxt'

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

# SSD 500 x 500 with auto encoder
model_def = '/home/leejang/lib/ssd_caffe/caffe/models/VGGNet/auto_enc_hdi_detection/SSD_640x360/deploy.prototxt'
model_weights = '/home/leejang/lib/ssd_caffe/caffe/models/VGGNet/auto_enc_hdi_detection/SSD_640x360/auto_enc_hdi_detection_SSD_640x360_iter_50000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# HDI future  regression model
# ten con
reg_model_def = '/home/leejang/lib/ssd_caffe/caffe/models/hdi_future_regression_2/hdi_regression_7cv_2fc_con10_test.prototxt'
reg_model_weights = '/home/leejang/lib/ssd_caffe/caffe/models/hdi_future_regression_2/hdi_regression_e7_iter_100000.caffemodel'

reg_net = caffe.Net(reg_model_def,      # defines the structure of the model
                reg_model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# for SSD 500
#image_resize = 500
#net.blobs['data'].reshape(1,3,image_resize,image_resize)

# write detection results as a file
sv_file = open('/data_b/leejang/Datasets/hdi_gestures/results/fut_detection_w_100000/fut_detection.txt', 'a+')

# layers to extract features
extract_layer = 'fc_e6'
if extract_layer not in net.blobs:
    raise TypeError("Invalid layer name: " + extract_layer)

int_cnt = 1
for test_img in sorted(glob.glob("/data_b/leejang/Datasets/hdi_gestures/height/p5_05_08/*.jpg"), key=key_func):

    # This is because it predicts "future" -- 1 second later
    # given ten images
    #sv_file.write("# %d:\n" %(int_cnt + 21))

    file_name = os.path.basename(test_img)
    print file_name
    #sv_file.write(file_name)

    # processing time check
    t = time.time()

    image = caffe.io.load_image(test_img)
    #print("Load Image in {:.3f} seconds.".format(time.time() - t))

    # to draw prediction results on "future" frames
    future_img = "/data_b/leejang/Datasets/hdi_gestures/height/p5_05_08/p5_05_08_%04d.jpg" % (int_cnt+ 21)
    img = cv2.imread(future_img)

    #t = time.time()
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    #print("pre-process in {:.3f} seconds.".format(time.time() - t))

    #t = time.time()
    # Forward pass.
    detections = net.forward()['detection_out']

    extract_features = net.blobs[extract_layer].data

    #if int_cnt == 0:
    #    print "single"
    if int_cnt == 1:
        con_extract_features = extract_features
    # concatenate five extracted features
    #elif int_cnt < 5:
    # concatenate ten extracted features
    elif int_cnt < 10:
        con_extract_features = np.concatenate((con_extract_features, extract_features), axis=1)
        #print con_extract_features.shape
    else:
        con_extract_features = np.concatenate((con_extract_features, extract_features), axis=1)

        # do regression
        # con
        reg_net.blobs['data'].data[...] = con_extract_features
        # single
        #reg_net.blobs['data'].data[...] = extract_features
        future_features = reg_net.forward()['fc2']
        #print type(future_features)

        # delete the oldest extracted features in concatanated feature maps
        con_extract_features = np.delete(con_extract_features,(range(0,256)),1)

        # do detection with future features    
        net.blobs[extract_layer].data[...] = future_features
        #net.blobs[extract_layer].data[...] = extract_features
        detections = net.forward(start='relu_e6', end='detection_out')['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        #sv_file.write(" {:.3f}\n".format(time.time() - t))

        # Get detections with confidence higher than 0.65.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.65]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        colors = {'selfie': (255,0,0), 'stop': (0,255,0), 'come': (0,0,255), 
                  'go': (255,255,0), 'height': (255,102,0)}

        for i in xrange(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            print(" %s: %.2f" %(label_name, score))
            #sv_file.write("%s %.2f " %(label_name, score))
            text = ("%s: %.2f" %(label_name, score))
            coords = xmin, ymin, xmax-xmin+1, ymax-ymin+1
            centers = (xmin + xmax)/2, (ymin + ymax)/2
            #sv_file.write("%d %d " %(centers))
            #sv_file.write("%d %d %d %d\n" %(coords))

            if i == 0:
              write_file_name = "p5_05_08_%04d.jpg" % (int_cnt+ 21)
              sv_file.write("%s, %s, %s\n" %(write_file_name, label_name, score))

            color = ('%s' %(label_name))

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[color], 3)

            # put text in images
            if ymin < 10:
              cv2.putText(img, text, (xmin, ymin + 20), cv2.FONT_HERSHEY_DUPLEX, 1, colors[color], 2)
            else:
              cv2.putText(img, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_DUPLEX, 1, colors[color], 2)

        print("Proesssed in {:.3f} seconds.".format(time.time() - t))

        # write image
        sv_image = "/data_b/leejang/Datasets/hdi_gestures/results/fut_detection_w_100000/r_p5_05_08_%04d.jpg" % (int_cnt+ 21)
        cv2.imwrite(sv_image, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    int_cnt += 1

sv_file.close()

print("done!!")
