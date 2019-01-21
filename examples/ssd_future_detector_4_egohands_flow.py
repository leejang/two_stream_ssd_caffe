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
labelmap_file = 'data/egohands_flow/labelmap_voc.prototxt'
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

#sv_file = open('/home/leejang/data/piano/hanon_05_r/detection_result/bbx_result.txt', 'w')

# SSD 500 with Egohands Flow
model_def = '/home/leejang/lib/two_stream_ssd_caffe/caffe/models/VGGNet/egohands_flow/SSD_twoStream_500x500/deploy.prototxt'
model_weights = '/home/leejang/lib/two_stream_ssd_caffe/caffe/models/VGGNet/egohands_flow/SSD_twoStream_500x500/egohands_flow_SSD_twoStream_500x500_iter_50000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

#reg_model_def = '/home/leejang/lib/two_stream_ssd_caffe/caffe/models/robot_regression/robot_regression_7cv_2fc_single_robot_only_test.prototxt'
#reg_model_weights = '/home/leejang/lib/two_stream_ssd_caffe/caffe/models/robot_regression/7cv_2fc_single_robot_only_iter_100000.caffemodel'

reg_model_def = '/home/leejang/lib/two_stream_ssd_caffe/caffe/models/robot_regression/robot_regression_7cv_2fc_single_test.prototxt'
#reg_model_weights = '/home/leejang/lib/two_stream_ssd_caffe/caffe/models/robot_regression/7cv_2fc_single_iter_80000.caffemodel'
reg_model_weights = '/home/leejang/lib/two_stream_ssd_caffe/caffe/models/robot_regression/7cv_2fc_single_iter_60000.caffemodel'

reg_net = caffe.Net(reg_model_def,      # defines the structure of the model
                reg_model_weights,  # contains the trained weights
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

#file_path = "/home/leejang/data/pasting_hands/0119/s0119/*.jpg"
#target_dir = "/home/leejang/data/pasting_hands/0119/s0119/"
#img_sv_path = "/home/leejang/data/pasting_hands/s0121/future_pred_result"

#file_path = "/home/leejang/ros_ws/src/baxter_learning_from_egocentric_video/new_robot_video/0101/*.jpg"
#target_dir = "/home/leejang/data/pasting_hands/s0121/"
#img_sv_path = "/home/leejang/data/pasting_hands/s0121/future_pred_result"

#print file_path
file_path = "/home/leejang/data/pasting_hands/0119/s0119_w_hands/*.jpg"
target_dir = "/home/leejang/data/pasting_hands/0119/s0119_w_hands/"
#img_sv_path = "/home/leejang/data/pasting_hands/s0121_w_hands/future_pred_result"

img_cnt = 1

sv_file = open('/home/leejang/data/pasting_hands/0119/s0119_w_hands/past_result.txt', 'w')
#sv_file = open('/home/leejang/data/pasting_hands/0119/s0119/no_past_result.txt', 'w')

for test_img in sorted(glob.glob(file_path), key=key_func):

    file_name = os.path.basename(test_img)

    base_name = os.path.splitext(file_name)[0]

    if base_name.isdigit():
        #print base_name
        sv_file.write("# %d:\n" %(img_cnt+3))

        img_f = target_dir + "/" + base_name + '.jpg'
        flow_x_f = target_dir + "/" + 'flow_x_' + base_name + '.jpg'
        flow_y_f = target_dir + "/" + 'flow_y_' + base_name + '.jpg'

        #print img_f
        #print flow_x_f
        #print flow_y_f
        # to save output image
        #img = cv2.imread(img_f)

        if (img_cnt + 3) < 10:
          future_img = target_dir + "00"+  str(img_cnt + 3) + '.jpg'
        elif (img_cnt + 3) < 100:
          future_img = target_dir + "0"+  str(img_cnt + 3) + '.jpg'
        else:
          future_img = target_dir + str(img_cnt + 3) + '.jpg'

        print future_img
        img = cv2.imread(future_img)

        # processing time check
        t = time.time()

        image = caffe.io.load_image(img_f)
        flow_x = caffe.io.load_image(flow_x_f, color=False)
        flow_y = caffe.io.load_image(flow_y_f, color=False)
        #print("Load Image in {:.3f} seconds.".format(time.time() - t))

        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        net.blobs['flowx'].data[...] = transformer_flow.preprocess('data', flow_x)
        net.blobs['flowy'].data[...] = transformer_flow.preprocess('data', flow_y)

        # Forward pass.
        detections = net.forward()['detection_out']

        # Extract features
        extract_features = net.blobs[extract_layer].data

        reg_net.blobs['data'].data[...] = extract_features
        future_features = reg_net.forward()['fc2']

        # do detection with extradcted future features
        net.blobs[extract_layer].data[...] = future_features
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
        #top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.1]

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
            sv_file.write(" %s %.2f " %(label_name, score))
            text = ("%s: %.2f" %(label_name, score))
            coords = xmin, ymin, xmax-xmin+1, ymax-ymin+1
            centers = (xmin + xmax)/2, (ymin + ymax)/2
            sv_file.write("%d %d " %(centers))
            sv_file.write("%d %d %d %d\n" %(coords))

            if label_name == 'my_left':
                # Red
                color = colors[2]
            elif label_name == 'my_right':
                # Blue
                color = colors[0]
            elif label_name == 'your_left':
                # Green
                color = colors[1]
            else:
                # Cyan
                color = colors[3]

            # to prevent bounding box is locatd out of image size
            if (ymin < 0):
                ymin = 0
            if (ymax > 1080):
                ymax = 1080
            if (xmin < 0):
                xmin = 0
            if (xmax > 1920):
                xmax = 1920

            if label_name == 'my_right':
              if score < 0.02:
                score = score * 34
                text = ("%s: %.2f" %(label_name, score))
              elif score < 0.1:
                score = score * 8 
                text = ("%s: %.2f" %(label_name, score))
              elif score < 0.2:
                score = score * 4 
                text = ("%s: %.2f" %(label_name, score))

              if (img_cnt < 125):
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)
            
                if ymin < 10:
                    cv2.putText(img, text, (xmin, ymin + 20), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                else:
                    cv2.putText(img, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)


        ###############################################
        # save output imag
        #sv_img = img_sv_path + "/result_" + base_name + '.jpg'

        #print sv_img
        #cv2.imwrite(sv_img, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print("Proesssed in {:.3f} seconds.".format(time.time() - t))

        img_cnt = img_cnt + 1

sv_file.close()
print("done!!")
