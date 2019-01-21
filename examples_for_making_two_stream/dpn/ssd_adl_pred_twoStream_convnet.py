import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import time
import re
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import random


if __name__ == '__main__':

    future = 1
    future = 5
    #contype = 1
    contype = 3


    if contype==1:
        suffix = 'con0'
        shift = 1
        max_iter = 50000
    if contype == 2:
        suffix = 'con5'
        shift = 5
    elif contype == 3:
        suffix = 'con10'
        shift = 10
        max_iter = 100000


    caffe.set_device(1)
    caffe.set_mode_gpu()

    wk_root = '/data/chenyou/caffe_SSD/examples/robot/procedures/dpn'
    model_root = wk_root + '/models'
    data_root = wk_root + '/data/ADLdataset'
    test_result_dir = wk_root + '/results/adl/test_twoStream_convnet_%ds' % (future,)
    labelmap_file = data_root + "/labelmap_voc.prototxt"

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
        mat = pat.search(os.path.split(x)[-1])  # match last group of digits
        if mat is None:
            return x
        return "{:>10}".format(mat.group(1))  # right align to 10 digits.


    # SSD 500 x 500 with auto encoder
    model_def = model_root + '/adl/SSD_twoStream_convnet_500x500/deploy.prototxt'
    model_weights = model_root + '/adl/SSD_twoStream_convnet_500x500/adl_SSD_twoStream_convnet_500x500_iter_50000.caffemodel'
    assert os.path.isfile(model_def)
    assert os.path.isfile(model_weights)

    reg_model_root = model_root + '/adl/SSD_regressTwoStream_convnet_%ds_25x25' % (future,)
    reg_model_def = reg_model_root + '/deploy_%s.prototxt' % (suffix,)
    reg_model_weights = reg_model_root + '/adl_SSD_regressTwoStream_convnet_%ds_25x25_%s_iter_%d.caffemodel' % (future,suffix,max_iter)
    print reg_model_weights
    # print reg_model_weights
    assert os.path.isfile(reg_model_def)
    assert os.path.isfile(reg_model_weights)

    net = caffe.Net(model_def, model_weights, caffe.TEST)
    reg_net = caffe.Net(reg_model_def, reg_model_weights, caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    print net.blobs['data'].data.shape
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    transformer_flowx = caffe.io.Transformer({'data': net.blobs['flowx'].data.shape})
    transformer_flowx.set_transpose('data', (2, 0, 1))
    transformer_flowx.set_mean('data', np.array([128])) # mean pixel
    transformer_flowx.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]



    index = []
    for j in range(-shift+1,1):
        index.append(j)

    # for SSD 500
    image_resize = 500
    net.blobs['data'].reshape(shift, 3, image_resize, image_resize)
    net.blobs['flowx'].reshape(shift, 1, image_resize, image_resize)
    net.blobs['flowy'].reshape(shift, 1, image_resize, image_resize)

    # layers to extract features
    extract_layer = 'fc_e6'
    if extract_layer not in net.blobs:
        raise TypeError("Invalid layer name: " + extract_layer)



    test_list = data_root + '/' + 'test_base_%ds.txt' % (future,)
    useFiles = open(test_list, 'r').read().splitlines()
    cnt = 0
    useFiles = useFiles[::20]



    data = {}

    for ix,line in enumerate(useFiles):

        print ix,len(useFiles)

        sv_line = sv2_line = ''

        if not line: continue
        line = line.strip()
        segs = line.split()
        assert len(segs)==2
        img_path = data_root + '/' + segs[0]
        ann_path = data_root + '/' + segs[1]
        assert os.path.isfile(img_path)
        assert os.path.isfile(ann_path)
        segs2 = segs[0].split('/')
        assert len(segs2)==3
        fd = segs2[1]
        assert fd.startswith('P_')


        if shift==1:
            image = caffe.io.load_image(img_path)
            net.blobs['data'].data[...] = transformer.preprocess('data', image)
            
            flwx_path = img_path.replace('frame','flow_x')
            flwy_path = img_path.replace('frame','flow_y')
            fx = caffe.io.load_image(flwx_path, color=False)
            fy = caffe.io.load_image(flwy_path, color=False)
            net.blobs['flowx'].data[...] = transformer_flowx.preprocess('data', fx)
            net.blobs['flowy'].data[...] = transformer_flowx.preprocess('data', fy)
            
            detections = net.forward()['detection_out']
            extract_features = net.blobs[extract_layer].data
        else:            
            tmp3 = segs2[2][:-4]
            segs3 = tmp3.split('_')
            assert len(segs3)==2
            img_fno = int(segs3[1])    #frame_00030.jpg
            
            for ix,j in enumerate(index):            
                img_path = data_root+'/'+'/'.join(segs2[:-1]) + '/' + segs3[0] + '_' + '%05d.jpg' % (img_fno+j)   
                image = caffe.io.load_image(img_path)
                net.blobs['data'].data[ix,...] = transformer.preprocess('data', image)
                
                flwx_path = img_path.replace('frame','flow_x')
                flwy_path = img_path.replace('frame','flow_y')
                fx = caffe.io.load_image(flwx_path, color=False)
                fy = caffe.io.load_image(flwy_path, color=False)
                net.blobs['flowx'].data[ix,...] = transformer_flowx.preprocess('data', fx)
                net.blobs['flowy'].data[ix,...] = transformer_flowx.preprocess('data', fy)
                
                
                
            detections = net.forward()['detection_out']
            extract_features = net.blobs[extract_layer].data
            extract_features = np.reshape(extract_features,(1,-1,25,25))



        reg_net.blobs['data'].data[...] = extract_features
        future_features = reg_net.forward()['fc2']

        net.blobs[extract_layer].data[...] = future_features
        detections = net.forward(start='relu_e6', end='detection_out')['detection_out']
        print detections.shape

        sv_line+="# %s %s\n" % (img_path, ann_path)
        sv2_line+="# %s %s\n" % (img_path, ann_path)

        # Parse the outputs.
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        conf_cls = [0.0] * 15
        conf_cls_idx = []
        for i in range(15):
            conf_cls_idx.append([])

        tmpLabel = det_label.tolist()

        for i, lb in enumerate(tmpLabel):
            cf = det_conf[i]
            lb = int(lb)
            if cf > conf_cls[lb]:
                conf_cls[lb] = cf
            conf_cls_idx[lb].append((cf, i))

        for i in range(15):
            sv2_line+='%.4f ' % (conf_cls[i])
            print(" %s: %.3f" % (i, conf_cls[i]))
        sv2_line+='\n'

        # sv_file.write(" {:.3f}\n".format(time.time() - t))

        # Get detections with confidence higher than 0.65.
        # print det_conf,det_label
        # top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.30]
        # top_indices = [conf_cls_idx[i] for i in range(15) if conf_cls[i]>0]


        top_indices = []
        for i in range(15):
            tmp = sorted(conf_cls_idx[i], key=lambda x: x[0])
            # print tmp
            if len(tmp) > 5:
                tmp = tmp[-5:]
            for pr in tmp:
                top_indices.append(pr[1])

        # print top_indices

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(labelmap, top_label_indices)

        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        for i in xrange(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            #print(" %s: %.3f" % (label_name, score))
            sv_line+="%s %.2f " % (label_name, score)
            coords = xmin, ymin, xmax - xmin + 1, ymax - ymin + 1
            centers = (xmin + xmax) / 2, (ymin + ymax) / 2
            sv_line+="%d %d " % (centers)
            sv_line+="%d %d %d %d\n" % (coords)

        if fd not in data:
            data[fd] = []
        data[fd].append([sv_line,sv2_line])






    for fd in sorted(data.keys()):

        sv_image_path = os.path.join(test_result_dir, suffix, fd)
        if not os.path.exists(sv_image_path):
            os.makedirs(sv_image_path)

        # write detection results as a file
        sv_file = open(sv_image_path + '/result.txt', 'w')
        sv_file_all = open(sv_image_path + '/pr.txt', 'w')  # for object presence

        for arr in data[fd]:
            sv_file.write(arr[0])
            sv_file_all.write(arr[1])

        sv_file.close()
        sv_file_all.close()

    print("done!!")
