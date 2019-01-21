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
from city_config import test_sets, num_classes


if __name__ == '__main__':

    future = 1
    #future = 3
    contype = 1
    #contype = 3


    #['cologne','zurich','jena']

    if contype==1:
        suffix = 'con0'
        shift = 1
        max_iter = 80000
    if contype == 2:
        suffix = 'con5'
        shift = 5
    elif contype == 3:
        suffix = 'con10'
        shift = 10
        max_iter = 80000


    caffe.set_device(1)
    caffe.set_mode_gpu()

    wk_root = '/data/chenyou/caffe_SSD/examples/robot/procedures/dpn'
    model_root = wk_root + '/models'
    data_root = wk_root + '/data/Cityscapes'
    test_result_dir = wk_root + '/results/cityscape/test_twoStream_%ds' % (future,)
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
    model_def = model_root + '/cityscape/SSD_twoStream_500x500/deploy.prototxt'
    model_weights = model_root + '/cityscape/SSD_twoStream_500x500/cityscape_SSD_twoStream_500x500_iter_38000.caffemodel'
    assert os.path.isfile(model_def)
    assert os.path.isfile(model_weights)

    reg_model_root = model_root + '/cityscape/SSD_regressTwoStream_%ds_25x25' % (future,)
    reg_model_def = reg_model_root + '/deploy_%s.prototxt' % (suffix,)
    reg_model_weights = reg_model_root + '/cityscape_SSD_regressTwoStream_%ds_25x25_%s_iter_%d.caffemodel' % (future,suffix,max_iter)
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



    test_list = data_root + '/' + 'demo_base_%ds.txt' % (future,)
    useFiles = open(test_list, 'r').read().splitlines()
    cnt = 0
    #useFiles = useFiles[::20]


    colors = plt.cm.rainbow(np.linspace(0, 1, 16)).tolist()
    for c in colors:
      for i in range(3):
        c[i]*=255


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
        
        fname = segs2[2][:-4]
        assert fname.startswith('frame_')
        fid = int(fname[6:])
        future_img = 'frame_%05d.jpg' % (fid+future)
        segs3 = [data_root,segs2[0],segs2[1],future_img]
        future_img_path = '/'.join(segs3)
        #print future_img_path
        assert os.path.isfile(future_img_path)
        img0 = cv2.imread(img_path)
        img = cv2.imread(future_img_path)
        

        if shift==1:
            image = caffe.io.load_image(img_path)

            net.blobs['data'].data[...] = transformer.preprocess('data', image)
            
            flwx_path = img_path.replace('frame','flow_x')
            flwy_path = img_path.replace('frame','flow_y')
            fx = caffe.io.load_image(flwx_path, color=False)
            fy = caffe.io.load_image(flwy_path, color=False)
            net.blobs['flowx'].data[...] = transformer_flowx.preprocess('data', fx)
            net.blobs['flowy'].data[...] = transformer_flowx.preprocess('data', fy)
            
            detections = net.forward()['fc_e6']
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
        #print detections.shape


        # Parse the outputs.
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        conf_cls = [0.0] * num_classes
        conf_cls_idx = []
        for i in range(num_classes):
            conf_cls_idx.append([])

        tmpLabel = det_label.tolist()

        for i, lb in enumerate(tmpLabel):
            cf = det_conf[i]
            lb = int(lb)
            if cf > conf_cls[lb]:
                conf_cls[lb] = cf
            conf_cls_idx[lb].append((cf, i))


        top_indices = []
        for i in range(num_classes):
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
            if label_name == 'traffic sign':
                label_name = 'sign'
            elif label_name == 'traffic light':
                label_name = 'light'
            
            coords = xmin, ymin, xmax - xmin + 1, ymax - ymin + 1
            centers = (xmin + xmax) / 2, (ymin + ymax) / 2
            color = colors[label]
            
            if score>0.6:
                #text = ("%s: %.2f" %(label_name, score))
                text = label_name
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

                if ymin < 10:
                    cv2.putText(img, text, (xmin, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    cv2.putText(img, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        # write image
        sv_image_path = os.path.join(test_result_dir, suffix, fd)
        if not os.path.exists(sv_image_path):
            os.makedirs(sv_image_path)

        print(img0.shape)
        h1, w1, c1 = img0.shape
        h2, w2, c2 = img.shape
        assert h1==h2 and w1==w2
        vis = np.zeros((max(h1, h2), w1+w2+10, 3), np.uint8)
        vis[:h1, :w1, :c1] = img0
        vis[:h2, w1+10:w1+w2+10, :c2] = img

        sv_image = sv_image_path + '/' + future_img  
        cv2.imwrite(sv_image, vis, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print img_path,'->', sv_image

    print("done!!")
