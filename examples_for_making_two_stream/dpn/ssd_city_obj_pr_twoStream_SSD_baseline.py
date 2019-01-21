import argparse
import os
import sys
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import glob
import time
import re
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2
from city_config import test_sets, num_classes

#future = 1
future = 3
contype = 1
#contype = 3

wk_root = '/data/chenyou/caffe_SSD/examples/robot/procedures/dpn'
model_root = wk_root + '/models'
data_root = wk_root + '/data/Cityscapes'
test_result_dir = wk_root + '/results/cityscape/test_twoStream_SSD_baseline_%ds' % (future,)
labelmap_file = data_root + "/labelmap_voc.prototxt"

# val = 0.1
# val = 0.3
val = 0.5
# test_sets=['dresden','freiburg','heilbronn','mannheim','saarbrucken']
# num_classes = 10

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


def parse_pred_file(fpath):
    trainEntry = open(fpath, 'r').read().splitlines()
    val_dict = {}
    name_dict = {}
    for line in trainEntry:
        if not line: continue
        if line.startswith('#'):
            segs = line.split()
            assert len(segs) == 3
            gt_path = segs[2]
            tmp = os.path.basename(gt_path)
            tmp = tmp[:-4]
            segs2 = tmp.split('_')
            fid = int(segs2[-1])
        else:
            segs = line.strip().split()
            # print segs
            assert len(segs) == num_classes
            val_dict[fid] = segs
            name_dict[fid] = gt_path

    return val_dict, name_dict


if __name__ == '__main__':

    # load PASCAL VOC labels
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)
    # print labelmap


    if contype == 1:
        suffix = 'con0'
    elif contype == 2:
        suffix = 'con5'
    elif contype == 3:
        suffix = 'con10'

    labels = get_labelname(labelmap, range(num_classes))
    #print labels

    indicator = []
    for i in range(num_classes):
        indicator.append({})


    threshold = 0.01


    while threshold <= 1.0:

        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes

        for fd in test_sets:
            pr_path = '%s/%s/%s' % (test_result_dir, suffix, fd)
            pr_txt = pr_path + '/pr.txt'
            if not os.path.isfile(pr_txt):
                continue

            val_dict, name_dict = parse_pred_file(pr_txt)


            for fid in sorted(val_dict.keys()):
                gt_path = name_dict[fid]
                trainEntry = open(gt_path, 'r').read().splitlines()
                str_gt = ''
                for line in trainEntry:
                    str_gt += line

                for i in range(num_classes):

                    glb = 0
                    if labels[i] in str_gt:
                        glb = 1
                    # print  glb, labels[i]
                    prob = float(val_dict[fid][i])

                    # print glb, prob, threshold

                    if prob >= threshold:
                        if glb == 1:
                            tp[i] += 1
                        else:
                            fp[i] += 1
                    else:
                        if glb == 1:
                            fn[i] += 1

        # print tp,fp,fn
        threshold += 1e-2

        for i in range(num_classes):
            if tp[i] + fp[i] == 0: continue
            if tp[i] + fn[i] == 0: continue
            precision = 1.0 * tp[i] / (tp[i] + fp[i])
            recall = 1.0 * tp[i] / (tp[i] + fn[i])
            #print recall,precision
            indicator[i][recall] = precision

            # print indicator
    # calculate meanAP


    meanAP = 0
    area = 0.0
    areas = []
    for i in range(num_classes):
        #indicator[i][0.0] = 1.0
        #print indicator[i]
        area = 0
        prev = None
        pval = None
        for k in sorted(indicator[i]):
            if prev is None:
                prev = k
                pval = indicator[i][k]
            else:
                area += pval * (k - prev)
                prev = k
                pval = indicator[i][k]
        #print round(area*100,2),'&',
        print area
        areas.append(str(round(area*100,1)))
        meanAP += max(0, area)
    meanAP /= (num_classes-1)
    print 'meanAP', meanAP
    areas.append(str(round(meanAP*100,1)))
    
    print '&'.join(areas)

        
    
    
    #import json

    # with open(test_data_root + '/predict/oneStream.json', 'w') as outfile:
    #     json.dump(indicator[1:], outfile)
