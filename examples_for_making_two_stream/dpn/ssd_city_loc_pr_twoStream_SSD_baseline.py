import argparse
import os
import sys
import numpy as np
import matplotlib
import json

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

#val = 0.1
#val = 0.3
val = 0.5



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


def parse_json_gt(fpath, ix2label, json_data):
    #lines = open(fpath, 'r').read().splitlines()
    assert fpath in json_data
    gts = {}
    for ins in json_data[fpath]:
        lb_str = ins[0]
        assert lb_str in ix2label
        lb = ix2label[lb_str]
        gts[lb] = ins[1]


    # convert to xmin,ymin,width,height
    for k, v in gts.iteritems():
        assert gts[k][2] > gts[k][0]
        assert gts[k][3] > gts[k][1]
        gts[k][2] -= gts[k][0]
        gts[k][3] -= gts[k][1]
    #print gts

    return gts


def parse_xml_gt(fpath, ix2label):
    lines = open(fpath, 'r').read().splitlines()
    gts = {}
    for line in lines:
        if '<name>' in line:
            assert '</name>' in line
            segs0 = line.split('<name>')
            assert len(segs0) == 2
            segs1 = segs0[1].split('</name>')
            assert len(segs1) == 2
            assert segs1[0] in ix2label
            lb = ix2label[segs1[0]]
            # print lb
            gts[lb] = [0] * 4
        else:
            for ix, tag in enumerate(['xmin', 'ymin', 'xmax', 'ymax']):
                if '<' + tag + '>' in line:
                    assert '</' + tag + '>' in line
                    segs0 = line.split('<' + tag + '>')
                    assert len(segs0) == 2
                    segs1 = segs0[1].split('</' + tag + '>')
                    assert len(segs1) == 2
                    val = int(segs1[0])
                    gts[lb][ix] = val

    # print gts

    # convert to xmin,ymin,width,height
    for k, v in gts.iteritems():
        assert gts[k][2] > gts[k][0]
        assert gts[k][3] > gts[k][1]
        gts[k][2] -= gts[k][0]
        gts[k][3] -= gts[k][1]
    #print gts

    return gts


def parse_pred_file(fd, fpath, ix2label, json_data):
    val_dict = {}
    gt_dict = {}

    num_f = 0  # total number of frame
    lines = open(fpath, 'r').read().splitlines()

    for line in lines:
        if len(line) == 0: continue

        if line.startswith('#'):
            segs = line.split()
            assert len(segs) == 3
            gt_path = segs[2]
            assert os.path.isfile(gt_path)

            # P_16_frame_00060.xml
            tmp = os.path.basename(gt_path)
            tmp = tmp[:-4]
            segs2 = tmp.split('_')
            fid = int(segs2[-1])

            #gt_dict[fid] = parse_xml_gt(gt_path, ix2label)
            gt_dict[fid] = parse_json_gt(gt_path, ix2label, json_data)
            val_dict[fid] = {}

        else:
            #if not os.path.isfile(gt_path): continue
            #print(line)
            segs = line.strip().split()
            # print segs
            if len(segs) == 8:
                lb = ix2label[segs[0]]
            else:
                # "traffic sign" makes len(segs)==9
                lb_str = ' '.join(segs[:2])
                lb = ix2label[lb_str]
                segs = [lb_str] + segs[2:]


            if lb not in val_dict[fid]:
                val_dict[fid][lb] = {}
                val_dict[fid][lb]['score'] = [float(segs[1])]
                # val_dict[fid][lb]['center'] = [int(a) for a in segs[2:4]]
                val_dict[fid][lb]['bbox'] = [[int(a) for a in segs[4:8]]]
                # print val_dict[fid][lb]['bbox'],val_dict[fid][lb]['center']
            else:
                val_dict[fid][lb]['score'].append([float(segs[1])])
                val_dict[fid][lb]['bbox'].append([int(a) for a in segs[4:8]])

    return val_dict, gt_dict


# http://stackoverflow.com/questions/9324339/how-much-do-two-rectangles-overlap
def bboxOverlapRatio(gt_bbox, pr_bbox):
    if 1 == 1:
        xa1 = gt_bbox[0]
        la1 = gt_bbox[2]
        xa2 = xa1 + la1
        ya1 = gt_bbox[1]
        wa1 = gt_bbox[3]
        ya2 = ya1 + wa1
        SA = la1 * wa1

    if 2 == 2:
        xb1 = pr_bbox[0]
        lb1 = pr_bbox[2]
        xb2 = xb1 + lb1
        yb1 = pr_bbox[1]
        wb1 = pr_bbox[3]
        yb2 = yb1 + wb1
        SB = lb1 * wb1

    SI = max(0, min(xa2, xa2) - max(xa1, xb1)) * max(0, min(ya2, yb2) - max(ya1, yb1))
    S = SA + SB - SI
    return 1.0 * SI / S


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
    ix2label = {}
    print labels
    for ix, lb in enumerate(labels):
        ix2label[lb] = ix

    val_dicts = {}
    gt_dicts = {}

    # see 5_cityscape/create_cityscape_annotation_2.py
    annJsonRoot = data_root + '/City_annotations_json'
    with open(annJsonRoot+'/label.json','r') as f:
        json_data = json.load(f)


    for fd in test_sets:
        pr_path = '%s/%s/%s' % (test_result_dir, suffix, fd)
        pr_txt = pr_path + '/result.txt'
        val_dict, gt_dict = parse_pred_file(fd, pr_txt, ix2label, json_data)
        val_dicts[fd] = val_dict
        gt_dicts[fd] = gt_dict


    indicator = []
    for i in range(num_classes):
        indicator.append({})




    threshold = 0.01

    while threshold <= 1.0:

        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes

        for fd in test_sets:

            val_dict, gt_dict = val_dicts[fd], gt_dicts[fd]

            assert len(val_dict) == len(gt_dict)

            for fid in sorted(val_dict.keys()):

                # if fid in val_dict and fid in gt_dict
                pred_boxes = val_dict[fid]
                gts_boxes = gt_dict[fid]

                for i in range(num_classes):
                    if i not in gts_boxes:  # if no gt box
                        if i in pred_boxes:
                            prob = min(pred_boxes[i]['score'])
                            if prob >= threshold:
                                fp[i] += 1
                    else:
                        if i not in pred_boxes:
                            fn[i] += 1
                        else:
                            prob = min(pred_boxes[i]['score'])
                            if prob >= threshold:
                                box1 = gts_boxes[i]
                                found = False
                                for box2 in pred_boxes[i]['bbox']:
                                    score = bboxOverlapRatio(box1, box2)
                                    if score>val:
                                        print score
                                    if score >= val:
                                        found = True
                                        #print box1, box2
                                        break

                                if found:
                                    tp[i] += 1
                                else:
                                    fp[i] += 1
                                    fn[i] += 1
                            else:
                                fn[i] += 1

        #print tp, fp, fn
        threshold += 1e-2

        for i in range(num_classes):
            if tp[i] + fp[i] == 0: continue
            if tp[i] + fn[i] == 0: continue
            print tp[i],tp[i] + fp[i]
            precision = 1.0 * tp[i] / (tp[i] + fp[i])
            recall = 1.0 * tp[i] / (tp[i] + fn[i])
            indicator[i][recall] = precision

            # print indicator
    # calculate meanAP

    meanAP = 0
    area = 0.0
    areas = []
    for i in range(num_classes):
        #indicator[i][0.0]=1.0
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
        print area
        areas.append(str(round(area*100,1)))
        meanAP += max(0, area)
    meanAP /= (num_classes-1)
    print 'meanAP', meanAP
    areas.append(str(round(meanAP*100,1)))
    
    print '&'.join(areas)
    
    
    #import json

    # with open(test_data_root + '/predict_loc/oneStream.json', 'w') as outfile:
    #     json.dump(indicator[1:], outfile)
