#!/usr/bin/env python

import json
import os
# import random
# import scipy.io
import argparse
# import numpy as np
# import code
# import sys
# import subprocess
import shutil
import pickle

def main(params):
    root_path = '/data/chenyou/caffe_SSD/examples/robot/procedures/dpn/data/ADLdataset'
    raw_root_path = root_path + '/ADL_videos'
    image_root_path = root_path + '/ADL_images_flows'
    ann_root_path = root_path + '/ADL_annotations'
    ann_root_path_origin = root_path + '/all_ann'

    data_dict = {}

    test_name_size = root_path + '/test_name_size.txt'

    train_path_cur = root_path + '/trainval.txt'
    test_path_cur = root_path + '/test.txt'
    
    train_path_1 = root_path + '/trainval_base_1s.txt'
    test_path_1 = root_path + '/test_base_1s.txt'
    train_path_5 = root_path + '/trainval_base_5s.txt'
    test_path_5 = root_path + '/test_base_5s.txt'

    test_sets=['P_16','P_17','P_18','P_19','P_20']

    for fn in sorted(os.listdir(ann_root_path_origin)):
        if not fn.endswith('.xml'):
            continue
        #print fn
        segs = fn[:-4].split('_')
        assert len(segs)==4
        assert segs[0]=='P'
        assert len(segs[1])==2
        dname = fn[:4]
        assert segs[2]=='frame'
        fno = int(segs[3])

        if dname not in data_dict:
            data_dict[dname] = {}

        frm_path = os.path.join(image_root_path,dname,  'frame_%05d.jpg'%(fno,))
        flx_path = os.path.join(image_root_path, dname, 'flow_x_%05d.jpg' % (fno,))
        fly_path = os.path.join(image_root_path, dname, 'flow_y_%05d.jpg' % (fno,))

        origin_ann_path = os.path.join(ann_root_path_origin,fn)
        new_ann_path = os.path.join(ann_root_path,'%s_frame_%05d.xml' % (dname,fno))
        shutil.copy2(origin_ann_path, new_ann_path)


        if os.path.isfile(frm_path) and os.path.isfile(flx_path) and os.path.isfile(fly_path):
            data_dict[dname][fno] = ['ADL_images_flows/%s/frame_%05d.jpg'%(dname,fno),'ADL_annotations/%s_frame_%05d.xml' % (dname,fno),
                                     frm_path,new_ann_path,flx_path,fly_path]

    f_size = open(test_name_size,'w')

    f10 = open(train_path_cur,'w')
    f11 = open(train_path_1,'w')
    f12 = open(train_path_5,'w')

    f20 = open(test_path_cur, 'w')
    f21 = open(test_path_1,'w')
    f22 = open(test_path_5, 'w')


    for dname in sorted(data_dict.keys()):
        if dname not in test_sets:
            f0,f1,f2 = f10,f11,f12
        else:
            f0,f1,f2 = f20,f21,f22

        cnt = 0
        print dname,
        for fns in sorted(data_dict[dname].keys()):
            #print fns,
            if fns in data_dict[dname]:
                f_size.write('%s 960 1280\n'% (data_dict[dname][fns][0],) )
                f0.write('%s %s\n' % (data_dict[dname][fns][0], data_dict[dname][fns][1]))
                cnt+=1
            if fns>10 and fns+30 in data_dict[dname]:
                f1.write('%s %s\n' % (data_dict[dname][fns][0], data_dict[dname][fns+30][1]))
            if fns>10 and fns+30*5 in data_dict[dname]:
                f2.write('%s %s\n' % (data_dict[dname][fns][0], data_dict[dname][fns+30*5][1]))
        print cnt

    with open(root_path+'/data_dict.pkl','wb') as f:
        pickle.dump(data_dict,f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # global setup settings, and checkpoints
    # parser.add_argument('data_path', type=str, help='path to video')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
