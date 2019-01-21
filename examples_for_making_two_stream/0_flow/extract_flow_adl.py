#!/usr/bin/env python

import json
import os
import random
import scipy.io
import argparse
import numpy as np
import code
import sys
import subprocess
import shutil


def main(params):

  f1 = open('run1.sh','w')
  f2 = open('run2.sh','w') 
  f3 = open('run3.sh','w')
    
  root_path = '/data/chenyou/caffe_SSD/examples/robot/procedures/dpn/data/ADLdataset'
  raw_root_path = root_path + '/ADL_videos'
  image_root_path=root_path + '/ADL_images_flows'

  if not os.path.exists(image_root_path):
    os.makedirs(image_root_path)


  total = len(os.listdir(raw_root_path))
  cnt = 0
  for dataset in sorted(os.listdir(raw_root_path)):
    #print dataset
    raw_mp4 = os.path.join(raw_root_path,dataset)
    #print raw_mp4
    if not raw_mp4.endswith('MP4'):
        continue

    image_dir = os.path.join(image_root_path,dataset[:-4])

    if not os.path.exists(image_dir):
      os.makedirs(image_dir)

    #print
    if cnt<total/3:
      gpuid=0
    elif cnt<total*2/3:
      gpuid=1
    else:
      gpuid=2
  
    getFlow = "./denseFlow_gpu -f %s -i %s/frame -x %s/flow_x -y %s/flow_y -b 20 -t 1 -d %d -s 1\n"  \
            % (raw_mp4, image_dir,image_dir,image_dir,gpuid)
    
    cnt += 1
    print getFlow



    if gpuid==0:
        f1.write(getFlow)
    elif gpuid==1:
        f2.write(getFlow)
    else:
        f3.write(getFlow)

  f1.close()
  f2.close() 
  f3.close()
    



if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # global setup settings, and checkpoints
  #parser.add_argument('data_path', type=str, help='path to video')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)

