#!/bin/bash

#VIDEO=/data/chenyou/robot/iccv17/egohands_data/Train/JPEGImages
#FLOW=/data/chenyou/robot/iccv17/egohands_data/Train/Flow

# img_dir=$VIDEO
# flow_dir=$FLOW
# echo "$img_dir-->$flow_dir"
# ./denseFlow_gpu \
#   -f $img_dir/%04d.jpg \
#   -x $flow_dir/flow_x \
#   -y $flow_dir/flow_y \
#   -b 20 -t 1 -d 0 -s 1
  
  
VIDEO=/data/chenyou/robot/iccv17/egohands_data/Test/tmp
FLOW=/data/chenyou/robot/iccv17/egohands_data/Test/tmp_flow

img_dir=$VIDEO
flow_dir=$FLOW
echo "$img_dir-->$flow_dir"
./denseFlow_gpu \
  -f $img_dir/%04d.jpg \
  -x $flow_dir/flow_x \
  -y $flow_dir/flow_y \
  -b 20 -t 1 -d 1 -s 1



