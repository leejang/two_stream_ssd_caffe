#!/bin/bash

# this is for rebuttal IJCAI, Mar20,2018
# to test if our full model is trained with one-stage, just to predict the future

source ../global.sh

redo=1
data_root_dir="/data/chenyou/caffe_SSD/examples/robot/procedures/dpn/data/ADLdataset"
dataset_name="adl"
mapfile="/data/chenyou/caffe_SSD/examples/robot/procedures/dpn/data/ADLdataset/labelmap_voc.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=png --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi

# 
# train=( 'trainval_base_1s' )
# for subset in "${train[@]}"
# do
#   python $CAFFE_ROOT/scripts/create_annoset_flow.py --anno-type=$anno_type \
#               --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim \
#               --resize-width=$width --resize-height=$height --check-label --shuffle \
#               $extra_cmd $data_root_dir $data_root_dir/${subset}.txt --shuffle \
#               $data_root_dir/lmdb_twoStream_ssd/$dataset_name"_"$subset"_"$db examples/$dataset_name
# done
# 
train=( 'test_base_1s' )
for subset in "${train[@]}"
do
  python $CAFFE_ROOT/scripts/create_annoset_flow.py --anno-type=$anno_type \
              --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim \
              --resize-width=$width --resize-height=$height --check-label \
              $extra_cmd $data_root_dir $data_root_dir/${subset}.txt \
              $data_root_dir/lmdb_twoStream_ssd/$dataset_name"_"$subset"_"$db examples/$dataset_name
done


# train=( 'trainval_base_5s' )
# for subset in "${train[@]}"
# do
#   python $CAFFE_ROOT/scripts/create_annoset_flow.py --anno-type=$anno_type \
#               --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim \
#               --resize-width=$width --resize-height=$height --check-label --shuffle \
#               $extra_cmd $data_root_dir $data_root_dir/${subset}.txt --shuffle \
#               $data_root_dir/lmdb_twoStream_ssd/$dataset_name"_"$subset"_"$db examples/$dataset_name
# done
# 
# train=( 'test_base_5s' )
# for subset in "${train[@]}"
# do
#   python $CAFFE_ROOT/scripts/create_annoset_flow.py --anno-type=$anno_type \
#               --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim \
#               --resize-width=$width --resize-height=$height --check-label \
#               $extra_cmd $data_root_dir $data_root_dir/${subset}.txt \
#               $data_root_dir/lmdb_twoStream_ssd/$dataset_name"_"$subset"_"$db examples/$dataset_name
# done