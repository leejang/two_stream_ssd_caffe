#!/bin/bash

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

# train=( 'trainval' 'test' )
# for subset in "${train[@]}"
# do
#   python $CAFFE_ROOT/scripts/create_annoset.py --anno-type=$anno_type \
#               --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim \
#               --resize-width=$width --resize-height=$height --check-label \
#               $extra_cmd $data_root_dir $data_root_dir/${subset}.txt \
#               $data_root_dir/lmdb_data/$dataset_name"_"$subset"_"$db examples/$dataset_name
# done


train=( 'trainval_1s' )
for subset in "${train[@]}"
do
  python $CAFFE_ROOT/scripts/create_annoset.py --anno-type=$anno_type \
              --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim \
              --resize-width=$width --resize-height=$height --check-label --shuffle \
              $extra_cmd $data_root_dir $data_root_dir/${subset}.txt \
              $data_root_dir/lmdb_base/$dataset_name"_"$subset"_"$db examples/$dataset_name"_base"
done

train=( 'test_1s' )
for subset in "${train[@]}"
do
  python $CAFFE_ROOT/scripts/create_annoset.py --anno-type=$anno_type \
              --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim \
              --resize-width=$width --resize-height=$height --check-label \
              $extra_cmd $data_root_dir $data_root_dir/${subset}.txt \
              $data_root_dir/lmdb_base/$dataset_name"_"$subset"_"$db examples/$dataset_name"_base"
done