from __future__ import print_function
import sys
sys.path.insert(0, 'python')

import caffe
from model_libs import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import subprocess


future=1   # 1s or 5s prediction
#future=5
#type=1     # 1,5,or 10 frames combination
type=3

wk_root='/data/chenyou/caffe_SSD/examples/robot/procedures/dpn'
data_root=wk_root+'/data/ADLdataset'

### Modify the following parameters accordingly ###
# The directory which contains the caffe code.
# We assume you are running the script at the CAFFE_ROOT.

# Set true if you want to start training right after generating all files.
run_soon = True
# Set true if you want to load from most recently saved snapshot.
# Otherwise, we will load from the pretrain_model defined below.
resume_training = True
# If true, Remove old model files.
remove_old_models = False




# Specify the batch sampler.
#resize_width = 300
#resize_height = 300
resize_width = 25
resize_height = 25
resize = "{}x{}".format(resize_width, resize_height)



# Modify the job name if you want.
job_name = "SSD_regressTwoStream_convnet_%ds_%s" % (future,resize)
# The name of the model. Modify it if you want.
model_name = "adl_{}".format(job_name)

# Directory which stores the model .prototxt file.
save_dir = wk_root+"/models/adl/{}".format(job_name)
# Directory which stores the snapshot of models.
snapshot_dir = wk_root+"/models/adl/{}".format(job_name)
# Directory which stores the job script and log file.
job_dir = wk_root+"/jobs/adl/{}".format(job_name)
# Directory which stores the detection results.
output_result_dir = '%s/results/adl/%s' % (wk_root,job_name)

# model definition files.
suffix = '_con0'
if type==2:
  suffix = '_con5'
elif type==3:
  suffix = '_con10'
train_net_file = "{}/train{}.prototxt".format(save_dir,suffix)
test_net_file = "{}/test{}.prototxt".format(save_dir,suffix)
deploy_net_file = "{}/deploy{}.prototxt".format(save_dir,suffix)
solver_file = "{}/solver{}.prototxt".format(save_dir,suffix)
# snapshot prefix.
snapshot_prefix = "{}/{}{}".format(snapshot_dir, model_name,suffix)
# job script path.
job_file = "{}/{}{}.sh".format(job_dir, model_name,suffix)

if type==1:
  train_list = data_root+"/hdf5s_twoStream_convnet_ssd_%ds/con0_train_list_h5_%ds.txt" % (future,future)
  test_list = data_root+"/hdf5s_twoStream_convnet_ssd_%ds/con0_test_list_h5_%ds.txt" % (future,future)
elif type==2:
  train_list = data_root+"/hdf5s_twoStream_convnet_ssd_%ds/con5_train_list_h5_%ds.txt" % (future,future)
  test_list = data_root+"/hdf5s_twoStream_convnet_ssd_%ds/con5_test_list_h5_%ds.txt" % (future,future)
elif type==3:
  train_list = data_root+"/hdf5s_twoStream_convnet_ssd_%ds/con10_train_list_h5_%ds.txt" % (future,future)
  test_list = data_root+"/hdf5s_twoStream_convnet_ssd_%ds/con10_test_list_h5_%ds.txt" % (future,future)



# Solver parameters.
# Defining which GPUs to use.
gpus = '0' #str((type-1)%3)
gpulist = gpus.split(",")
num_gpus = len(gpulist)

# Divide the mini-batch to different GPUs.
#batch_size = 32
#accum_batch_size = 32
batch_size = 32
accum_batch_size = 32
iter_size = 1

if num_gpus > 0:
  batch_size_per_device = batch_size/num_gpus
  solver_mode = P.Solver.GPU
  device_id = int(gpulist[0])


# Evaluate on whole test set.
num_test_image = 2740
test_batch_size = 32
# Ideally test_batch_size should be divisible by num_test_image,
# otherwise mAP will be slightly off the true value.
test_iter = int(math.ceil(float(num_test_image) / test_batch_size))

solver_param = {
    # Train parameters
    'base_lr': 1e-7 if type==1 else 1e-8,
    'weight_decay': 0.0005,
    'lr_policy': "multistep",
    'stepvalue': [30000, 50000, 80000],
    'gamma': 0.1,
    'momentum': 0.95,
    'iter_size': iter_size,
    'max_iter': 100000,
    'snapshot': 10000,
    'display': 20,
    'average_loss': 20,
    'type': "SGD",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
    'test_iter': [test_iter],
    'test_interval': 5000,
    'test_initialization': True,
    }



### Hopefully you don't need to change the following ###
# Check file.
check_if_exist(train_list)
check_if_exist(test_list)
make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(snapshot_dir)

######################## 
### Create train net ###
######################## 


net = caffe.NetSpec()
AlexNetRegressNetBody(net, batch_size=batch_size, source=train_list, type=type, train=True)
with open(train_net_file, 'w') as f:
    print('name: "{}_train"'.format(model_name), file=f)
    print(net.to_proto(), file=f)
shutil.copy(train_net_file, job_dir)



######################## 
### Create test net  ###
######################## 

net = caffe.NetSpec()
VGGRegressNetBody(net, batch_size=batch_size, source=test_list, type=type, train=False)
with open(test_net_file, 'w') as f:
    print('name: "{}_test"'.format(model_name), file=f)
    print(net.to_proto(), file=f)
shutil.copy(test_net_file, job_dir)


# Create deploy net.
# Remove the first and last layer from test net.
deploy_net = net
with open(deploy_net_file, 'w') as f:
    net_param = deploy_net.to_proto()
    # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
    del net_param.layer[0]
    del net_param.layer[-1]
    
    if type==1: 
      dim0 = 256
    elif type==2: 
      dim0 = 256*5
    elif type==3: 
      dim0 = 256*10
    
    net_param.name = '{}_deploy'.format(model_name)
    net_param.input.extend(['data'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, dim0, resize_height, resize_width])])
    print(net_param, file=f)
shutil.copy(deploy_net_file, job_dir)


# Create solver.
solver = caffe_pb2.SolverParameter(
        train_net=train_net_file,
        test_net=[test_net_file],
        snapshot_prefix=snapshot_prefix,
        **solver_param)

with open(solver_file, 'w') as f:
    print(solver, file=f)
shutil.copy(solver_file, job_dir)


max_iter = 0

train_src_param = ''

if type==1:
  pretrain_model = ''
elif type==2:
  pretrain_model = ''
elif type==3:
  pretrain_model = '' #snapshot_dir+'/adl_SSD_regone_25x25_con10_iter_20000.caffemodel'

train_src_param = '--weights="{}" \\\n'.format(pretrain_model)

if resume_training:
  if max_iter > 0:
    train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)

if remove_old_models:
  # Remove any snapshots smaller than max_iter.
  for file in os.listdir(snapshot_dir):
    if file.endswith(".solverstate"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))
    if file.endswith(".caffemodel"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))

# Create job file.
with open(job_file, 'w') as f:
  f.write('/data/chenyou/caffe_SSD/build/tools/caffe train \\\n')
  f.write('--solver="{}" \\\n'.format(solver_file))
  f.write(train_src_param)
  if solver_param['solver_mode'] == P.Solver.GPU:
    f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(gpus, job_dir, model_name))
  else:
    f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))

# Copy the python script to job_dir.
py_file = os.path.abspath(__file__)
shutil.copy(py_file, job_dir)

# Run the job.
os.chmod(job_file, stat.S_IRWXU)
if run_soon:
  subprocess.call(job_file, shell=True)
