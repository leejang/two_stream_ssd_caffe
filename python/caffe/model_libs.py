import os
import sys
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

sys.path.append("pycaffe")


def check_if_exist(path):
    return os.path.exists(path)


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def UnpackVariable(var, num):
    assert len > 0
    if type(var) is list and len(var) == num:
        return var
    else:
        ret = []
        if type(var) is list:
            assert len(var) == 1
            for i in xrange(0, num):
                ret.append(var[0])
        else:
            for i in xrange(0, num):
                ret.append(var)
        return ret


def conv_relu_share(name_w, name_b, bottom, nout, ks=3, stride=1, pad=1):
    w_filler = dict(type='xavier')
    b_filler = dict(type='constant', value=0.01)
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, weight_filler=w_filler, bias_filler=b_filler,
                         param=[dict(name=name_w, lr_mult=1, decay_mult=1),
                                dict(name=name_b, lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)


def conv_relu_dilation_share(name_w, name_b, bottom, nout, ks=3, stride=1, pad=1, dilation=1):
    w_filler = dict(type='xavier')
    b_filler = dict(type='constant', value=0.01)
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, dilation=dilation,
                         num_output=nout, pad=pad, weight_filler=w_filler, bias_filler=b_filler,
                         param=[dict(name=name_w, lr_mult=1, decay_mult=1),
                                dict(name=name_b, lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)


def ConvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
                kernel_size, pad, stride, dilation=1, use_scale=True, lr_mult=1,
                conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
                scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',
                **bn_params):
    if use_bn:
        # parameters for convolution layer with batchnorm.
        kwargs = {
            'param': [dict(lr_mult=lr_mult, decay_mult=1)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_term': False,
        }
        eps = bn_params.get('eps', 0.001)
        moving_average_fraction = bn_params.get('moving_average_fraction', 0.999)
        use_global_stats = bn_params.get('use_global_stats', False)
        # parameters for batchnorm layer.
        bn_kwargs = {
            'param': [
                dict(lr_mult=0, decay_mult=0),
                dict(lr_mult=0, decay_mult=0),
                dict(lr_mult=0, decay_mult=0)],
            'eps': eps,
            'moving_average_fraction': moving_average_fraction,
        }
        bn_lr_mult = lr_mult
        if use_global_stats:
            # only specify if use_global_stats is explicitly provided;
            # otherwise, use_global_stats_ = this->phase_ == TEST;
            bn_kwargs = {
                'param': [
                    dict(lr_mult=0, decay_mult=0),
                    dict(lr_mult=0, decay_mult=0),
                    dict(lr_mult=0, decay_mult=0)],
                'eps': eps,
                'use_global_stats': use_global_stats,
            }
            # not updating scale/bias parameters
            bn_lr_mult = 0
        # parameters for scale bias layer after batchnorm.
        if use_scale:
            sb_kwargs = {
                'bias_term': True,
                'param': [
                    dict(lr_mult=bn_lr_mult, decay_mult=0),
                    dict(lr_mult=bn_lr_mult, decay_mult=0)],
                'filler': dict(type='constant', value=1.0),
                'bias_filler': dict(type='constant', value=0.0),
            }
        else:
            bias_kwargs = {
                'param': [dict(lr_mult=bn_lr_mult, decay_mult=0)],
                'filler': dict(type='constant', value=0.0),
            }
    else:
        kwargs = {
            'param': [
                dict(lr_mult=lr_mult, decay_mult=1),
                dict(lr_mult=2 * lr_mult, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)
        }

    conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
    [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
    [pad_h, pad_w] = UnpackVariable(pad, 2)
    [stride_h, stride_w] = UnpackVariable(stride, 2)
    if kernel_h == kernel_w:
        net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
                                       kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
    else:
        net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
                                       kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
                                       stride_h=stride_h, stride_w=stride_w, **kwargs)
    if dilation > 1:
        net.update(conv_name, {'dilation': dilation})
    if use_bn:
        bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
        net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
        if use_scale:
            sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
            net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
        else:
            bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
            net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
    if use_relu:
        relu_name = '{}_relu'.format(conv_name)
        net[relu_name] = L.ReLU(net[conv_name], in_place=True)


def ResBody(net, from_layer, block_name, out2a, out2b, out2c, stride, use_branch1, dilation=1, **bn_param):
    # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

    conv_prefix = 'res{}_'.format(block_name)
    conv_postfix = ''
    bn_prefix = 'bn{}_'.format(block_name)
    bn_postfix = ''
    scale_prefix = 'scale{}_'.format(block_name)
    scale_postfix = ''
    use_scale = True

    if use_branch1:
        branch_name = 'branch1'
        ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
                    num_output=out2c, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
                    conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                    bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
        branch1 = '{}{}'.format(conv_prefix, branch_name)
    else:
        branch1 = from_layer

    branch_name = 'branch2a'
    ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
                num_output=out2a, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = 'branch2b'
    if dilation == 1:
        ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
                    num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
                    conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                    bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    else:
        pad = int((3 + (dilation - 1) * 2) - 1) / 2
        ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
                    num_output=out2b, kernel_size=3, pad=pad, stride=1, use_scale=use_scale,
                    dilation=dilation, conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                    bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = 'branch2c'
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
                num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    branch2 = '{}{}'.format(conv_prefix, branch_name)

    res_name = 'res{}'.format(block_name)
    net[res_name] = L.Eltwise(net[branch1], net[branch2])
    relu_name = '{}_relu'.format(res_name)
    net[relu_name] = L.ReLU(net[res_name], in_place=True)


def InceptionTower(net, from_layer, tower_name, layer_params, **bn_param):
    use_scale = False
    for param in layer_params:
        tower_layer = '{}/{}'.format(tower_name, param['name'])
        del param['name']
        if 'pool' in tower_layer:
            net[tower_layer] = L.Pooling(net[from_layer], **param)
        else:
            param.update(bn_param)
            ConvBNLayer(net, from_layer, tower_layer, use_bn=True, use_relu=True,
                        use_scale=use_scale, **param)
        from_layer = tower_layer
    return net[from_layer]


def CreateAnnotatedDataLayer(source, batch_size=32, backend=P.Data.LMDB,
                             output_label=True, train=True, label_map_file='', anno_type=None,
                             transform_param={}, batch_sampler=[{}]):
    if train:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
            'transform_param': transform_param,
        }
    else:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
            'transform_param': transform_param,
        }
    ntop = 1
    if output_label:
        ntop = 2
    annotated_data_param = {
        'label_map_file': label_map_file,
        'batch_sampler': batch_sampler,
    }
    if anno_type is not None:
        annotated_data_param.update({'anno_type': anno_type})
    return L.AnnotatedData(name="data", annotated_data_param=annotated_data_param,
                           data_param=dict(batch_size=batch_size, backend=backend, source=source),
                           ntop=ntop, **kwargs)


################
## Input Data ##
################
def CreateAnnotatedDataFlowLayer(source, batch_size=32, backend=P.Data.LMDB,
                                 output_label=True, train=True, label_map_file='', anno_type=None,
                                 transform_param={}, batch_sampler=[{}]):
    if train:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
            'transform_param': transform_param,
        }
    else:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
            'transform_param': transform_param,
        }
    ntop = 4

    annotated_data_param = {
        'label_map_file': label_map_file,
        'batch_sampler': batch_sampler,
    }
    if anno_type is not None:
        annotated_data_param.update({'anno_type': anno_type})
    return L.AnnotatedDataFlow(name="data", annotated_data_param=annotated_data_param,
                               data_param=dict(batch_size=batch_size, backend=backend, source=source),
                               ntop=ntop, **kwargs)


def VGGNetBody(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
               dilated=False, nopool=False, dropout=True, freeze_layers=[], dilate_pool4=False):
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    net.conv1_1 = L.Convolution(net[from_layer], num_output=64, pad=1, kernel_size=3, **kwargs)

    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
    net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)

    if nopool:
        name = 'conv1_3'
        net[name] = L.Convolution(net.relu1_2, num_output=64, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool1'
        net.pool1 = L.Pooling(net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2_1 = L.Convolution(net[name], num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
    net.conv2_2 = L.Convolution(net.relu2_1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)

    if nopool:
        name = 'conv2_3'
        net[name] = L.Convolution(net.relu2_2, num_output=128, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool2'
        net[name] = L.Pooling(net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv3_1 = L.Convolution(net[name], num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
    net.conv3_3 = L.Convolution(net.relu3_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_3 = L.ReLU(net.conv3_3, in_place=True)

    if nopool:
        name = 'conv3_4'
        net[name] = L.Convolution(net.relu3_3, num_output=256, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool3'
        net[name] = L.Pooling(net.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv4_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(net.relu4_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    net.conv4_3 = L.Convolution(net.relu4_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_3 = L.ReLU(net.conv4_3, in_place=True)

    if nopool:
        name = 'conv4_4'
        net[name] = L.Convolution(net.relu4_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool4'
        if dilate_pool4:
            net[name] = L.Pooling(net.relu4_3, pool=P.Pooling.MAX, kernel_size=3, stride=1, pad=1)
            dilation = 2
        else:
            net[name] = L.Pooling(net.relu4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)
            dilation = 1

    kernel_size = 3
    pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
    net.conv5_1 = L.Convolution(net[name], num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation,
                                **kwargs)
    net.relu5_1 = L.ReLU(net.conv5_1, in_place=True)
    net.conv5_2 = L.Convolution(net.relu5_1, num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation,
                                **kwargs)
    net.relu5_2 = L.ReLU(net.conv5_2, in_place=True)
    net.conv5_3 = L.Convolution(net.relu5_2, num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation,
                                **kwargs)
    net.relu5_3 = L.ReLU(net.conv5_3, in_place=True)

    if need_fc:
        if dilated:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=1, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    dilation = dilation * 6
                    kernel_size = 3
                    num_output = 1024
                else:
                    dilation = dilation * 2
                    kernel_size = 7
                    num_output = 4096
            else:
                if reduced:
                    dilation = dilation * 3
                    kernel_size = 3
                    num_output = 1024
                else:
                    kernel_size = 7
                    num_output = 4096
            pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
            net.fc6 = L.Convolution(net[name], num_output=num_output, pad=pad, kernel_size=kernel_size,
                                    dilation=dilation, **kwargs)

            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc7 = L.Convolution(net.relu6, num_output=4096, kernel_size=1, **kwargs)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
        else:
            net.fc6 = L.InnerProduct(net.pool5, num_output=4096)
            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)
            net.fc7 = L.InnerProduct(net.relu6, num_output=4096)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)

    # Update freeze layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net


def VGGAutoEncoderNetBody(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
                          dilated=False, nopool=False, dropout=True, freeze_layers=[], dilate_pool7=False):
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)}

    dkwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]}

    assert from_layer in net.keys()
    net.conv1_1 = L.Convolution(net[from_layer], num_output=64, pad=1, kernel_size=3, **kwargs)

    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
    net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)

    if nopool:
        name = 'conv1_3'
        net[name] = L.Convolution(net.relu1_2, num_output=64, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool1'
        net.pool1 = L.Pooling(net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2_1 = L.Convolution(net[name], num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
    net.conv2_2 = L.Convolution(net.relu2_1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)

    if nopool:
        name = 'conv2_3'
        net[name] = L.Convolution(net.relu2_2, num_output=128, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool2'
        net[name] = L.Pooling(net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv3_1 = L.Convolution(net[name], num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
    net.conv3_3 = L.Convolution(net.relu3_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_3 = L.ReLU(net.conv3_3, in_place=True)

    if nopool:
        name = 'conv3_4'
        net[name] = L.Convolution(net.relu3_3, num_output=256, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool3'
        net[name] = L.Pooling(net.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv4_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(net.relu4_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    net.conv4_3 = L.Convolution(net.relu4_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_3 = L.ReLU(net.conv4_3, in_place=True)

    #######################################
    # Auto Encoder

    # 1:
    """
    net.conv5_e1 = L.Convolution(net.relu4_3, num_output=2048, pad=1, kernel_size=3, stride=1, **kwargs)
    net.relu5_e1 = L.ReLU(net.conv5_e1, in_place=True)
    net.conv5_e2 = L.Convolution(net.relu5_e1, num_output=2048, pad=2, kernel_size=3, stride=2, **kwargs)
    net.relu5_e2 = L.ReLU(net.conv5_e2, in_place=True)
    net.conv5_e3 = L.Convolution(net.relu5_e2, num_output=2048, pad=2, kernel_size=3, stride=2, dilation=2, **kwargs)
    net.relu5_e3 = L.ReLU(net.conv5_e3, in_place=True)
    net.conv5_e4 = L.Convolution(net.relu5_e3, num_output=1024, pad=4, kernel_size=3, stride=1, dilation=4, **kwargs)
    net.relu5_e4 = L.ReLU(net.conv5_e4, in_place=True)
    net.conv5_e5 = L.Convolution(net.relu5_e4, num_output=1024, pad=8, kernel_size=3, stride=2, dilation=8, **kwargs)
    net.relu5_e5 = L.ReLU(net.conv5_e5, in_place=True)

    net.fc_e6 = L.Convolution(net.relu5_e5, num_output=1024, pad=4, kernel_size=3, stride=1, dilation=4, **kwargs)
    net.relu_e6 = L.ReLU(net.fc_e6, in_place=True)

    net['deconv7_1']= L.Deconvolution(net.relu_e6,
        convolution_param=dict(num_output=1024, pad=8, kernel_size=3, stride=2, dilation=8,
                          weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_1 = L.ReLU(net['deconv7_1'], in_place=True)
    net['deconv7_2']= L.Deconvolution(net.relu7_1,
        convolution_param=dict(num_output=2048, pad=4, kernel_size=3, stride=1, dilation=4,
                          weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_2 = L.ReLU(net['deconv7_2'], in_place=True)
    net['deconv7_3'] = L.Deconvolution(net.relu7_2,
        convolution_param=dict(num_output=2048, pad=2, kernel_size=3, stride=2, dilation=2,
                          weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_3 = L.ReLU(net['deconv7_3'], in_place=True)
    net['deconv7_4'] = L.Deconvolution(net.relu7_3,
        convolution_param=dict(num_output=2048, pad=2, kernel_size=3, stride=2,
                          weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_4 = L.ReLU(net['deconv7_4'], in_place=True)
    net['deconv7_5'] = L.Deconvolution(net.relu7_4,
        convolution_param=dict(num_output=512, pad=1, kernel_size=3, stride=1,
                          weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_5 = L.ReLU(net['deconv7_5'], in_place=True)
    """

    # 2:
    net.conv5_e1 = L.Convolution(net.relu4_3, num_output=512, kernel_size=5, stride=1, **kwargs)
    net.relu5_e1 = L.ReLU(net.conv5_e1, in_place=True)
    net.conv5_e2 = L.Convolution(net.relu5_e1, num_output=256, kernel_size=5, stride=1, **kwargs)
    net.relu5_e2 = L.ReLU(net.conv5_e2, in_place=True)
    net.conv5_e3 = L.Convolution(net.relu5_e2, num_output=128, kernel_size=5, stride=1, **kwargs)
    net.relu5_e3 = L.ReLU(net.conv5_e3, in_place=True)
    net.conv5_e4 = L.Convolution(net.relu5_e3, num_output=64, kernel_size=5, stride=1, **kwargs)
    net.relu5_e4 = L.ReLU(net.conv5_e4, in_place=True)
    net.conv5_e5 = L.Convolution(net.relu5_e4, num_output=256, pad=3, kernel_size=5, stride=2, **kwargs)
    net.relu5_e5 = L.ReLU(net.conv5_e5, in_place=True)

    net.fc_e6 = L.Convolution(net.relu5_e5, num_output=256, pad=12, kernel_size=5, stride=1, dilation=6, **kwargs)
    net.relu_e6 = L.ReLU(net.fc_e6, in_place=True)

    net['deconv7_5'] = L.Deconvolution(net.relu_e6,
                                       convolution_param=dict(num_output=64, pad=3, kernel_size=5, stride=2,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_5 = L.ReLU(net['deconv7_5'], in_place=True)
    net['deconv7_4'] = L.Deconvolution(net.relu7_5,
                                       convolution_param=dict(num_output=128, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_4 = L.ReLU(net['deconv7_4'], in_place=True)
    net['deconv7_3'] = L.Deconvolution(net.relu7_4,
                                       convolution_param=dict(num_output=256, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_3 = L.ReLU(net['deconv7_3'], in_place=True)
    net['deconv7_2'] = L.Deconvolution(net.relu7_3,
                                       convolution_param=dict(num_output=512, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_2 = L.ReLU(net['deconv7_2'], in_place=True)
    net['deconv7_1'] = L.Deconvolution(net.relu7_2,
                                       convolution_param=dict(num_output=512, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_1 = L.ReLU(net['deconv7_1'], in_place=True)
    #######################################

    # 1:
    """
    if nopool:
        name = 'conv7_1'
        net[name] = L.Convolution(net.relu7_5, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool7'
        if dilate_pool7:
            net[name] = L.Pooling(net.relu7_5, pool=P.Pooling.MAX, kernel_size=3, stride=1, pad=1)
            dilation = 2
        else:
            net[name] = L.Pooling(net.relu7_5, pool=P.Pooling.MAX, kernel_size=2, stride=2)
            dilation = 1
    """

    # 2:
    if nopool:
        name = 'conv7_1'
        net[name] = L.Convolution(net.relu7_1, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool7'
        if dilate_pool7:
            net[name] = L.Pooling(net.relu7_1, pool=P.Pooling.MAX, kernel_size=3, stride=1, pad=1)
            dilation = 2
        else:
            net[name] = L.Pooling(net.relu7_1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
            dilation = 1

    kernel_size = 3
    pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
    net.conv8_1 = L.Convolution(net[name], num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation,
                                **kwargs)
    net.relu8_1 = L.ReLU(net.conv8_1, in_place=True)
    net.conv8_2 = L.Convolution(net.relu8_1, num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation,
                                **kwargs)
    net.relu8_2 = L.ReLU(net.conv8_2, in_place=True)
    net.conv8_3 = L.Convolution(net.relu8_2, num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation,
                                **kwargs)
    net.relu8_3 = L.ReLU(net.conv8_3, in_place=True)

    if need_fc:
        if dilated:
            if nopool:
                name = 'conv8_4'
                net[name] = L.Convolution(net.relu8_3, num_output=512, pad=1, kernel_size=3, stride=1, **kwargs)
            else:
                name = 'pool8'
                net[name] = L.Pooling(net.relu8_3, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            if nopool:
                name = 'conv8_4'
                net[name] = L.Convolution(net.relu8_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
            else:
                name = 'pool8'
                net[name] = L.Pooling(net.relu8_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    dilation = dilation * 6
                    kernel_size = 3
                    num_output = 1024
                else:
                    dilation = dilation * 2
                    kernel_size = 7
                    num_output = 4096
            else:
                if reduced:
                    dilation = dilation * 3
                    kernel_size = 3
                    num_output = 1024
                else:
                    kernel_size = 7
                    num_output = 4096
            pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
            net.fc9 = L.Convolution(net[name], num_output=num_output, pad=pad, kernel_size=kernel_size,
                                    dilation=dilation, **kwargs)

            net.relu9 = L.ReLU(net.fc9, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu9, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc10 = L.Convolution(net.relu9, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc10 = L.Convolution(net.relu9, num_output=4096, kernel_size=1, **kwargs)
            net.relu10 = L.ReLU(net.fc10, in_place=True)
            if dropout:
                net.drop10 = L.Dropout(net.relu10, dropout_ratio=0.5, in_place=True)
        else:
            net.fc9 = L.InnerProduct(net.pool8, num_output=4096)
            net.relu9 = L.ReLU(net.fc9, in_place=True)
            if dropout:
                net.drop9 = L.Dropout(net.relu9, dropout_ratio=0.5, in_place=True)
            net.fc10 = L.InnerProduct(net.relu9, num_output=4096)
            net.relu10 = L.ReLU(net.fc10, in_place=True)
            if dropout:
                net.drop10 = L.Dropout(net.relu10, dropout_ratio=0.5, in_place=True)

    # Update freeze layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    # print layers
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net



def AlexNetAutoEncoderTwoStreamNetBody(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
                                   dilated=False, nopool=False, dropout=True, freeze_layers=[], dilate_pool7=False):
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0.1)}

    dkwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]}

    assert from_layer in net.keys()

    net.flow = L.Concat(net.flowx, net.flowy, axis=1)

    net.conv1_1_tp = L.Convolution(net.flow, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_1_tp = L.ReLU(net.conv1_1_tp, in_place=True)
    net.conv1_2_tp = L.Convolution(net.relu1_1_tp, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_2_tp = L.ReLU(net.conv1_2_tp, in_place=True)
    net.pool1_tp = L.Pooling(net.relu1_2_tp, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net.conv2_1_tp = L.Convolution(net.pool1_tp, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_1_tp = L.ReLU(net.conv2_1_tp, in_place=True)
    net.conv2_2_tp = L.Convolution(net.relu2_1_tp, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_2_tp = L.ReLU(net.conv2_2_tp, in_place=True)
    net.pool2_tp = L.Pooling(net.relu2_2_tp, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net.conv3_1_tp = L.Convolution(net.pool2_tp, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_1_tp = L.ReLU(net.conv3_1_tp, in_place=True)
    net.conv3_2_tp = L.Convolution(net.relu3_1_tp, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_2_tp = L.ReLU(net.conv3_2_tp, in_place=True)
    net.conv3_3_tp = L.Convolution(net.relu3_2_tp, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_3_tp = L.ReLU(net.conv3_3_tp, in_place=True)
    net.pool3_tp = L.Pooling(net.relu3_3_tp, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net.conv4_1_tp = L.Convolution(net.pool3_tp, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_1_tp = L.ReLU(net.conv4_1_tp, in_place=True)
#     net.conv4_2_tp = L.Convolution(net.relu4_1_tp, num_output=512, pad=1, kernel_size=3, **kwargs)
#     net.relu4_2_tp = L.ReLU(net.conv4_2_tp, in_place=True)
#     net.conv4_3_tp = L.Convolution(net.relu4_2_tp, num_output=512, pad=1, kernel_size=3, **kwargs)
#     net.relu4_3_tp = L.ReLU(net.conv4_3_tp, in_place=True)

    net.conv1_1 = L.Convolution(net.data, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
    net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)
    net.pool1 = L.Pooling(net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net.conv2_1 = L.Convolution(net.pool1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
    net.conv2_2 = L.Convolution(net.relu2_1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)
    net.pool2 = L.Pooling(net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net.conv3_1 = L.Convolution(net.pool2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
    net.conv3_3 = L.Convolution(net.relu3_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_3 = L.ReLU(net.conv3_3, in_place=True)
    net.pool3 = L.Pooling(net.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net.conv4_1 = L.Convolution(net.pool3, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
#     net.conv4_2 = L.Convolution(net.relu4_1, num_output=512, pad=1, kernel_size=3, **kwargs)
#     net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
#     net.conv4_3 = L.Convolution(net.relu4_2, num_output=512, pad=1, kernel_size=3, **kwargs)
#     net.relu4_3 = L.ReLU(net.conv4_3, in_place=True)

    net.relu4_3_concat = L.Concat(net.relu4_1_tp, net.relu4_1, axis=1)
    net.relu4_map = L.Convolution(net.relu4_3_concat, num_output=512, kernel_size=1, stride=1, **kwargs)

    #######################################
    # Auto Encoder
    #######################################


    # 2:
    net.conv5_e1 = L.Convolution(net.relu4_map, num_output=512, kernel_size=5, stride=1, **kwargs)
    net.relu5_e1 = L.ReLU(net.conv5_e1, in_place=True)
    net.conv5_e2 = L.Convolution(net.relu5_e1, num_output=256, kernel_size=5, stride=1, **kwargs)
    net.relu5_e2 = L.ReLU(net.conv5_e2, in_place=True)
    net.conv5_e3 = L.Convolution(net.relu5_e2, num_output=128, kernel_size=5, stride=1, **kwargs)
    net.relu5_e3 = L.ReLU(net.conv5_e3, in_place=True)
    net.conv5_e4 = L.Convolution(net.relu5_e3, num_output=64, kernel_size=5, stride=1, **kwargs)
    net.relu5_e4 = L.ReLU(net.conv5_e4, in_place=True)
    net.conv5_e5 = L.Convolution(net.relu5_e4, num_output=256, pad=3, kernel_size=5, stride=2, **kwargs)
    net.relu5_e5 = L.ReLU(net.conv5_e5, in_place=True)

    net.fc_e6 = L.Convolution(net.relu5_e5, num_output=256, pad=12, kernel_size=5, stride=1, dilation=6, **kwargs)
    net.relu_e6 = L.ReLU(net.fc_e6, in_place=True)

    net['deconv7_5'] = L.Deconvolution(net.relu_e6,
                                       convolution_param=dict(num_output=64, pad=3, kernel_size=5, stride=2,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_5 = L.ReLU(net['deconv7_5'], in_place=True)
    net['deconv7_4'] = L.Deconvolution(net.relu7_5,
                                       convolution_param=dict(num_output=128, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_4 = L.ReLU(net['deconv7_4'], in_place=True)
    net['deconv7_3'] = L.Deconvolution(net.relu7_4,
                                       convolution_param=dict(num_output=256, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_3 = L.ReLU(net['deconv7_3'], in_place=True)
    net['deconv7_2'] = L.Deconvolution(net.relu7_3,
                                       convolution_param=dict(num_output=512, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_2 = L.ReLU(net['deconv7_2'], in_place=True)
    net['deconv7_1'] = L.Deconvolution(net.relu7_2,
                                       convolution_param=dict(num_output=512, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_1 = L.ReLU(net['deconv7_1'], in_place=True)
    #######################################

    # 1:
    """
    if nopool:
        name = 'conv7_1'
        net[name] = L.Convolution(net.relu7_5, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool7'
        if dilate_pool7:
            net[name] = L.Pooling(net.relu7_5, pool=P.Pooling.MAX, kernel_size=3, stride=1, pad=1)
            dilation = 2
        else:
            net[name] = L.Pooling(net.relu7_5, pool=P.Pooling.MAX, kernel_size=2, stride=2)
            dilation = 1
    """

    # 2:
    if nopool:
        name = 'conv7_1'
        net[name] = L.Convolution(net.relu7_1, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool7'
        if dilate_pool7:
            net[name] = L.Pooling(net.relu7_1, pool=P.Pooling.MAX, kernel_size=3, stride=1, pad=1)
            dilation = 2
        else:
            net[name] = L.Pooling(net.relu7_1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
            dilation = 1

    kernel_size = 3
    pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
    net.conv8_1 = L.Convolution(net[name], num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation,
                                **kwargs)
    net.relu8_1 = L.ReLU(net.conv8_1, in_place=True)
    net.conv8_2 = L.Convolution(net.relu8_1, num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation,
                                **kwargs)
    net.relu8_2 = L.ReLU(net.conv8_2, in_place=True)
    net.conv8_3 = L.Convolution(net.relu8_2, num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation,
                                **kwargs)
    net.relu8_3 = L.ReLU(net.conv8_3, in_place=True)

    if need_fc:
        if dilated:
            if nopool:
                name = 'conv8_4'
                net[name] = L.Convolution(net.relu8_3, num_output=512, pad=1, kernel_size=3, stride=1, **kwargs)
            else:
                name = 'pool8'
                net[name] = L.Pooling(net.relu8_3, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            if nopool:
                name = 'conv8_4'
                net[name] = L.Convolution(net.relu8_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
            else:
                name = 'pool8'
                net[name] = L.Pooling(net.relu8_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    dilation = dilation * 6
                    kernel_size = 3
                    num_output = 1024
                else:
                    dilation = dilation * 2
                    kernel_size = 7
                    num_output = 4096
            else:
                if reduced:
                    dilation = dilation * 3
                    kernel_size = 3
                    num_output = 1024
                else:
                    kernel_size = 7
                    num_output = 4096
            pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
            net.fc9 = L.Convolution(net[name], num_output=num_output, pad=pad, kernel_size=kernel_size,
                                    dilation=dilation, **kwargs)

            net.relu9 = L.ReLU(net.fc9, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu9, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc10 = L.Convolution(net.relu9, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc10 = L.Convolution(net.relu9, num_output=4096, kernel_size=1, **kwargs)
            net.relu10 = L.ReLU(net.fc10, in_place=True)
            if dropout:
                net.drop10 = L.Dropout(net.relu10, dropout_ratio=0.5, in_place=True)
        else:
            net.fc9 = L.InnerProduct(net.pool8, num_output=4096)
            net.relu9 = L.ReLU(net.fc9, in_place=True)
            if dropout:
                net.drop9 = L.Dropout(net.relu9, dropout_ratio=0.5, in_place=True)
            net.fc10 = L.InnerProduct(net.relu9, num_output=4096)
            net.relu10 = L.ReLU(net.fc10, in_place=True)
            if dropout:
                net.drop10 = L.Dropout(net.relu10, dropout_ratio=0.5, in_place=True)

    # Update freeze layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    # print layers
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net



def VGGAutoEncoderTwoStreamNetBody(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
                                   dilated=False, nopool=False, dropout=True, freeze_layers=[], dilate_pool7=False):
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0.1)}

    dkwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]}

    assert from_layer in net.keys()

    net.flow = L.Concat(net.flowx, net.flowy, axis=1)

    net.conv1_1_tp = L.Convolution(net.flow, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_1_tp = L.ReLU(net.conv1_1_tp, in_place=True)
    net.conv1_2_tp = L.Convolution(net.relu1_1_tp, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_2_tp = L.ReLU(net.conv1_2_tp, in_place=True)
    net.pool1_tp = L.Pooling(net.relu1_2_tp, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net.conv2_1_tp = L.Convolution(net.pool1_tp, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_1_tp = L.ReLU(net.conv2_1_tp, in_place=True)
    net.conv2_2_tp = L.Convolution(net.relu2_1_tp, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_2_tp = L.ReLU(net.conv2_2_tp, in_place=True)
    net.pool2_tp = L.Pooling(net.relu2_2_tp, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net.conv3_1_tp = L.Convolution(net.pool2_tp, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_1_tp = L.ReLU(net.conv3_1_tp, in_place=True)
    net.conv3_2_tp = L.Convolution(net.relu3_1_tp, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_2_tp = L.ReLU(net.conv3_2_tp, in_place=True)
    net.conv3_3_tp = L.Convolution(net.relu3_2_tp, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_3_tp = L.ReLU(net.conv3_3_tp, in_place=True)
    net.pool3_tp = L.Pooling(net.relu3_3_tp, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net.conv4_1_tp = L.Convolution(net.pool3_tp, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_1_tp = L.ReLU(net.conv4_1_tp, in_place=True)
    net.conv4_2_tp = L.Convolution(net.relu4_1_tp, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_2_tp = L.ReLU(net.conv4_2_tp, in_place=True)
    net.conv4_3_tp = L.Convolution(net.relu4_2_tp, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_3_tp = L.ReLU(net.conv4_3_tp, in_place=True)

    net.conv1_1 = L.Convolution(net.data, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
    net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)
    net.pool1 = L.Pooling(net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net.conv2_1 = L.Convolution(net.pool1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
    net.conv2_2 = L.Convolution(net.relu2_1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)
    net.pool2 = L.Pooling(net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net.conv3_1 = L.Convolution(net.pool2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
    net.conv3_3 = L.Convolution(net.relu3_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_3 = L.ReLU(net.conv3_3, in_place=True)
    net.pool3 = L.Pooling(net.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net.conv4_1 = L.Convolution(net.pool3, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(net.relu4_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    net.conv4_3 = L.Convolution(net.relu4_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_3 = L.ReLU(net.conv4_3, in_place=True)

    net.relu4_3_concat = L.Concat(net.relu4_3, net.relu4_3_tp, axis=1)
    net.relu4_map = L.Convolution(net.relu4_3_concat, num_output=512, kernel_size=1, stride=1, **kwargs)

    #######################################
    # Auto Encoder
    #######################################


    # 2:
    net.conv5_e1 = L.Convolution(net.relu4_map, num_output=512, kernel_size=5, stride=1, **kwargs)
    net.relu5_e1 = L.ReLU(net.conv5_e1, in_place=True)
    net.conv5_e2 = L.Convolution(net.relu5_e1, num_output=256, kernel_size=5, stride=1, **kwargs)
    net.relu5_e2 = L.ReLU(net.conv5_e2, in_place=True)
    net.conv5_e3 = L.Convolution(net.relu5_e2, num_output=128, kernel_size=5, stride=1, **kwargs)
    net.relu5_e3 = L.ReLU(net.conv5_e3, in_place=True)
    net.conv5_e4 = L.Convolution(net.relu5_e3, num_output=64, kernel_size=5, stride=1, **kwargs)
    net.relu5_e4 = L.ReLU(net.conv5_e4, in_place=True)
    net.conv5_e5 = L.Convolution(net.relu5_e4, num_output=256, pad=3, kernel_size=5, stride=2, **kwargs)
    net.relu5_e5 = L.ReLU(net.conv5_e5, in_place=True)

    net.fc_e6 = L.Convolution(net.relu5_e5, num_output=256, pad=12, kernel_size=5, stride=1, dilation=6, **kwargs)
    net.relu_e6 = L.ReLU(net.fc_e6, in_place=True)

    net['deconv7_5'] = L.Deconvolution(net.relu_e6,
                                       convolution_param=dict(num_output=64, pad=3, kernel_size=5, stride=2,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_5 = L.ReLU(net['deconv7_5'], in_place=True)
    net['deconv7_4'] = L.Deconvolution(net.relu7_5,
                                       convolution_param=dict(num_output=128, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_4 = L.ReLU(net['deconv7_4'], in_place=True)
    net['deconv7_3'] = L.Deconvolution(net.relu7_4,
                                       convolution_param=dict(num_output=256, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_3 = L.ReLU(net['deconv7_3'], in_place=True)
    net['deconv7_2'] = L.Deconvolution(net.relu7_3,
                                       convolution_param=dict(num_output=512, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_2 = L.ReLU(net['deconv7_2'], in_place=True)
    net['deconv7_1'] = L.Deconvolution(net.relu7_2,
                                       convolution_param=dict(num_output=512, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_1 = L.ReLU(net['deconv7_1'], in_place=True)
    #######################################

    # 1:
    """
    if nopool:
        name = 'conv7_1'
        net[name] = L.Convolution(net.relu7_5, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool7'
        if dilate_pool7:
            net[name] = L.Pooling(net.relu7_5, pool=P.Pooling.MAX, kernel_size=3, stride=1, pad=1)
            dilation = 2
        else:
            net[name] = L.Pooling(net.relu7_5, pool=P.Pooling.MAX, kernel_size=2, stride=2)
            dilation = 1
    """

    # 2:
    if nopool:
        name = 'conv7_1'
        net[name] = L.Convolution(net.relu7_1, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool7'
        if dilate_pool7:
            net[name] = L.Pooling(net.relu7_1, pool=P.Pooling.MAX, kernel_size=3, stride=1, pad=1)
            dilation = 2
        else:
            net[name] = L.Pooling(net.relu7_1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
            dilation = 1

    kernel_size = 3
    pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
    net.conv8_1 = L.Convolution(net[name], num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation,
                                **kwargs)
    net.relu8_1 = L.ReLU(net.conv8_1, in_place=True)
    net.conv8_2 = L.Convolution(net.relu8_1, num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation,
                                **kwargs)
    net.relu8_2 = L.ReLU(net.conv8_2, in_place=True)
    net.conv8_3 = L.Convolution(net.relu8_2, num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation,
                                **kwargs)
    net.relu8_3 = L.ReLU(net.conv8_3, in_place=True)

    if need_fc:
        if dilated:
            if nopool:
                name = 'conv8_4'
                net[name] = L.Convolution(net.relu8_3, num_output=512, pad=1, kernel_size=3, stride=1, **kwargs)
            else:
                name = 'pool8'
                net[name] = L.Pooling(net.relu8_3, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            if nopool:
                name = 'conv8_4'
                net[name] = L.Convolution(net.relu8_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
            else:
                name = 'pool8'
                net[name] = L.Pooling(net.relu8_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    dilation = dilation * 6
                    kernel_size = 3
                    num_output = 1024
                else:
                    dilation = dilation * 2
                    kernel_size = 7
                    num_output = 4096
            else:
                if reduced:
                    dilation = dilation * 3
                    kernel_size = 3
                    num_output = 1024
                else:
                    kernel_size = 7
                    num_output = 4096
            pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
            net.fc9 = L.Convolution(net[name], num_output=num_output, pad=pad, kernel_size=kernel_size,
                                    dilation=dilation, **kwargs)

            net.relu9 = L.ReLU(net.fc9, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu9, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc10 = L.Convolution(net.relu9, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc10 = L.Convolution(net.relu9, num_output=4096, kernel_size=1, **kwargs)
            net.relu10 = L.ReLU(net.fc10, in_place=True)
            if dropout:
                net.drop10 = L.Dropout(net.relu10, dropout_ratio=0.5, in_place=True)
        else:
            net.fc9 = L.InnerProduct(net.pool8, num_output=4096)
            net.relu9 = L.ReLU(net.fc9, in_place=True)
            if dropout:
                net.drop9 = L.Dropout(net.relu9, dropout_ratio=0.5, in_place=True)
            net.fc10 = L.InnerProduct(net.relu9, num_output=4096)
            net.relu10 = L.ReLU(net.fc10, in_place=True)
            if dropout:
                net.drop10 = L.Dropout(net.relu10, dropout_ratio=0.5, in_place=True)

    # Update freeze layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    # print layers
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net


def VGGAutoEncoderTwoStreamRegressNetBody(net, need_fc=True, fully_conv=False, reduced=False,
                                          dilated=False, nopool=False, dropout=True, freeze_layers=[],
                                          dilate_pool7=False):
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0.1)}

    dkwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]}

    net.flow = L.Concat(net.flowx, net.flowy, axis=1)
    net.flow_r1 = L.Concat(net.flow_x_r1, net.flow_y_r1, axis=1)
    net.flow_r2 = L.Concat(net.flow_x_r2, net.flow_y_r2, axis=1)

    def twoS(net, input_img, input_flow, suffix):

        net['conv1_1_tp' + suffix], net['relu1_1_tp' + suffix] = conv_relu_share('conv1_1_tp_w', 'conv1_1_tp_b',
                                                                                 input_flow, 64, ks=3, stride=1, pad=1)
        net['conv1_2_tp' + suffix], net['relu1_2_tp' + suffix] = conv_relu_share('conv1_2_tp_w', 'conv1_2_tp_b',
                                                                                 net['relu1_1_tp' + suffix], 64, ks=3,
                                                                                 stride=1, pad=1)
        net['pool1_tp' + suffix] = L.Pooling(net['relu1_2_tp' + suffix], pool=P.Pooling.MAX, kernel_size=2, stride=2)
        net['conv2_1_tp' + suffix], net['relu2_1_tp' + suffix] = conv_relu_share('conv2_1_tp_w', 'conv2_1_tp_b',
                                                                                 net['pool1_tp' + suffix], 128, ks=3,
                                                                                 stride=1, pad=1)
        net['conv2_2_tp' + suffix], net['relu2_2_tp' + suffix] = conv_relu_share('conv2_2_tp_w', 'conv2_2_tp_b',
                                                                                 net['relu2_1_tp' + suffix], 128, ks=3,
                                                                                 stride=1, pad=1)
        net['pool2_tp' + suffix] = L.Pooling(net['relu2_2_tp' + suffix], pool=P.Pooling.MAX, kernel_size=2, stride=2)
        net['conv3_1_tp' + suffix], net['relu3_1_tp' + suffix] = conv_relu_share('conv3_1_tp_w', 'conv3_1_tp_b',
                                                                                 net['pool2_tp' + suffix], 256, ks=3,
                                                                                 stride=1, pad=1)
        net['conv3_2_tp' + suffix], net['relu3_2_tp' + suffix] = conv_relu_share('conv3_2_tp_w', 'conv3_2_tp_b',
                                                                                 net['relu3_1_tp' + suffix], 256, ks=3,
                                                                                 stride=1, pad=1)
        net['conv3_3_tp' + suffix], net['relu3_3_tp' + suffix] = conv_relu_share('conv3_3_tp_w', 'conv3_3_tp_b',
                                                                                 net['relu3_2_tp' + suffix], 256, ks=3,
                                                                                 stride=1, pad=1)
        net['pool3_tp' + suffix] = L.Pooling(net['relu3_3_tp' + suffix], pool=P.Pooling.MAX, kernel_size=2, stride=2)
        net['conv4_1_tp' + suffix], net['relu4_1_tp' + suffix] = conv_relu_share('conv4_1_tp_w', 'conv4_1_tp_b',
                                                                                 net['pool3_tp' + suffix], 512, ks=3,
                                                                                 stride=1, pad=1)
        net['conv4_2_tp' + suffix], net['relu4_2_tp' + suffix] = conv_relu_share('conv4_2_tp_w', 'conv4_2_tp_b',
                                                                                 net['relu4_1_tp' + suffix], 512, ks=3,
                                                                                 stride=1, pad=1)
        net['conv4_3_tp' + suffix], net['relu4_3_tp' + suffix] = conv_relu_share('conv4_3_tp_w', 'conv4_3_tp_b',
                                                                                 net['relu4_2_tp' + suffix], 512, ks=3,
                                                                                 stride=1, pad=1)

        net['conv1_1' + suffix], net['relu1_1' + suffix] = conv_relu_share('conv1_1_w', 'conv1_1_b', input_img, 64,
                                                                           ks=3, stride=1, pad=1)
        net['conv1_2' + suffix], net['relu1_2' + suffix] = conv_relu_share('conv1_2_w', 'conv1_2_b',
                                                                           net['relu1_1' + suffix], 64, ks=3, stride=1,
                                                                           pad=1)
        net['pool1' + suffix] = L.Pooling(net['relu1_2' + suffix], pool=P.Pooling.MAX, kernel_size=2, stride=2)
        net['conv2_1' + suffix], net['relu2_1' + suffix] = conv_relu_share('conv2_1_w', 'conv2_1_b',
                                                                           net['pool1' + suffix], 128, ks=3, stride=1,
                                                                           pad=1)
        net['conv2_2' + suffix], net['relu2_2' + suffix] = conv_relu_share('conv2_2_w', 'conv2_2_b',
                                                                           net['relu2_1' + suffix], 128, ks=3, stride=1,
                                                                           pad=1)
        net['pool2' + suffix] = L.Pooling(net['relu2_2' + suffix], pool=P.Pooling.MAX, kernel_size=2, stride=2)
        net['conv3_1' + suffix], net['relu3_1' + suffix] = conv_relu_share('conv3_1_w', 'conv3_1_b',
                                                                           net['pool2' + suffix], 256, ks=3, stride=1,
                                                                           pad=1)
        net['conv3_2' + suffix], net['relu3_2' + suffix] = conv_relu_share('conv3_2_w', 'conv3_2_b',
                                                                           net['relu3_1' + suffix], 256, ks=3, stride=1,
                                                                           pad=1)
        net['conv3_3' + suffix], net['relu3_3' + suffix] = conv_relu_share('conv3_3_w', 'conv3_3_b',
                                                                           net['relu3_2' + suffix], 256, ks=3, stride=1,
                                                                           pad=1)
        net['pool3' + suffix] = L.Pooling(net['relu3_3' + suffix], pool=P.Pooling.MAX, kernel_size=2, stride=2)
        net['conv4_1' + suffix], net['relu4_1' + suffix] = conv_relu_share('conv4_1_w', 'conv4_1_b',
                                                                           net['pool3' + suffix], 512, ks=3, stride=1,
                                                                           pad=1)
        net['conv4_2' + suffix], net['relu4_2' + suffix] = conv_relu_share('conv4_2_w', 'conv4_2_b',
                                                                           net['relu4_1' + suffix], 512, ks=3, stride=1,
                                                                           pad=1)
        net['conv4_3' + suffix], net['relu4_3' + suffix] = conv_relu_share('conv4_3_w', 'conv4_3_b',
                                                                           net['relu4_2' + suffix], 512, ks=3, stride=1,
                                                                           pad=1)

        net['relu4_3_concat' + suffix] = L.Concat(net['relu4_3_tp' + suffix], net['relu4_3' + suffix], axis=1)
        net['relu4_map' + suffix] = L.Convolution(net['relu4_3_concat' + suffix], num_output=512, kernel_size=1,
                                                  stride=1, **kwargs)

        # Auto Encoder
        net['conv5_e1' + suffix], net['relu5_e1' + suffix] = conv_relu_share('conv5_e1_w', 'conv5_e1_b',
                                                                             net['relu4_map' + suffix], 512, ks=5,
                                                                             stride=1, pad=1)
        net['conv5_e2' + suffix], net['relu5_e2' + suffix] = conv_relu_share('conv5_e2_w', 'conv5_e2_b',
                                                                             net['relu5_e1' + suffix], 256, ks=5,
                                                                             stride=1, pad=1)
        net['conv5_e3' + suffix], net['relu5_e3' + suffix] = conv_relu_share('conv5_e3_w', 'conv5_e3_b',
                                                                             net['relu5_e2' + suffix], 128, ks=5,
                                                                             stride=1, pad=1)
        net['conv5_e4' + suffix], net['relu5_e4' + suffix] = conv_relu_share('conv5_e4_w', 'conv5_e4_b',
                                                                             net['relu5_e3' + suffix], 64, ks=5,
                                                                             stride=1, pad=1)
        net['conv5_e5' + suffix], net['relu5_e5' + suffix] = conv_relu_share('conv5_e5_w', 'conv5_e5_b',
                                                                             net['relu5_e4' + suffix], 256, ks=5,
                                                                             stride=2, pad=3)
        net['fc_e6' + suffix], net['relu5_e6' + suffix] = conv_relu_dilation_share('fc_e6_w', 'fc_e6_b',
                                                                                   net['relu5_e5' + suffix], 256, ks=5,
                                                                                   stride=1, pad=12, dilation=6)

    twoS(net, net.data, net.flow, '')
    twoS(net, net.data_r1, net.flow_r1, '_r1')
    twoS(net, net.data_r2, net.flow_r2, '_r2')
    # print net.keys()

    assert 'relu5_e6' in net.keys()
    assert 'relu5_e6_r1' in net.keys()
    assert 'relu5_e6_r2' in net.keys()
    # 2:

    # regressor
    net.conv_rg6 = L.Convolution(net['relu5_e6_r1'], num_output=256, pad=2, kernel_size=5, **kwargs)
    net.relu_rg6 = L.ReLU(net.conv_rg6, in_place=True)
    net.conv_rg7 = L.Convolution(net.relu_rg6, num_output=256, pad=2, kernel_size=5, **kwargs)
    net.relu_rg7 = L.ReLU(net.conv_rg7, in_place=True)
    net.conv_rg8 = L.Convolution(net.relu_rg7, num_output=1024, pad=6, kernel_size=5, dilation=3, **kwargs)
    net.relu_rg8 = L.ReLU(net.conv_rg8, in_place=True)
    net.conv_rg9 = L.Convolution(net.relu_rg8, num_output=256, kernel_size=1, **kwargs)
    net.relu_rg9 = L.ReLU(net.conv_rg9, in_place=True)
    net.loss_rg = L.EuclideanLoss(net.relu_rg9, net['relu5_e6_r2'], loss_weight=0.001)

    net['deconv7_5'] = L.Deconvolution(net['relu5_e6'],
                                       convolution_param=dict(num_output=64, pad=3, kernel_size=5, stride=2,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_5 = L.ReLU(net['deconv7_5'], in_place=True)
    net['deconv7_4'] = L.Deconvolution(net.relu7_5,
                                       convolution_param=dict(num_output=128, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_4 = L.ReLU(net['deconv7_4'], in_place=True)
    net['deconv7_3'] = L.Deconvolution(net.relu7_4,
                                       convolution_param=dict(num_output=256, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_3 = L.ReLU(net['deconv7_3'], in_place=True)
    net['deconv7_2'] = L.Deconvolution(net.relu7_3,
                                       convolution_param=dict(num_output=512, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_2 = L.ReLU(net['deconv7_2'], in_place=True)
    net['deconv7_1'] = L.Deconvolution(net.relu7_2,
                                       convolution_param=dict(num_output=512, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_1 = L.ReLU(net['deconv7_1'], in_place=True)
    #######################################

    # 1:
    """
    if nopool:
        name = 'conv7_1'
        net[name] = L.Convolution(net.relu7_5, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool7'
        if dilate_pool7:
            net[name] = L.Pooling(net.relu7_5, pool=P.Pooling.MAX, kernel_size=3, stride=1, pad=1)
            dilation = 2
        else:
            net[name] = L.Pooling(net.relu7_5, pool=P.Pooling.MAX, kernel_size=2, stride=2)
            dilation = 1
    """

    # 2:
    if nopool:
        name = 'conv7_1'
        net[name] = L.Convolution(net.relu7_1, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool7'
        if dilate_pool7:
            net[name] = L.Pooling(net.relu7_1, pool=P.Pooling.MAX, kernel_size=3, stride=1, pad=1)
            dilation = 2
        else:
            net[name] = L.Pooling(net.relu7_1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
            dilation = 1

    kernel_size = 3
    pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
    net.conv8_1 = L.Convolution(net[name], num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation,
                                **kwargs)
    net.relu8_1 = L.ReLU(net.conv8_1, in_place=True)
    net.conv8_2 = L.Convolution(net.relu8_1, num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation,
                                **kwargs)
    net.relu8_2 = L.ReLU(net.conv8_2, in_place=True)
    net.conv8_3 = L.Convolution(net.relu8_2, num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation,
                                **kwargs)
    net.relu8_3 = L.ReLU(net.conv8_3, in_place=True)

    if need_fc:
        if dilated:
            if nopool:
                name = 'conv8_4'
                net[name] = L.Convolution(net.relu8_3, num_output=512, pad=1, kernel_size=3, stride=1, **kwargs)
            else:
                name = 'pool8'
                net[name] = L.Pooling(net.relu8_3, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            if nopool:
                name = 'conv8_4'
                net[name] = L.Convolution(net.relu8_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
            else:
                name = 'pool8'
                net[name] = L.Pooling(net.relu8_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    dilation = dilation * 6
                    kernel_size = 3
                    num_output = 1024
                else:
                    dilation = dilation * 2
                    kernel_size = 7
                    num_output = 4096
            else:
                if reduced:
                    dilation = dilation * 3
                    kernel_size = 3
                    num_output = 1024
                else:
                    kernel_size = 7
                    num_output = 4096
            pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
            net.fc9 = L.Convolution(net[name], num_output=num_output, pad=pad, kernel_size=kernel_size,
                                    dilation=dilation, **kwargs)

            net.relu9 = L.ReLU(net.fc9, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu9, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc10 = L.Convolution(net.relu9, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc10 = L.Convolution(net.relu9, num_output=4096, kernel_size=1, **kwargs)
            net.relu10 = L.ReLU(net.fc10, in_place=True)
            if dropout:
                net.drop10 = L.Dropout(net.relu10, dropout_ratio=0.5, in_place=True)
        else:
            net.fc9 = L.InnerProduct(net.pool8, num_output=4096)
            net.relu9 = L.ReLU(net.fc9, in_place=True)
            if dropout:
                net.drop9 = L.Dropout(net.relu9, dropout_ratio=0.5, in_place=True)
            net.fc10 = L.InnerProduct(net.relu9, num_output=4096)
            net.relu10 = L.ReLU(net.fc10, in_place=True)
            if dropout:
                net.drop10 = L.Dropout(net.relu10, dropout_ratio=0.5, in_place=True)

    # Update freeze layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    # print layers
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net


def VGGAutoEncoderTwoStreamRegressNetTestBody(net, need_fc=True, fully_conv=False, reduced=False,
                                              dilated=False, nopool=False, dropout=True, freeze_layers=[],
                                              dilate_pool7=False):
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0.1)}

    dkwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]}

    def twoS(net, input_img, input_flow, suffix):

        net['conv1_1_tp' + suffix], net['relu1_1_tp' + suffix] = conv_relu_share('conv1_1_tp_w', 'conv1_1_tp_b',
                                                                                 input_flow, 64, ks=3, stride=1, pad=1)
        net['conv1_2_tp' + suffix], net['relu1_2_tp' + suffix] = conv_relu_share('conv1_2_tp_w', 'conv1_2_tp_b',
                                                                                 net['relu1_1_tp' + suffix], 64, ks=3,
                                                                                 stride=1, pad=1)
        net['pool1_tp' + suffix] = L.Pooling(net['relu1_2_tp' + suffix], pool=P.Pooling.MAX, kernel_size=2, stride=2)
        net['conv2_1_tp' + suffix], net['relu2_1_tp' + suffix] = conv_relu_share('conv2_1_tp_w', 'conv2_1_tp_b',
                                                                                 net['pool1_tp' + suffix], 128, ks=3,
                                                                                 stride=1, pad=1)
        net['conv2_2_tp' + suffix], net['relu2_2_tp' + suffix] = conv_relu_share('conv2_2_tp_w', 'conv2_2_tp_b',
                                                                                 net['relu2_1_tp' + suffix], 128, ks=3,
                                                                                 stride=1, pad=1)
        net['pool2_tp' + suffix] = L.Pooling(net['relu2_2_tp' + suffix], pool=P.Pooling.MAX, kernel_size=2, stride=2)
        net['conv3_1_tp' + suffix], net['relu3_1_tp' + suffix] = conv_relu_share('conv3_1_tp_w', 'conv3_1_tp_b',
                                                                                 net['pool2_tp' + suffix], 256, ks=3,
                                                                                 stride=1, pad=1)
        net['conv3_2_tp' + suffix], net['relu3_2_tp' + suffix] = conv_relu_share('conv3_2_tp_w', 'conv3_2_tp_b',
                                                                                 net['relu3_1_tp' + suffix], 256, ks=3,
                                                                                 stride=1, pad=1)
        net['conv3_3_tp' + suffix], net['relu3_3_tp' + suffix] = conv_relu_share('conv3_3_tp_w', 'conv3_3_tp_b',
                                                                                 net['relu3_2_tp' + suffix], 256, ks=3,
                                                                                 stride=1, pad=1)
        net['pool3_tp' + suffix] = L.Pooling(net['relu3_3_tp' + suffix], pool=P.Pooling.MAX, kernel_size=2, stride=2)
        net['conv4_1_tp' + suffix], net['relu4_1_tp' + suffix] = conv_relu_share('conv4_1_tp_w', 'conv4_1_tp_b',
                                                                                 net['pool3_tp' + suffix], 512, ks=3,
                                                                                 stride=1, pad=1)
        net['conv4_2_tp' + suffix], net['relu4_2_tp' + suffix] = conv_relu_share('conv4_2_tp_w', 'conv4_2_tp_b',
                                                                                 net['relu4_1_tp' + suffix], 512, ks=3,
                                                                                 stride=1, pad=1)
        net['conv4_3_tp' + suffix], net['relu4_3_tp' + suffix] = conv_relu_share('conv4_3_tp_w', 'conv4_3_tp_b',
                                                                                 net['relu4_2_tp' + suffix], 512, ks=3,
                                                                                 stride=1, pad=1)

        net['conv1_1' + suffix], net['relu1_1' + suffix] = conv_relu_share('conv1_1_w', 'conv1_1_b', input_img, 64,
                                                                           ks=3, stride=1, pad=1)
        net['conv1_2' + suffix], net['relu1_2' + suffix] = conv_relu_share('conv1_2_w', 'conv1_2_b',
                                                                           net['relu1_1' + suffix], 64, ks=3, stride=1,
                                                                           pad=1)
        net['pool1' + suffix] = L.Pooling(net['relu1_2' + suffix], pool=P.Pooling.MAX, kernel_size=2, stride=2)
        net['conv2_1' + suffix], net['relu2_1' + suffix] = conv_relu_share('conv2_1_w', 'conv2_1_b',
                                                                           net['pool1' + suffix], 128, ks=3, stride=1,
                                                                           pad=1)
        net['conv2_2' + suffix], net['relu2_2' + suffix] = conv_relu_share('conv2_2_w', 'conv2_2_b',
                                                                           net['relu2_1' + suffix], 128, ks=3, stride=1,
                                                                           pad=1)
        net['pool2' + suffix] = L.Pooling(net['relu2_2' + suffix], pool=P.Pooling.MAX, kernel_size=2, stride=2)
        net['conv3_1' + suffix], net['relu3_1' + suffix] = conv_relu_share('conv3_1_w', 'conv3_1_b',
                                                                           net['pool2' + suffix], 256, ks=3, stride=1,
                                                                           pad=1)
        net['conv3_2' + suffix], net['relu3_2' + suffix] = conv_relu_share('conv3_2_w', 'conv3_2_b',
                                                                           net['relu3_1' + suffix], 256, ks=3, stride=1,
                                                                           pad=1)
        net['conv3_3' + suffix], net['relu3_3' + suffix] = conv_relu_share('conv3_3_w', 'conv3_3_b',
                                                                           net['relu3_2' + suffix], 256, ks=3, stride=1,
                                                                           pad=1)
        net['pool3' + suffix] = L.Pooling(net['relu3_3' + suffix], pool=P.Pooling.MAX, kernel_size=2, stride=2)
        net['conv4_1' + suffix], net['relu4_1' + suffix] = conv_relu_share('conv4_1_w', 'conv4_1_b',
                                                                           net['pool3' + suffix], 512, ks=3, stride=1,
                                                                           pad=1)
        net['conv4_2' + suffix], net['relu4_2' + suffix] = conv_relu_share('conv4_2_w', 'conv4_2_b',
                                                                           net['relu4_1' + suffix], 512, ks=3, stride=1,
                                                                           pad=1)
        net['conv4_3' + suffix], net['relu4_3' + suffix] = conv_relu_share('conv4_3_w', 'conv4_3_b',
                                                                           net['relu4_2' + suffix], 512, ks=3, stride=1,
                                                                           pad=1)

        net['relu4_3_concat' + suffix] = L.Concat(net['relu4_3_tp' + suffix], net['relu4_3' + suffix], axis=1)
        net['relu4_map' + suffix] = L.Convolution(net['relu4_3_concat' + suffix], num_output=512, kernel_size=1,
                                                  stride=1, **kwargs)

        # Auto Encoder
        net['conv5_e1' + suffix], net['relu5_e1' + suffix] = conv_relu_share('conv5_e1_w', 'conv5_e1_b',
                                                                             net['relu4_map' + suffix], 512, ks=5,
                                                                             stride=1, pad=1)
        net['conv5_e2' + suffix], net['relu5_e2' + suffix] = conv_relu_share('conv5_e2_w', 'conv5_e2_b',
                                                                             net['relu5_e1' + suffix], 256, ks=5,
                                                                             stride=1, pad=1)
        net['conv5_e3' + suffix], net['relu5_e3' + suffix] = conv_relu_share('conv5_e3_w', 'conv5_e3_b',
                                                                             net['relu5_e2' + suffix], 128, ks=5,
                                                                             stride=1, pad=1)
        net['conv5_e4' + suffix], net['relu5_e4' + suffix] = conv_relu_share('conv5_e4_w', 'conv5_e4_b',
                                                                             net['relu5_e3' + suffix], 64, ks=5,
                                                                             stride=1, pad=1)
        net['conv5_e5' + suffix], net['relu5_e5' + suffix] = conv_relu_share('conv5_e5_w', 'conv5_e5_b',
                                                                             net['relu5_e4' + suffix], 256, ks=5,
                                                                             stride=2, pad=3)
        net['fc_e6' + suffix], net['relu5_e6' + suffix] = conv_relu_dilation_share('fc_e6_w', 'fc_e6_b',
                                                                                   net['relu5_e5' + suffix], 256, ks=5,
                                                                                   stride=1, pad=12, dilation=6)

    # net.flow_r1 = L.Concat(net.flow_x_r1, net.flow_y_r1, axis=1)
    twoS(net, net.data_r1, net.flow_r1, '_r1')

    # regressor
    net.conv_rg6 = L.Convolution(net['relu5_e6_r1'], num_output=256, pad=2, kernel_size=5, **kwargs)
    net.relu_rg6 = L.ReLU(net.conv_rg6, in_place=True)
    net.conv_rg7 = L.Convolution(net.relu_rg6, num_output=256, pad=2, kernel_size=5, **kwargs)
    net.relu_rg7 = L.ReLU(net.conv_rg7, in_place=True)
    net.conv_rg8 = L.Convolution(net.relu_rg7, num_output=1024, pad=6, kernel_size=5, dilation=3, **kwargs)
    net.relu_rg8 = L.ReLU(net.conv_rg8, in_place=True)
    net.conv_rg9 = L.Convolution(net.relu_rg8, num_output=256, kernel_size=1, **kwargs)
    net.relu_rg9 = L.ReLU(net.conv_rg9, in_place=True)

    net['deconv7_5'] = L.Deconvolution(net['relu_rg9'],
                                       convolution_param=dict(num_output=64, pad=3, kernel_size=5, stride=2,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_5 = L.ReLU(net['deconv7_5'], in_place=True)
    net['deconv7_4'] = L.Deconvolution(net.relu7_5,
                                       convolution_param=dict(num_output=128, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_4 = L.ReLU(net['deconv7_4'], in_place=True)
    net['deconv7_3'] = L.Deconvolution(net.relu7_4,
                                       convolution_param=dict(num_output=256, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_3 = L.ReLU(net['deconv7_3'], in_place=True)
    net['deconv7_2'] = L.Deconvolution(net.relu7_3,
                                       convolution_param=dict(num_output=512, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_2 = L.ReLU(net['deconv7_2'], in_place=True)
    net['deconv7_1'] = L.Deconvolution(net.relu7_2,
                                       convolution_param=dict(num_output=512, kernel_size=5, stride=1,
                                                              weight_filler=dict(type='xavier'),
                                                              bias_filler=dict(type='constant', value=0)), **dkwargs)
    net.relu7_1 = L.ReLU(net['deconv7_1'], in_place=True)
    #######################################

    # 2:
    if nopool:
        name = 'conv7_1'
        net[name] = L.Convolution(net.relu7_1, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool7'
        if dilate_pool7:
            net[name] = L.Pooling(net.relu7_1, pool=P.Pooling.MAX, kernel_size=3, stride=1, pad=1)
            dilation = 2
        else:
            net[name] = L.Pooling(net.relu7_1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
            dilation = 1

    kernel_size = 3
    pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
    net.conv8_1 = L.Convolution(net[name], num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation,
                                **kwargs)
    net.relu8_1 = L.ReLU(net.conv8_1, in_place=True)
    net.conv8_2 = L.Convolution(net.relu8_1, num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation,
                                **kwargs)
    net.relu8_2 = L.ReLU(net.conv8_2, in_place=True)
    net.conv8_3 = L.Convolution(net.relu8_2, num_output=512, pad=pad, kernel_size=kernel_size, dilation=dilation,
                                **kwargs)
    net.relu8_3 = L.ReLU(net.conv8_3, in_place=True)

    if need_fc:
        if dilated:
            if nopool:
                name = 'conv8_4'
                net[name] = L.Convolution(net.relu8_3, num_output=512, pad=1, kernel_size=3, stride=1, **kwargs)
            else:
                name = 'pool8'
                net[name] = L.Pooling(net.relu8_3, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            if nopool:
                name = 'conv8_4'
                net[name] = L.Convolution(net.relu8_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
            else:
                name = 'pool8'
                net[name] = L.Pooling(net.relu8_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    dilation = dilation * 6
                    kernel_size = 3
                    num_output = 1024
                else:
                    dilation = dilation * 2
                    kernel_size = 7
                    num_output = 4096
            else:
                if reduced:
                    dilation = dilation * 3
                    kernel_size = 3
                    num_output = 1024
                else:
                    kernel_size = 7
                    num_output = 4096
            pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) / 2
            net.fc9 = L.Convolution(net[name], num_output=num_output, pad=pad, kernel_size=kernel_size,
                                    dilation=dilation, **kwargs)

            net.relu9 = L.ReLU(net.fc9, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu9, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc10 = L.Convolution(net.relu9, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc10 = L.Convolution(net.relu9, num_output=4096, kernel_size=1, **kwargs)
            net.relu10 = L.ReLU(net.fc10, in_place=True)
            if dropout:
                net.drop10 = L.Dropout(net.relu10, dropout_ratio=0.5, in_place=True)
        else:
            net.fc9 = L.InnerProduct(net.pool8, num_output=4096)
            net.relu9 = L.ReLU(net.fc9, in_place=True)
            if dropout:
                net.drop9 = L.Dropout(net.relu9, dropout_ratio=0.5, in_place=True)
            net.fc10 = L.InnerProduct(net.relu9, num_output=4096)
            net.relu10 = L.ReLU(net.fc10, in_place=True)
            if dropout:
                net.drop10 = L.Dropout(net.relu10, dropout_ratio=0.5, in_place=True)

    # Update freeze layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    # print layers
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net


def AlexNetRegressNetBody(net, batch_size, source, type, train=False):
    if train:
        kwargs0 = {'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')), 'shuffle': True}
    else:
        kwargs0 = {'include': dict(phase=caffe_pb2.Phase.Value('TEST')), 'shuffle': False}

    kwargs = {'stride': 1,
              'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
              'weight_filler': dict(type='xavier'),
              'bias_filler': dict(type='constant', value=0.001)}

    net.data, net.label = L.HDF5Data(hdf5_data_param=dict(batch_size=batch_size, source=source), ntop=2, **kwargs0)

    net.conv1 = L.Convolution(net.data, num_output=256, pad=2, kernel_size=5, **kwargs)
    net.relu1 = L.ReLU(net.conv1, in_place=True)
    net.conv2 = L.Convolution(net.relu1, num_output=256, pad=2, kernel_size=5, **kwargs)
    net.relu2 = L.ReLU(net.conv2, in_place=True)
    net.conv3 = L.Convolution(net.relu2, num_output=256, pad=2, kernel_size=5, **kwargs)
    net.relu3 = L.ReLU(net.conv3, in_place=True)
    net.conv4 = L.Convolution(net.relu3, num_output=256, pad=2, kernel_size=5, **kwargs)
    net.relu4 = L.ReLU(net.conv4, in_place=True)
    net.conv5 = L.Convolution(net.relu4, num_output=256, pad=2, kernel_size=5, **kwargs)
    net.relu5 = L.ReLU(net.conv5, in_place=True)
#     net.conv6 = L.Convolution(net.relu5, num_output=256, pad=2, kernel_size=5, **kwargs)
#     net.relu6 = L.ReLU(net.conv6, in_place=True)
#     net.conv7 = L.Convolution(net.relu6, num_output=256, pad=2, kernel_size=5, **kwargs)
#     net.relu7 = L.ReLU(net.conv7, in_place=True)

    net.fc1 = L.Convolution(net.relu5, num_output=1024, pad=6, kernel_size=5, dilation=3, **kwargs)
    net.relu8 = L.ReLU(net.fc1, in_place=True)
    net.fc2 = L.Convolution(net.relu8, num_output=256, kernel_size=1, **kwargs)

    net.loss = L.EuclideanLoss(net.fc2, net.label)
    #net.loss = L.Python(net.fc2, net.label, module='robustL2', layer='RobustL2L1LossLayer', ntop=1, loss_weight=1)
    return net


def VGGRegressNetBody(net, batch_size, source, type, train=False):
    if train:
        kwargs0 = {'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')), 'shuffle': True}
    else:
        kwargs0 = {'include': dict(phase=caffe_pb2.Phase.Value('TEST')), 'shuffle': False}

    kwargs = {'stride': 1,
              'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
              'weight_filler': dict(type='xavier'),
              'bias_filler': dict(type='constant', value=0.001)}

    net.data, net.label = L.HDF5Data(hdf5_data_param=dict(batch_size=batch_size, source=source), ntop=2, **kwargs0)

    net.conv1 = L.Convolution(net.data, num_output=256, pad=2, kernel_size=5, **kwargs)
    net.relu1 = L.ReLU(net.conv1, in_place=True)
    net.conv2 = L.Convolution(net.relu1, num_output=256, pad=2, kernel_size=5, **kwargs)
    net.relu2 = L.ReLU(net.conv2, in_place=True)
    net.conv3 = L.Convolution(net.relu2, num_output=256, pad=2, kernel_size=5, **kwargs)
    net.relu3 = L.ReLU(net.conv3, in_place=True)
    net.conv4 = L.Convolution(net.relu3, num_output=256, pad=2, kernel_size=5, **kwargs)
    net.relu4 = L.ReLU(net.conv4, in_place=True)
    net.conv5 = L.Convolution(net.relu4, num_output=256, pad=2, kernel_size=5, **kwargs)
    net.relu5 = L.ReLU(net.conv5, in_place=True)
    net.conv6 = L.Convolution(net.relu5, num_output=256, pad=2, kernel_size=5, **kwargs)
    net.relu6 = L.ReLU(net.conv6, in_place=True)
    net.conv7 = L.Convolution(net.relu6, num_output=256, pad=2, kernel_size=5, **kwargs)
    net.relu7 = L.ReLU(net.conv7, in_place=True)

    net.fc1 = L.Convolution(net.relu7, num_output=1024, pad=6, kernel_size=5, dilation=3, **kwargs)
    net.relu8 = L.ReLU(net.fc1, in_place=True)
    net.fc2 = L.Convolution(net.relu8, num_output=256, kernel_size=1, **kwargs)

    net.loss = L.EuclideanLoss(net.fc2, net.label)
    #net.loss = L.Python(net.fc2, net.label, module='robustL2', layer='RobustL2L1LossLayer', ntop=1, loss_weight=1)
    return net




def CreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
                       use_objectness=False, normalizations=[], use_batchnorm=True, lr_mult=1,
                       use_scale=True, min_sizes=[], max_sizes=[], prior_variance=[0.1],
                       aspect_ratios=[], steps=[], img_height=0, img_width=0, share_location=True,
                       flip=True, clip=True, offset=0.5, inter_layer_depth=[], kernel_size=1, pad=0,
                       conf_postfix='', loc_postfix='', **bn_param):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"
    if inter_layer_depth:
        assert len(from_layers) == len(inter_layer_depth), "from_layers and inter_layer_depth should have same length"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    objectness_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer],
                                             scale_filler=dict(type="constant", value=normalizations[i]),
                                             across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth:
            if inter_layer_depth[i] > 0:
                inter_name = "{}_inter".format(from_layer)
                ConvBNLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True, lr_mult=lr_mult,
                            num_output=inter_layer_depth[i], kernel_size=3, pad=1, stride=1, **bn_param)
                from_layer = inter_name

        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]

        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4;
        if not share_location:
            num_loc_output *= num_classes
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                    num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
        num_conf_output = num_priors_per_location * num_classes;
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                    num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_size,
                               clip=clip, variance=prior_variance, offset=offset)
        if max_size:
            net.update(name, {'max_size': max_size})
        if aspect_ratio:
            net.update(name, {'aspect_ratio': aspect_ratio, 'flip': flip})
        if step:
            net.update(name, {'step': step})
        if img_height != 0 and img_width != 0:
            if img_height == img_width:
                net.update(name, {'img_size': img_height})
            else:
                net.update(name, {'img_h': img_height, 'img_w': img_width})
        priorbox_layers.append(net[name])

        # Create objectness prediction layer.
        if use_objectness:
            name = "{}_mbox_objectness".format(from_layer)
            num_obj_output = num_priors_per_location * 2;
            ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                        num_output=num_obj_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layers.append(net[flatten_name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    if use_objectness:
        name = "mbox_objectness"
        net[name] = L.Concat(*objectness_layers, axis=1)
        mbox_layers.append(net[name])

    return mbox_layers
