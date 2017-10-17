# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx

eps = 2e-5
use_global_stats = True
workspace = 512
res_type = '101'
res_deps = {'34': (3, 4, 6, 3), '50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}
units = res_deps[res_type]
if res_type != '34':
    filter_list = [256, 512, 1024, 2048]
else:
    filter_list = [64, 128, 256, 512]

def residual_unit(data, num_filter, stride, dim_match, name):
    if res_type == '34':
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter), kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter), kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc')
        sum = mx.sym.ElementWiseSum(*[conv2, shortcut], name=name + '_plus')
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats,
                               name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                                   pad=(0, 0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats,
                               name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride,
                                   pad=(1, 1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats,
                               name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name + '_sc')
        sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')
    return sum

def get_resnet_conv(data):
    # res1
    data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='bn_data')
    conv0 = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                               no_bias=True, name="conv0", workspace=workspace)
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    # res2
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1')
    for i in range(2, units[0] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True, name='stage1_unit%s' % i)

    # res3
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1')
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True, name='stage2_unit%s' % i)

    # res4
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=False, name='stage3_unit1')
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True, name='stage3_unit%s' % i)
    return unit

def get_stage_1(conv_feat):
    conv5_1_CPM_L1 = mx.symbol.Convolution(name='conv5_1_CPM_L1', data=conv_feat, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_1_CPM_L1 = mx.symbol.Activation(name='relu5_1_CPM_L1', data=conv5_1_CPM_L1, act_type='relu')
    conv5_1_CPM_L2 = mx.symbol.Convolution(name='conv5_1_CPM_L2', data=conv_feat, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_1_CPM_L2 = mx.symbol.Activation(name='relu5_1_CPM_L2', data=conv5_1_CPM_L2, act_type='relu')
    conv5_2_CPM_L1 = mx.symbol.Convolution(name='conv5_2_CPM_L1', data=relu5_1_CPM_L1, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_2_CPM_L1 = mx.symbol.Activation(name='relu5_2_CPM_L1', data=conv5_2_CPM_L1, act_type='relu')
    conv5_2_CPM_L2 = mx.symbol.Convolution(name='conv5_2_CPM_L2', data=relu5_1_CPM_L2, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_2_CPM_L2 = mx.symbol.Activation(name='relu5_2_CPM_L2', data=conv5_2_CPM_L2, act_type='relu')
    conv5_3_CPM_L1 = mx.symbol.Convolution(name='conv5_3_CPM_L1', data=relu5_2_CPM_L1, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_3_CPM_L1 = mx.symbol.Activation(name='relu5_3_CPM_L1', data=conv5_3_CPM_L1, act_type='relu')
    conv5_3_CPM_L2 = mx.symbol.Convolution(name='conv5_3_CPM_L2', data=relu5_2_CPM_L2, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_3_CPM_L2 = mx.symbol.Activation(name='relu5_3_CPM_L2', data=conv5_3_CPM_L2, act_type='relu')
    conv5_4_CPM_L1 = mx.symbol.Convolution(name='conv5_4_CPM_L1', data=relu5_3_CPM_L1, num_filter=512, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=False)
    relu5_4_CPM_L1 = mx.symbol.Activation(name='relu5_4_CPM_L1', data=conv5_4_CPM_L1, act_type='relu')
    conv5_4_CPM_L2 = mx.symbol.Convolution(name='conv5_4_CPM_L2', data=relu5_3_CPM_L2, num_filter=512, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=False)
    relu5_4_CPM_L2 = mx.symbol.Activation(name='relu5_4_CPM_L2', data=conv5_4_CPM_L2, act_type='relu')
    conv5_5_CPM_L1 = mx.symbol.Convolution(name='conv5_5_CPM_L1', data=relu5_4_CPM_L1, num_filter=26, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=False)
    conv5_5_CPM_L2 = mx.symbol.Convolution(name='conv5_5_CPM_L2', data=relu5_4_CPM_L2, num_filter=15, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=False)

    return conv5_5_CPM_L1, conv5_5_CPM_L2

def get_stage_n(conv_feat, pre_l1, pre_l2, N):
    concat_feat = mx.symbol.Concat(name='concat_stage_' + str(N), *[pre_l1, pre_l2, conv_feat])

    Mconv1_stage_L1 = mx.symbol.Convolution(name='Mconv1_stage%s_L1' % N, data=concat_feat, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage_L1 = mx.symbol.Activation(name='Mrelu1_stage%s_L1' % N, data=Mconv1_stage_L1, act_type='relu')
    Mconv1_stage_L2 = mx.symbol.Convolution(name='Mconv1_stage%s_L2' % N, data=concat_feat, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage_L2 = mx.symbol.Activation(name='Mrelu1_stage%s_L2' % N, data=Mconv1_stage_L2, act_type='relu')
    Mconv2_stage_L1 = mx.symbol.Convolution(name='Mconv2_stage%s_L1' % N, data=Mrelu1_stage_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage_L1 = mx.symbol.Activation(name='Mrelu2_stage%s_L1' % N, data=Mconv2_stage_L1, act_type='relu')
    Mconv2_stage_L2 = mx.symbol.Convolution(name='Mconv2_stage%s_L2' % N, data=Mrelu1_stage_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage_L2 = mx.symbol.Activation(name='Mrelu2_stage%s_L2' % N, data=Mconv2_stage_L2, act_type='relu')
    Mconv3_stage_L1 = mx.symbol.Convolution(name='Mconv3_stage%s_L1' % N, data=Mrelu2_stage_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage_L1 = mx.symbol.Activation(name='Mrelu3_stage%s_L1' % N, data=Mconv3_stage_L1, act_type='relu')
    Mconv3_stage_L2 = mx.symbol.Convolution(name='Mconv3_stage%s_L2' % N, data=Mrelu2_stage_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage_L2 = mx.symbol.Activation(name='Mrelu3_stage%s_L2' % N, data=Mconv3_stage_L2, act_type='relu')
    Mconv4_stage_L1 = mx.symbol.Convolution(name='Mconv4_stage%s_L1' % N, data=Mrelu3_stage_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage_L1 = mx.symbol.Activation(name='Mrelu4_stage%s_L1' % N, data=Mconv4_stage_L1, act_type='relu')
    Mconv4_stage_L2 = mx.symbol.Convolution(name='Mconv4_stage%s_L2' % N, data=Mrelu3_stage_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage_L2 = mx.symbol.Activation(name='Mrelu4_stage%s_L2' % N, data=Mconv4_stage_L2, act_type='relu')
    Mconv5_stage_L1 = mx.symbol.Convolution(name='Mconv5_stage%s_L1' % N, data=Mrelu4_stage_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage_L1 = mx.symbol.Activation(name='Mrelu5_stage%s_L1' % N, data=Mconv5_stage_L1, act_type='relu')
    Mconv5_stage_L2 = mx.symbol.Convolution(name='Mconv5_stage%s_L2' % N, data=Mrelu4_stage_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage_L2 = mx.symbol.Activation(name='Mrelu5_stage%s_L2' % N, data=Mconv5_stage_L2, act_type='relu')
    Mconv6_stage_L1 = mx.symbol.Convolution(name='Mconv6_stage%s_L1' % N, data=Mrelu5_stage_L1, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage_L1 = mx.symbol.Activation(name='Mrelu6_stage%s_L1' % N, data=Mconv6_stage_L1, act_type='relu')
    Mconv6_stage_L2 = mx.symbol.Convolution(name='Mconv6_stage%s_L2' % N, data=Mrelu5_stage_L2, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage_L2 = mx.symbol.Activation(name='Mrelu6_stage%s_L2' % N, data=Mconv6_stage_L2, act_type='relu')
    Mconv7_stage_L1 = mx.symbol.Convolution(name='Mconv7_stage%s_L1' % N, data=Mrelu6_stage_L1, num_filter=26, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mconv7_stage_L2 = mx.symbol.Convolution(name='Mconv7_stage%s_L2' % N, data=Mrelu6_stage_L2, num_filter=15, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)

    return Mconv7_stage_L1, Mconv7_stage_L2

def get_resnet_train():
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="l2_label")

    partaffinityglabel = mx.symbol.slice_axis(label, axis=1, begin=0, end=26)
    heatmaplabel = mx.symbol.slice_axis(label, axis=1, begin=26, end=41)
    partaffinityglabel_reshape = mx.symbol.Reshape(data=partaffinityglabel, shape=(-1,),
                                                   name='partaffinityglabel_reshape')
    heatmaplabel_reshape = mx.symbol.Reshape(data=heatmaplabel, shape=(-1,), name='heatmaplabel_reshape')

    res_feat = get_resnet_conv(data)

    conv4_4_CPM = mx.symbol.Convolution(name='conv4_4_CPM', data=res_feat, num_filter=256, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu4_4_CPM = mx.symbol.Activation(name='relu4_4_CPM', data=conv4_4_CPM, act_type='relu')
    conv4_5_CPM = mx.symbol.Convolution(name='conv4_5_CPM', data=relu4_4_CPM, num_filter=128, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu4_5_CPM = mx.symbol.Activation(name='relu4_5_CPM', data=conv4_5_CPM, act_type='relu')

    stage1_l1, stage1_l2 = get_stage_1(relu4_5_CPM)

    stage2_l1, stage2_l2 = get_stage_n(relu4_5_CPM, stage1_l1, stage1_l2, 2)
    stage3_l1, stage3_l2 = get_stage_n(relu4_5_CPM, stage2_l1, stage2_l2, 3)
    stage4_l1, stage4_l2 = get_stage_n(relu4_5_CPM, stage3_l1, stage3_l2, 4)
    stage5_l1, stage5_l2 = get_stage_n(relu4_5_CPM, stage4_l1, stage4_l2, 5)
    stage6_l1, stage6_l2 = get_stage_n(relu4_5_CPM, stage5_l1, stage5_l2, 6)

    ##################
    stage1_l1_reshape = mx.symbol.Reshape(data=stage1_l1, shape=(-1,), name='stage1_l1_reshape')
    stage1_l1_square = mx.symbol.square(stage1_l1_reshape - partaffinityglabel_reshape)
    stage1_loss_l1 = mx.symbol.MakeLoss(stage1_l1_square)

    stage1_l2_reshape = mx.symbol.Reshape(data=stage1_l2, shape=(-1,), name='stage1_l2_reshape')
    stage1_l2_square = mx.symbol.square(stage1_l2_reshape - heatmaplabel_reshape)
    stage1_loss_l2 = mx.symbol.MakeLoss(stage1_l2_square)

    ##################
    stage2_l1_reshape = mx.symbol.Reshape(data=stage2_l1, shape=(-1,), name='stage2_l1_reshape')
    stage2_l1_square = mx.symbol.square(stage2_l1_reshape - partaffinityglabel_reshape)
    stage2_loss_l1 = mx.symbol.MakeLoss(stage2_l1_square)

    stage2_l2_reshape = mx.symbol.Reshape(data=stage2_l2, shape=(-1,), name='stage2_l2_reshape')
    stage2_l2_square = mx.symbol.square(stage2_l2_reshape - heatmaplabel_reshape)
    stage2_loss_l2 = mx.symbol.MakeLoss(stage2_l2_square)

    ##################
    stage3_l1_reshape = mx.symbol.Reshape(data=stage3_l1, shape=(-1,), name='stage3_l1_reshape')
    stage3_l1_square = mx.symbol.square(stage3_l1_reshape - partaffinityglabel_reshape)
    stage3_loss_l1 = mx.symbol.MakeLoss(stage3_l1_square)

    stage3_l2_reshape = mx.symbol.Reshape(data=stage3_l2, shape=(-1,), name='stage3_l2_reshape')
    stage3_l2_square = mx.symbol.square(stage3_l2_reshape - heatmaplabel_reshape)
    stage3_loss_l2 = mx.symbol.MakeLoss(stage3_l2_square)

    ##################
    stage4_l1_reshape = mx.symbol.Reshape(data=stage4_l1, shape=(-1,), name='stage4_l1_reshape')
    stage4_l1_square = mx.symbol.square(stage4_l1_reshape - partaffinityglabel_reshape)
    stage4_loss_l1 = mx.symbol.MakeLoss(stage4_l1_square)

    stage4_l2_reshape = mx.symbol.Reshape(data=stage4_l2, shape=(-1,), name='stage4_l2_reshape')
    stage4_l2_square = mx.symbol.square(stage4_l2_reshape - heatmaplabel_reshape)
    stage4_loss_l2 = mx.symbol.MakeLoss(stage4_l2_square)

    ##################
    stage5_l1_reshape = mx.symbol.Reshape(data=stage5_l1, shape=(-1,), name='stage5_l1_reshape')
    stage5_l1_square = mx.symbol.square(stage5_l1_reshape - partaffinityglabel_reshape)
    stage5_loss_l1 = mx.symbol.MakeLoss(stage5_l1_square)

    stage5_l2_reshape = mx.symbol.Reshape(data=stage5_l2, shape=(-1,), name='stage5_l2_reshape')
    stage5_l2_square = mx.symbol.square(stage5_l2_reshape - heatmaplabel_reshape)
    stage5_loss_l2 = mx.symbol.MakeLoss(stage5_l2_square)

    ##################
    stage6_l1_reshape = mx.symbol.Reshape(data=stage6_l1, shape=(-1,), name='stage6_l1_reshape')
    stage6_l1_square = mx.symbol.square(stage6_l1_reshape - partaffinityglabel_reshape)
    stage6_loss_l1 = mx.symbol.MakeLoss(stage6_l1_square)

    stage6_l2_reshape = mx.symbol.Reshape(data=stage6_l2, shape=(-1,), name='stage6_l2_reshape')
    stage6_l2_square = mx.symbol.square(stage6_l2_reshape - heatmaplabel_reshape)
    stage6_loss_l2 = mx.symbol.MakeLoss(stage6_l2_square)

    group = mx.symbol.Group([stage1_loss_l1, stage1_loss_l2,
                             stage2_loss_l1, stage2_loss_l2,
                             stage3_loss_l1, stage3_loss_l2,
                             stage4_loss_l1, stage4_loss_l2,
                             stage5_loss_l1, stage5_loss_l2,
                             stage6_loss_l1, stage6_loss_l2])
    return group

if __name__ == "__main__":
    network = get_resnet_train()

    tmp = mx.viz.plot_network(network, shape={'data': (1, 3, 368, 368), 'l2_label': (1, 41, 46, 46)})
    tmp.view()
