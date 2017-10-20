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

def get_vgg_conv(data):
    """
    shared convolutional layers
    :param data: Symbol
    :return: Symbol
    """
    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")

    return relu2_2, relu3_3, relu4_3

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

def get_vgg_train():
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="l2_label")

    partaffinityglabel = mx.symbol.slice_axis(label, axis=1, begin=0, end=26)
    heatmaplabel = mx.symbol.slice_axis(label, axis=1, begin=26, end=41)
    partaffinityglabel_reshape = mx.symbol.Reshape(data=partaffinityglabel, shape=(-1,),
                                                   name='partaffinityglabel_reshape')
    heatmaplabel_reshape = mx.symbol.Reshape(data=heatmaplabel, shape=(-1,), name='heatmaplabel_reshape')

    relu2_2, relu3_3, relu4_3 = get_vgg_conv(data)

    conv4_4_CPM = mx.symbol.Convolution(name='conv4_4_CPM', data=relu4_3, num_filter=256, pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=False)
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
                             # , mx.sym.BlockGrad(data), mx.sym.BlockGrad(stage6_l1), mx.sym.BlockGrad(stage6_l2),
                             # mx.sym.BlockGrad(relu4_5_CPM)])
    return group

def get_vgg_test():
    data = mx.symbol.Variable(name="data")

    relu2_2, relu3_3, relu4_3 = get_vgg_conv(data)

    conv4_4_CPM = mx.symbol.Convolution(name='conv4_4_CPM', data=relu4_3, num_filter=256, pad=(1, 1), kernel=(3, 3),
                                        stride=(1, 1), no_bias=False)
    relu4_4_CPM = mx.symbol.Activation(name='relu4_4_CPM', data=conv4_4_CPM, act_type='relu')
    conv4_5_CPM = mx.symbol.Convolution(name='conv4_5_CPM', data=relu4_4_CPM, num_filter=128, pad=(1, 1), kernel=(3, 3),
                                        stride=(1, 1), no_bias=False)
    relu4_5_CPM = mx.symbol.Activation(name='relu4_5_CPM', data=conv4_5_CPM, act_type='relu')

    stage1_l1, stage1_l2 = get_stage_1(relu4_5_CPM)

    stage2_l1, stage2_l2 = get_stage_n(relu4_5_CPM, stage1_l1, stage1_l2, 2)
    stage3_l1, stage3_l2 = get_stage_n(relu4_5_CPM, stage2_l1, stage2_l2, 3)
    stage4_l1, stage4_l2 = get_stage_n(relu4_5_CPM, stage3_l1, stage3_l2, 4)
    stage5_l1, stage5_l2 = get_stage_n(relu4_5_CPM, stage4_l1, stage4_l2, 5)
    stage6_l1, stage6_l2 = get_stage_n(relu4_5_CPM, stage5_l1, stage5_l2, 6)

    return mx.symbol.Group([stage6_l1, stage6_l2])

if __name__ == "__main__":
    network = get_vgg_test()
    network.save('vgg_test_network.json')

    tmp = mx.viz.plot_network(network, shape={'data': (1, 3, 368, 368)})
    tmp.view()
