import sys, os
import argparse
import pprint
import mxnet as mx
import mxnet.metric
import numpy as np
import random
import symbol_vgg
import symbol_resnet
from data import FileIter
from metric import AICRMSE
from utils import *
from module import CustomModule

import logging
# set up logger
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def train_net(args, ctx, pretrained, epoch, prefix, lr=0.001):
    root_dir = args.dataset
    train_dir = os.path.join(root_dir, "keypoint_train_images_20170902")
    anno_file = os.path.join(root_dir, "keypoint_train_annotations_20170909.json")

    if args.network == 'vgg':
        mean_value = 127
        div_num = 255.
    else:
        mean_value = 0
        div_num = 1.
    train_data = FileIter(train_dir, anno_file, batch_size=args.batch_size, no_shuffle=args.no_shuffle, mean_value=mean_value, div_num=div_num)
    if args.network == 'vgg':
        sym = symbol_vgg.get_vgg_train()
    else:
        sym = symbol_resnet.get_resnet_train()

    arg_params, aux_params = load_param(pretrained, epoch, convert=True)

    # infer shape
    data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
    arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    out_shape_dict = dict(zip(sym.list_outputs(), out_shape))
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))
    print('output shape %s' % pprint.pformat(out_shape_dict))

    # check parameter shapes
    for k in sym.list_arguments():
        if k in data_shape_dict:
            continue
        if k not in arg_params:
            print('reset not in model params ' + k)
            arg_params[k] = mx.random.normal(0, 0.01, shape=arg_shape_dict[k])
        assert k in arg_params, k + ' not initialized'
        if arg_params[k].shape != arg_shape_dict[k]:
            arg_params[k] = mx.random.normal(0, 0.01, shape=arg_shape_dict[k])
            print 'need init', k
        # assert arg_params[k].shape == arg_shape_dict[k], \
        #     'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(
        #         arg_params[k].shape)
    for k in sym.list_auxiliary_states():
        if k not in aux_params:
            print('reset not in model params ' + k)
            aux_params[k] = mx.nd.zeros(shape=aux_shape_dict[k])
        assert k in aux_params, k + ' not initialized'
        if aux_params[k].shape != aux_shape_dict[k]:
            aux_params[k] = mx.nd.zeros(shape=aux_shape_dict[k])
            print 'need init', k
        # assert aux_params[k].shape == aux_shape_dict[k], \
        #     'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(
        #         aux_params[k].shape)

    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]

    if args.network == 'resnet':
        fixed_param_prefix = ['conv0', 'stage1', 'gamma', 'beta']
        fixed_param_names = list()
        if fixed_param_prefix is not None:
            for name in sym.list_arguments():
                for prefix_ in fixed_param_prefix:
                    if prefix_ in name:
                        fixed_param_names.append(name)
        print('fix params ' + str(fixed_param_names))
    else:
        fixed_param_prefix = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4']
        fixed_param_prefix = []
        fixed_param_names = list()
        if fixed_param_prefix is not None:
            for name in sym.list_arguments():
                for prefix_ in fixed_param_prefix:
                    if prefix_ in name:
                        fixed_param_names.append(name)
        print('fix params ' + str(fixed_param_names))
    mod = CustomModule(symbol=sym, data_names=data_names, label_names=label_names, context=ctx, logger=logger, fixed_param_names=fixed_param_names)

    batch_end_callback = mx.callback.Speedometer(train_data.batch_size, frequent=10, auto_reset=False)
    epoch_end_callback = mx.callback.do_checkpoint(prefix)

    eval_metrics = mx.metric.CompositeEvalMetric()
    for branch in range(1, 3):
        for stage in range(1, 7):
            eval_metrics.add(AICRMSE(train_data.batch_size / len(ctx), stage=stage, branch=branch))

    # optimizer
    optimizer_params = {'learning_rate': lr, 'lr_scheduler': mx.lr_scheduler.FactorScheduler(100000, factor=0.3), 'rescale_grad': 1.0 / train_data.batch_size / len(ctx)}

    mod.fit(train_data, epoch_end_callback=epoch_end_callback, batch_end_callback=batch_end_callback,
            eval_metric=eval_metrics,
            optimizer=mx.optimizer.RMSProp(learning_rate=lr), optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, num_epoch=100)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Pose')
    # general
    parser.add_argument('--dataset', help='dataset dir', type=str)
    parser.add_argument('--network', help='network type', default='vgg', type=str)
    parser.add_argument('--gpus', help='GPU device to train with', default='0', type=str)
    parser.add_argument('--pretrained', help='pretrained model prefix', type=str)
    parser.add_argument('--pretrained_epoch', help='pretrained model epoch', type=int)
    parser.add_argument('--prefix', help='new model prefix', type=str)
    parser.add_argument('--lr', help='base learning rate', default=0.00004, type=float)
    parser.add_argument('--no_shuffle', help='disable random shuffle', action='store_true')
    parser.add_argument('--batch_size', help='batch size per gpu', default=10, type=int)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print('Called with argument: %s' % args)
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    train_net(args, ctx, args.pretrained, args.pretrained_epoch, args.prefix, lr=args.lr)

if __name__ == '__main__':
    main()
