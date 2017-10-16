import sys, os
import argparse
import pprint
import mxnet as mx
import mxnet.metric
import numpy as np
import random
import symbol_vgg
from data import FileIter
from metric import AICRMSE

import logging
# set up logger
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def load_checkpoint(prefix, epoch):
    """
    Load model checkpoint from file.
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params

def convert_context(params, ctx):
    """
    :param params: dict of str to NDArray
    :param ctx: the context to convert to
    :return: dict of str of NDArray with context ctx
    """
    new_params = dict()
    for k, v in params.items():
        new_params[k] = v.as_in_context(ctx)
    return new_params

def load_param(prefix, epoch, convert=False, ctx=None):
    """
    wrapper for load checkpoint
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :param convert: reference model should be converted to GPU NDArray first
    :param ctx: if convert then ctx must be designated.
    :param process: model should drop any test
    :return: (arg_params, aux_params)
    """
    arg_params, aux_params = load_checkpoint(prefix, epoch)
    if convert:
        if ctx is None:
            ctx = mx.cpu()
        arg_params = convert_context(arg_params, ctx)
        aux_params = convert_context(aux_params, ctx)
    return arg_params, aux_params


def train_net(args, ctx, pretrained, epoch, prefix, lr=0.001):
    root_dir = args.dataset
    train_dir = os.path.join(root_dir, "keypoint_train_images_20170902")
    anno_file = os.path.join(root_dir, "keypoint_train_annotations_20170909.json")

    train_data = FileIter(train_dir, anno_file)
    sym = symbol_vgg.get_vgg_train()

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
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(
                arg_params[k].shape)
    for k in sym.list_auxiliary_states():
        if k not in aux_params:
            print('reset not in model params ' + k)
            aux_params[k] = mx.nd.zeros(shape=aux_shape_dict[k])
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(
                aux_params[k].shape)

    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]
    mod = mx.mod.Module(symbol=sym, data_names=data_names, label_names=label_names, context=ctx, logger=logger)

    batch_end_callback = mx.callback.Speedometer(train_data.batch_size, frequent=10, auto_reset=False)
    epoch_end_callback = mx.callback.do_checkpoint(prefix)

    eval_metrics = mx.metric.CompositeEvalMetric()
    for branch in range(1, 3):
        for stage in range(1, 7):
            eval_metrics.add(AICRMSE(stage=stage, branch=branch))

    # optimizer

    optimizer_params = {'learning_rate': lr}

    mod.fit(train_data, epoch_end_callback=epoch_end_callback, batch_end_callback=batch_end_callback,
            eval_metric=eval_metrics,
            optimizer='adam', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, num_epoch=100)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Pose')
    # general
    parser.add_argument('--dataset', help='dataset dir', type=str)
    parser.add_argument('--gpus', help='GPU device to train with', default='0', type=str)
    parser.add_argument('--pretrained', help='pretrained model prefix', type=str)
    parser.add_argument('--pretrained_epoch', help='pretrained model epoch', type=int)
    parser.add_argument('--prefix', help='new model prefix', type=str)
    parser.add_argument('--lr', help='base learning rate', default=0.0001, type=float)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print('Called with argument: %s' % args)
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    train_net(args, ctx, args.pretrained, args.pretrained_epoch, args.prefix, lr=args.lr)

if __name__ == '__main__':
    main()
