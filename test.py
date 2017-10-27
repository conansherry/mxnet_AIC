import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import time
import argparse
import glob
import json
import symbol_resnet
import symbol_vgg
import cv2
import mxnet as mx
from testutils import *
from utils import *
import random

import logging
# set up logger
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='For AIC Champion Test')
parser.add_argument('--network', default='vgg', help='network type', type=str)
parser.add_argument('--prefix', help='model to test with', type=str)
parser.add_argument('--epoch', help='model to test with', type=int)
parser.add_argument('--gpu', help='GPU device to test with', default=1, type=int)
parser.add_argument('--dataset', metavar='DATA', help='path to dataset')
parser.add_argument('--o','--outputjson', dest='outputjson',
                    default='outputjson.json', metavar='FILE', help='file to save result')
parser.add_argument('--s', default=0, type=int, metavar='N',
                    help='start test number')
parser.add_argument('--e', default=500, type=int, metavar='N',
                    help='end test number')
parser.add_argument('-v', '--visual', dest='visual', action='store_true',
                    help='show results')

# Test Config
parser.add_argument('--octave', default=1, type=int, metavar='N',
                    help='scale number')
parser.add_argument('--starting_range', default=1, type=float, metavar='F',
                    help='start scale')
parser.add_argument('--ending_range', default=1.2, type=float, metavar='F',
                    help='end scale')
parser.add_argument('--thre1', default=0.1, type=float, metavar='F',
                    help='threshold for peak')
parser.add_argument('--thre2', default=0.1, type=float, metavar='F',
                    help='threshold for vector')
parser.add_argument('--thre3', default=0.5, type=float, metavar='F',
                    help='threshold for subset')

parser.add_argument('--boxsize', default=368, type=int, metavar='N',
                    help='net input size')
parser.add_argument('--stride', default=8, type=int, metavar='N',
                    help='net stride')
parser.add_argument('--padValue', default=127, type=int, metavar='N',
                    help='padValue for stride x pad')

args = parser.parse_args()

# ImageFile
anno_file = os.path.join(args.dataset, 'keypoint_validation_annotations_20170911.json')
if 'test' not in args.dataset and os.path.exists(anno_file):  # VAL MODE
    anno = json.load(open(anno_file, 'r'))
    image_ids = [a['image_id'] for a in anno]
    keypoints_anno = [a['keypoint_annotations'] for a in anno]
    image_files = [os.path.join(args.dataset, 'keypoint_validation_images_20170911', f+'.jpg') for f in image_ids]
else:  # IMAGE MODE
    image_files = glob.glob(os.path.join(args.dataset, '*.jpg'))
    image_ids = [os.path.basename(path)[:-4] for path in image_files]

num_test = len(image_files)
start_f = min(max(args.s, 0), num_test)
end_f = min(max(args.e, start_f), num_test)

ctx = mx.gpu(args.gpu)
if args.network == 'vgg':
    sym = symbol_vgg.get_vgg_test()
elif args.network == 'resnet':
    sym = symbol_resnet.get_resnet_test()
arg_params, aux_params = load_param(args.prefix, args.epoch, convert=True, ctx=ctx)

# infer shape
data_shape_dict = {'data': (1, 3, 368, 368)}
arg_shape, _, aux_shape = sym.infer_shape(**data_shape_dict)
arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))

# check parameters
for k in sym.list_arguments():
    if k in data_shape_dict:
        continue
    assert k in arg_params, k + ' not initialized'
    assert arg_params[k].shape == arg_shape_dict[k], \
        'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
for k in sym.list_auxiliary_states():
    assert k in aux_params, k + ' not initialized'
    assert aux_params[k].shape == aux_shape_dict[k], \
        'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)

mod = mx.mod.Module(sym, data_names=('data', ), label_names=None, logger=logger, context=ctx)
mod.bind([('data', (1, 3, 368, 368))], for_training=False, force_rebind=True)
mod.init_params(arg_params=arg_params, aux_params=aux_params)

res = []
pic_index = range(len(image_files))
random.shuffle(pic_index)
for f in range(start_f, end_f):
    tic = time.time()
    oriImg = cv2.imread(image_files[pic_index[f]])

    heatmap_avg, paf_avg = multiscale_cnn_forward(oriImg, mod, args, arg_params, aux_params, args, keypoints_anno[pic_index[f]])
    candidate, subset = connect_aic_LineVec(oriImg, heatmap_avg, paf_avg, args, keypoints_anno[pic_index[f]])

    # gt_batch_img = cvim_with_heatmap(oriImg, heatmap_avg, num_rows=4)
    # cv2.imshow('test1', cv2.cvtColor(gt_batch_img, cv2.COLOR_RGB2BGR))
    # gt_batch_img = cvim_with_vectormap(oriImg, paf_avg, num_rows=4)
    # cv2.imshow('test2', cv2.cvtColor(gt_batch_img, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

    predictions = dict()
    predictions['image_id'] = image_ids[pic_index[f]]
    predictions['keypoint_annotations'] = dict()
    for p in range(len(subset)):
        temp = np.zeros(3 * 14)
        for i in range(14):
            index = subset[p][i, ].astype(int)
            if -1 == index:
                continue
            X = candidate[index, 0]
            Y = candidate[index, 1]
            temp[3 * i] = X.astype(int)
            temp[3 * i + 1] = Y.astype(int)
            temp[3 * i + 2] = 1
        predictions['keypoint_annotations']['human' + str(p + 1)] = temp.astype(int).tolist()
    res.append(predictions)

    print('Process %5d / %5d \t time:%1.2fs' % (f + 1, num_test, time.time() - tic))

with open(args.outputjson, 'w') as outfile:
    json.dump(res, outfile)


