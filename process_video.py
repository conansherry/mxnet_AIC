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
import re

import logging
# set up logger
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='For AIC Champion Test')
parser.add_argument('--network', default='vgg', help='network type', type=str)
parser.add_argument('--prefix', help='model to test with', type=str)
parser.add_argument('--dataset', help='dataset', type=str)
parser.add_argument('--data_output', help='dataset', type=str)
parser.add_argument('--epoch', help='model to test with', type=int)
parser.add_argument('--gpu', help='GPU device to test with', default=0, type=int)
parser.add_argument('--s', default=0, type=int, metavar='N',
                    help='start test number')
parser.add_argument('--e', default=100, type=int, metavar='N',
                    help='end test number')
parser.add_argument('-v', '--visual', dest='visual', action='store_true',
                    help='show results')

# Test Config
parser.add_argument('--octave', default=6, type=int, metavar='N',
                    help='scale number')
parser.add_argument('--starting_range', default=0.5, type=float, metavar='F',
                    help='start scale')
parser.add_argument('--ending_range', default=1.2, type=float, metavar='F',
                    help='end scale')
parser.add_argument('--flip', help='flip test', action='store_true')
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

cap = cv2.VideoCapture()

pattern = re.compile(r'.*mp4')
video_list = []
for dirpath, dirnames, filenames in os.walk(args.dataset):
    for filename in filenames:
        match = pattern.match(filename)
        if match:
            video_list.append(filename[:-4])

video_index = 0
cap.open(os.path.join(args.dataset, video_list[video_index] + '.mp4'))
video_name = video_list[video_index]
video_index += 1
output_dir = args.data_output

f = 0
while True:
    tic = time.time()
    # oriImg = cv2.imread(image_files[pic_index[f]])
    go_next = False
    for c in range(10):
        ret, oriImg = cap.read()
        if oriImg is None:
            go_next = True
            break
    if go_next:
        if video_index >= len(video_list):
            break
        video_name = video_list[video_index]
        cap.open(os.path.join(args.dataset, video_list[video_index] + '.mp4'))
        video_index += 1
        f = 0
        continue

    pic_id = 'dance_' + str(video_name) + '_' + str(f)
    cv2.imwrite(os.path.join(output_dir, pic_id + '.jpg'), oriImg)

    f += 1

    heatmap_avg, paf_avg = multiscale_cnn_forward(oriImg, mod, args, arg_params, aux_params, args, None)
    candidate, subset = connect_aic_LineVec(oriImg, heatmap_avg, paf_avg, args, None)

    predictions = dict()
    predictions['image_id'] = pic_id
    predictions['keypoint_annotations'] = dict()
    predictions['human_annotations'] = dict()
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
        temp_reshape = temp.reshape((-1, 3))
        valid_index = np.where(temp_reshape[:, 2] != 0)[0]
        valid_keypoint = temp_reshape[valid_index, :2].astype(np.float32)
        box = cv2.boundingRect(np.expand_dims(valid_keypoint, axis=0))
        predictions['human_annotations']['human' + str(p + 1)] = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        predictions['keypoint_annotations']['human' + str(p + 1)] = temp.astype(int).tolist()

    for k, v in predictions['human_annotations'].items():
        cv2.rectangle(oriImg, (v[0], v[1]), (v[2], v[3]), (255, 0, 0), 2)

    for k, v in predictions['keypoint_annotations'].items():
        keypoint = np.array(v)
        keypoint = keypoint.reshape((-1, 3)).astype(np.int32)
        for j in range(keypoint.shape[0]):
            if keypoint[j, 2] != 0:
                cv2.putText(oriImg, str(j), (keypoint[j, 0], keypoint[j, 1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                cv2.circle(oriImg, (keypoint[j, 0], keypoint[j, 1]), 1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, 'draw_' + pic_id + '.jpg'), oriImg)
    with open(os.path.join(output_dir, pic_id + '.json'), 'w') as outfile:
        json.dump(predictions, outfile)

    cv2.imshow('oriImg', oriImg)
    cv2.waitKey(1)

    print('Process %5d / %s \t time:%1.2fs' % (f, video_name, time.time() - tic))


