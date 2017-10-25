import os
import time
import argparse
import glob
import json
import cv2
import caffe
from testutils_caffe import *

import logging
# set up logger
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='For AIC Champion Test')
parser.add_argument('--network', metavar='NET', help='path to network file')
parser.add_argument('--prefix', help='model to test with', type=str)
parser.add_argument('--epoch', help='model to test with', type=int)
parser.add_argument('--gpu', help='GPU device to test with', default=1, type=int)
parser.add_argument('--dataset', metavar='DATA', help='path to dataset')
parser.add_argument('--o','--outputjson', dest='outputjson',
                    default='outputjson.json', metavar='FILE', help='file to save result')
parser.add_argument('--s', default=0, type=int, metavar='N',
                    help='start test number')
parser.add_argument('--e', default=10, type=int, metavar='N',
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
parser.add_argument('--thre2', default=0.05, type=float, metavar='F',
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
    image_files = [os.path.join(args.dataset, 'keypoint_validation_images_20170911', f+'.jpg') for f in image_ids]
else:  # IMAGE MODE
    image_files = glob.glob(os.path.join(args.dataset, '*.jpg'))
    image_ids = [os.path.basename(path)[:-4] for path in image_files]

num_test = len(image_files)
start_f = min(max(args.s, 0), num_test)
end_f = min(max(args.e, start_f), num_test)

caffe.set_mode_gpu()
prototxt = r'model_convert/vgg_test_network.prototxt'
caffemodel = r'model_convert/vgg_test_network.caffemodel'
prototxt = r'model_convert/pose_deploy_coco.prototxt'
caffemodel = r'model_convert/pose_coco_iter_2000.caffemodel'
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

res = []
for f in range(start_f, end_f):
    tic = time.time()
    oriImg = cv2.imread(image_files[f])

    heatmap_avg, paf_avg = multiscale_cnn_forward(oriImg, net, args)
    candidate, subset = connect_aic_LineVec(oriImg, heatmap_avg, paf_avg, args)

    # gt_batch_img = cvim_with_heatmap(oriImg, heatmap_avg, num_rows=4)
    # cv2.imshow('test1', cv2.cvtColor(gt_batch_img, cv2.COLOR_RGB2BGR))
    # gt_batch_img = cvim_with_vectormap(oriImg, paf_avg, num_rows=4)
    # cv2.imshow('test2', cv2.cvtColor(gt_batch_img, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

    predictions = dict()
    predictions['image_id'] = image_ids[f]
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


