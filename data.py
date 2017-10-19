import mxnet as mx
import numpy as np
import sys, os
from mxnet.io import DataIter
from PIL import Image
import cv2
import random
import imutils
import re
import logging
import json
import scipy.stats as st
import cPickle as pickle
import math
import time
import imutils

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def swapleftright(x, width):
    """
    flip coords
    """
    matchedParts = (
        [0, 3], [1, 4], [2, 5],
        [6, 9], [7, 10], [8, 11]
    )

    # Flip horizontal
    x[:, 0] = width - 1 - x[:, 0]

    # Change left-right parts
    for pair in matchedParts:
        tmp = np.copy(x[pair[0], :])
        x[pair[0], :] = x[pair[1], :]
        x[pair[1], :] = tmp

    return x

def putGaussianMaps(img, pt, stride, sigma):
    img = np.copy(img)
    start = float(stride) / 2.0 - 0.5
    center_x = pt[0]
    center_y = pt[1]
    grid_y = img.shape[0]
    grid_x = img.shape[1]
    for i in range(grid_y):
        for j in range(grid_x):
            x = start + j * stride
            y = start + i * stride
            d2 = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)
            exponent = d2 / 2.0 / sigma / sigma
            if exponent > 4.6052:
                continue
            img[i, j] += math.exp(-exponent)
    img = np.clip(img, 0, 1)
    return img

def putVecMaps(entryX, entryY, centerA, centerB, stride, thre):
    centerA = centerA / float(stride)
    centerA_x = centerA[0]
    centerA_y = centerA[1]
    centerB = centerB / float(stride)
    centerB_x = centerB[0]
    centerB_y = centerB[1]
    grid_x = entryX.shape[1]
    grid_y = entryX.shape[0]
    bc = centerA - centerB
    min_x = max(int(round(min(centerA_x, centerB_x) - thre)), 0)
    max_x = min(int(round(max(centerA_x, centerB_x) + thre)), grid_x)
    min_y = max(int(round(min(centerA_y, centerB_y) - thre)), 0)
    max_y = min(int(round(max(centerA_y, centerB_y) + thre)), grid_y)
    norm_bc = np.linalg.norm(bc)
    if norm_bc == 0:
        return entryX, entryY
    bc = bc / norm_bc
    bc_x = bc[0]
    bc_y = bc[1]
    count = np.zeros((grid_y, grid_x))

    for g_y in range(min_y, max_y):
        for g_x in range(min_x, max_x):
            ba_x = g_x - centerA_x
            ba_y = g_y - centerA_y
            dist = np.absolute(ba_x * bc_y - ba_y * bc_x)

            if (dist <= thre):
                cnt = count[g_y, g_x]
                if (cnt == 0):
                    entryX[g_y, g_x] = bc_x
                    entryY[g_y, g_x] = bc_y
                else:
                    entryX[g_y, g_x] = (entryX[g_y, g_x] * cnt + bc_x) / (cnt + 1)
                    entryY[g_y, g_x] = (entryY[g_y, g_x] * cnt + bc_y) / (cnt + 1)
                    count[g_y, g_x] = cnt + 1
    return entryX, entryY

class FileIter(DataIter):
    def __init__(self, img_folder, anno_file, batch_size=20, no_shuffle=True, mean_value=0, div_num=1., num=1e6, inp_res=368, stride=8, train=True, sigma=7, thre=1, target_dist=0.6,
                 flip_prob=0.5, scale_min=0.5, scale_max=1.2, rot_factor=40, center_perterb_max=40, label_type='Gaussian'):
        super(FileIter, self).__init__()
        self.img_folder = img_folder  # root image folders
        self.is_train = train  # training set or test set
        self.inp_res = inp_res
        self.stride = stride
        self.out_res = inp_res / stride
        self.sigma = sigma
        self.thre = thre
        self.target_dist = target_dist
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.center_perterb_max = center_perterb_max
        self.rot_factor = rot_factor
        self.label_type = label_type
        self.nparts = 14
        self.npaf = 26
        self.flip_prob = flip_prob

        print('===> Loading training json')
        start_time = time.time()
        self.annos = json.load(open(anno_file, 'r'))
        deleteIdx = []
        for index in range(len(self.annos)):
            human_annotations = self.annos[index]['human_annotations']
            human_count = len(human_annotations.keys())
            for i in range(human_count):
                anno_key = human_annotations.keys()[i]
                rect = human_annotations[anno_key]
                if (rect[2] - rect[0]) * (rect[3] - rect[1]) == 0:
                    deleteIdx.append(index)
                    break
        self.annos = np.delete(self.annos, deleteIdx, axis=0)
        num = min(num, len(self.annos))
        self.annos = self.annos[:num]
        print('Complete reading annotation JSON file in %.2f seconds.' % (time.time() - start_time))

        self.data_name = ['data']
        self.label_name = ['l2_label']
        self.batch_size = batch_size
        self.data_size = len(self.annos)
        self.index = np.arange(self.data_size)
        self.no_shuffle = no_shuffle
        if not self.no_shuffle:
            random.shuffle(self.index)
        self.data_cursor = 0
        self.mean_value = mean_value
        self.div_num = div_num

        self._read()

    def _getitem(self, index):
        smin = self.scale_min
        smax = self.scale_max
        rf = self.rot_factor
        cp_max = self.center_perterb_max
        img_src = cv2.imread(os.path.join(self.img_folder, self.annos[index]['image_id'] + '.jpg'))
        key_point = self.annos[index]['keypoint_annotations']
        human_annotations = self.annos[index]['human_annotations']
        human_count = len(key_point.keys())
        pts = []
        rect = []
        objpos = []
        nparts = self.nparts
        npaf = self.npaf
        for i in range(human_count):
            anno_key = key_point.keys()[i]
            anno_keypoints = np.reshape(key_point[anno_key], (nparts, 3))
            pts.append(anno_keypoints)
            rect.append(np.array(human_annotations[anno_key]))
            cx = (human_annotations[anno_key][0] + human_annotations[anno_key][2]) / 2
            cy = (human_annotations[anno_key][1] + human_annotations[anno_key][3]) / 2
            objpos.append(np.array([cx, cy]))

        visual = False

        if self.is_train:
            people_select = random.randrange(human_count)
            scale_self = (rect[people_select][3] - rect[people_select][1]) / 368.

            # Augmentation Scale
            s = np.clip((random.random() * (smax - smin)) + smin, smin, smax)
            scale_abs = self.target_dist / scale_self
            s = scale_abs * s
            if s == 0 or img_src is None:
                assert False, self.annos[index]['image_id'] + '.jpg'
            img_temp = cv2.resize(img_src, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

            for i in range(human_count):
                pts[i][:, :2] = pts[i][:, :2] * s
                rect[i] = rect[i] * s
                objpos[i] = objpos[i] * s
                if visual:
                    cv2.circle(img_temp, (int(objpos[i][0]), int(objpos[i][1])), 10, (0, 255, 255), -1)
                    cv2.rectangle(img_temp, (int(rect[i][0]), int(rect[i][1])),
                                  (int(rect[i][2]), int(rect[i][3])), (0, 255, 0), 3)
                    for j in [jj for jj in range(nparts) if int(pts[i][jj, 2]) != 3]:
                        cv2.circle(img_temp, (int(pts[i][j, 0]), int(pts[i][j, 1])), 3, colors[j], -1)
                    if i == human_count-1:
                        cv2.imshow("scale test", img_temp)
                        cv2.waitKey(0)

            # Augmentation Rotate
            r = np.clip((random.random() - 0.5) * (2 * rf), -rf, rf)
            center = (img_temp.shape[1] / 2, img_temp.shape[0] / 2)
            R = cv2.getRotationMatrix2D(center, r, 1.0)
            bbox = cv2.boundingRect(cv2.boxPoints((center, img_temp.shape[1::-1], r)))
            R[0, 2] += bbox[2] / 2 - center[0]
            R[1, 2] += bbox[3] / 2 - center[1]

            img_temp2 = cv2.warpAffine(img_temp, R, bbox[2:4], flags=cv2.INTER_CUBIC+cv2.BORDER_CONSTANT,
                                       borderValue=(127, 127, 127))

            for i in range(human_count):
                temp = np.ones((nparts, 3))
                temp[:, :2] = pts[i][:, :2]
                pts[i][:, :2] = np.transpose(np.dot(R, np.transpose(temp)))
                temp = np.ones((1, 3))
                temp[:, 0] = objpos[i][0]
                temp[:, 1] = objpos[i][1]
                objpos[i][:2] = np.dot(R, np.transpose(temp)).flat
                if visual:
                    cv2.circle(img_temp2, (int(objpos[i][0]), int(objpos[i][1])), 10, (0, 255, 255), -1)
                    for j in [jj for jj in range(nparts) if int(pts[i][jj, 2]) != 3]:
                            cv2.circle(img_temp2, (int(pts[i][j, 0]), int(pts[i][j, 1])), 3, colors[j], -1)
                    if i == human_count - 1:
                        cv2.imshow("scale test", img_temp2)
                        cv2.waitKey(0)

            # Augmentation Crop
            center = objpos[people_select]
            R[0, 0] = 1
            R[0, 1] = 0
            R[0, 2] = 0
            R[1, 0] = 0
            R[1, 1] = 1
            R[1, 2] = 0
            x_offset = np.clip((random.random() - 0.5) * (2 * cp_max), -cp_max, cp_max)
            y_offset = np.clip((random.random() - 0.5) * (2 * cp_max), -cp_max, cp_max)
            bbox = (center[0] - self.inp_res / 2, center[1] - self.inp_res / 2, self.inp_res, self.inp_res)
            R[0, 2] += bbox[2] / 2 - center[0] + x_offset
            R[1, 2] += bbox[3] / 2 - center[1] + y_offset

            img_temp3 = cv2.warpAffine(img_temp2, R, bbox[2:4], flags=cv2.INTER_CUBIC + cv2.BORDER_CONSTANT,
                                       borderValue=(127, 127, 127))

            for i in range(human_count):
                temp = np.ones((nparts, 3))
                temp[:, :2] = pts[i][:, :2]
                pts[i][:, :2] = np.transpose(np.dot(R, np.transpose(temp)))
                temp = np.ones((1, 3))
                temp[:, 0] = objpos[i][0]
                temp[:, 1] = objpos[i][1]
                objpos[i][:2] = np.dot(R, np.transpose(temp)).flat
                if visual:
                    cv2.circle(img_temp3, (int(objpos[i][0]), int(objpos[i][1])), 10, (0, 255, 255), -1)
                    for j in [jj for jj in range(nparts) if int(pts[i][jj, 2]) != 3]:
                            cv2.circle(img_temp3, (int(pts[i][j, 0]), int(pts[i][j, 1])), 3, colors[j], -1)
                    if i == human_count - 1:
                        cv2.imshow("crop test", img_temp3)
                        cv2.waitKey(0)

            # Augmentation Flip
            if random.random() <= self.flip_prob:
                img_aug = cv2.flip(img_temp3, 1)
                for i in range(human_count):
                    pts[i] = swapleftright(pts[i], img_aug.shape[1])
                    objpos[i][0] = img_aug.shape[1] - 1 - objpos[i][0]

                for i in range(human_count):
                    if visual:
                        cv2.circle(img_aug, (int(objpos[i][0]), int(objpos[i][1])), 10, (0, 255, 255), -1)
                        for j in [jj for jj in range(nparts) if int(pts[i][jj, 2]) != 3]:
                            cv2.circle(img_aug, (int(pts[i][j, 0]), int(pts[i][j, 1])), 3, colors[j], -1)
                        if i == human_count - 1:
                            cv2.imshow("Flip test", img_aug)
                            cv2.waitKey(0)
            else:
                img_aug = img_temp3
        else:
            scale = self.inp_res / float(max(img_src.shape[0], img_src.shape[1]))
            dsize = (int(round(float(img_src.shape[1]) * scale)), int(round(float(img_src.shape[0]) * scale)))
            img_aug = cv2.resize(img_src, dsize, interpolation=cv2.INTER_CUBIC)
            img_aug = cv2.copyMakeBorder(img_aug, 0, self.inp_res-img_aug.shape[0], 0, self.inp_res-img_aug.shape[1],
                                         cv2.BORDER_CONSTANT, value=(127, 127, 127))
            for i in range(human_count):
                pts[i][:, :2] = pts[i][:, :2] * scale
                rect[i] = rect[i] * scale
                objpos[i] = objpos[i] * scale

            for i in range(human_count):
                if visual:
                    cv2.imshow("img_src", img_src)
                    cv2.circle(img_aug, (int(objpos[i][0]), int(objpos[i][1])), 10, (0, 255, 255), -1)
                    for j in [jj for jj in range(nparts) if int(pts[i][jj, 2]) != 3]:
                        cv2.circle(img_aug, (int(pts[i][j, 0]), int(pts[i][j, 1])), 3, colors[j], -1)
                    if i == human_count - 1:
                        cv2.imshow("board test", img_aug)
                        cv2.waitKey(0)

        # copy transformed img (img_aug) into transformed_data, do the mean-subtraction here
        # img_aug is not float32, bug!!
        transformed_data = (img_aug.transpose((2, 0, 1)).astype(np.float32) - self.mean_value) / self.div_num

        # Generate ground truth
        target = np.zeros((npaf + nparts + 1, self.out_res, self.out_res))
        mid_1 = [13, 6, 7, 13, 9, 10, 13, 0, 1, 13, 3, 4, 13]
        mid_2 = [ 6, 7, 8,  9,10, 11,  0, 1, 2,  3, 4, 5, 12]
        for i in range(human_count):
            for j in range(npaf / 2):
                if int(pts[i][mid_1[j], 2]) != 3 and int(pts[i][mid_2[j], 2]) != 3:
                    target[2 * j], target[2 * j + 1] = \
                        putVecMaps(target[2 * j], target[2 * j + 1],
                                   pts[i][mid_1[j], :2], pts[i][mid_2[j], :2], self.stride, self.thre)

        for i in range(human_count):
            for j in range(nparts):
                if int(pts[i][j, 2]) != 3:
                    target[npaf+j] = putGaussianMaps(target[npaf+j], pts[i][j], self.stride, self.sigma)

        for j in range(nparts):
            target[npaf + j] = np.clip(target[npaf + j], 0, 1)

        target[nparts + npaf] = 1. - np.max(target[npaf:(npaf + nparts), :, :], 0)

        return transformed_data, target, pts

    def _read(self):
        """get two list, each list contains two elements: name and nd.array value"""
        data, label = self._read_img()
        self.data = [mx.nd.array(data)]
        self.label = [mx.nd.array(label)]

    def _read_img(self):

        train_img = np.zeros((self.batch_size, 3, self.inp_res, self.inp_res), dtype=np.float32)
        train_label = np.zeros((self.batch_size, self.npaf + self.nparts + 1, self.out_res, self.out_res), dtype=np.float32)

        for i in range(self.batch_size):
            data, label, _ = self._getitem(self.index[self.data_cursor + i])

            train_img[i] = data
            train_label[i] = label

            # show data
            if False:
                print data.shape, label.shape
                img = (data.transpose((1, 2, 0)) * self.div_num + self.mean_value).astype(np.uint8)

                perimg_len = 200

                show_height = 4
                show_width = 7
                show_pafs = np.zeros((show_height * perimg_len, show_width * perimg_len, 3), dtype=np.uint8)
                for j in range(self.npaf):
                    pafs = (np.abs(label[j, :, :]) * 255).astype(np.uint8)
                    pafs = cv2.resize(pafs, (0, 0), fx=self.stride, fy=self.stride)
                    pafs = cv2.applyColorMap(pafs, cv2.COLORMAP_JET)
                    img_pafs = (0.6 * img + 0.4 * pafs).astype(np.uint8)
                    index_row = j / show_width
                    index_col = j % show_width
                    show_pafs[index_row * perimg_len:(index_row + 1) * perimg_len, index_col * perimg_len:(index_col + 1) * perimg_len, :] = cv2.resize(img_pafs, (perimg_len, perimg_len))
                cv2.imshow('show_pafs', show_pafs)

                show_len = 4
                show_parts = np.zeros((show_len * perimg_len, show_len * perimg_len, 3), dtype=np.uint8)
                for j in range(self.nparts):
                    parts = (label[self.npaf + j, :, :] * 255).astype(np.uint8)
                    parts = cv2.resize(parts, (0, 0), fx=self.stride, fy=self.stride)
                    parts = cv2.applyColorMap(parts, cv2.COLORMAP_JET)
                    img_parts = (0.6 * img + 0.4 * parts).astype(np.uint8)
                    index_row = j / show_len
                    index_col = j % show_len
                    show_parts[index_row * perimg_len:(index_row + 1) * perimg_len, index_col * perimg_len:(index_col + 1) * perimg_len, :] = cv2.resize(img_parts, (perimg_len, perimg_len))
                cv2.imshow('show_parts', show_parts)

                parts = (label[self.npaf + self.nparts] * 255).astype(np.uint8)
                parts = cv2.resize(parts, (0, 0), fx=self.stride, fy=self.stride)
                parts = cv2.applyColorMap(parts, cv2.COLORMAP_JET)
                img_parts = (0.6 * img + 0.4 * parts).astype(np.uint8)
                cv2.imshow('img_parts_bg', img_parts)

                cv2.imshow('img', img)
                cv2.waitKey()

        return (train_img, train_label)

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def get_batch_size(self):
        return self.batch_size

    def reset(self):
        self.data_cursor = 0
        if not self.no_shuffle:
            random.shuffle(self.index)

    def iter_next(self):
        if self.data_cursor + self.batch_size > self.data_size:
            return False
        else:
            return True

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            self._read()
            self.data_cursor = self.data_cursor + self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label, provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

if __name__ == "__main__":

    root_dir = r'E:\ai_challenger\ai_challenger_keypoint_train_20170909'

    def get_training_set():
        train_dir = os.path.join(root_dir, "keypoint_train_images_20170902")
        anno_file = os.path.join(root_dir, "keypoint_train_annotations_20170909.json")
        train_set = FileIter(train_dir, anno_file)
        return train_set

    data_set = get_training_set()
