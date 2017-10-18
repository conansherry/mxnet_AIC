import mxnet as mx
import numpy as np
import cv2

class AICRMSE(mx.metric.EvalMetric):
    def __init__(self, batch_size, name='AICRMSE', stage=6, branch=1):
        self.stage = stage
        self.branch = branch
        self.batch_size = batch_size
        super(AICRMSE, self).__init__(name + '_stage_' + str(stage) + '_L' + str(branch))

        self.div_num = 255.
        self.mean_value = 127
        self.npaf = 26
        self.nparts = 14
        self.stride = 8

    def update(self, labels, preds):

        # show data
        if False and self.stage == 1 and self.branch == 1:
            for i in range(preds[12].asnumpy().shape[0]):
                data = preds[12].asnumpy()[i]
                label = np.concatenate((preds[13].asnumpy()[i], preds[14].asnumpy()[i]), axis=0)
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
                    show_pafs[index_row * perimg_len:(index_row + 1) * perimg_len,
                    index_col * perimg_len:(index_col + 1) * perimg_len, :] = cv2.resize(img_pafs, (perimg_len, perimg_len))
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
                    show_parts[index_row * perimg_len:(index_row + 1) * perimg_len,
                    index_col * perimg_len:(index_col + 1) * perimg_len, :] = cv2.resize(img_parts,
                                                                                         (perimg_len, perimg_len))
                cv2.imshow('show_parts', show_parts)

                parts = (label[self.npaf + self.nparts] * 255).astype(np.uint8)
                parts = cv2.resize(parts, (0, 0), fx=self.stride, fy=self.stride)
                parts = cv2.applyColorMap(parts, cv2.COLORMAP_JET)
                img_parts = (0.6 * img + 0.4 * parts).astype(np.uint8)
                cv2.imshow('img_parts_bg', img_parts)

                cv2.imshow('img', img)
                cv2.waitKey()

        if self.branch == 1:
            pred_l1 = preds[(self.stage - 1) * 2].asnumpy()

            self.sum_metric += (np.sum(pred_l1) / self.batch_size)
            self.num_inst += 1
        else:
            pred_l2 = preds[(self.stage - 1) * 2 + 1].asnumpy()

            self.sum_metric += (np.sum(pred_l2) / self.batch_size)
            self.num_inst += 1