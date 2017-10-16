import mxnet as mx
import numpy as np

class AICRMSE(mx.metric.EvalMetric):
    def __init__(self, name='AICMSE', stage=6, branch=1):
        self.stage = stage
        self.branch = branch
        super(AICRMSE, self).__init__(name + '_stage_' + str(stage) + '_L' + str(branch))

    def update(self, labels, preds):
        if self.branch == 1:
            label_l1 = labels[0].asnumpy()[:, 0:26, :, :]
            pred_l1 = preds[(self.stage - 1) * 2].asnumpy()

            label_l1 = label_l1.reshape((-1))
            pred_l1 = pred_l1.reshape((-1))

            self.sum_metric += np.sqrt(((label_l1 - pred_l1)**2.0).mean())
            self.num_inst += 1
        else:
            label_l2 = labels[0].asnumpy()[:, 26:, :, :]
            pred_l2 = preds[(self.stage - 1) * 2 + 1].asnumpy()

            label_l2 = label_l2.reshape((-1))
            pred_l2 = pred_l2.reshape((-1))

            self.sum_metric += np.sqrt(((label_l2 - pred_l2) ** 2.0).mean())
            self.num_inst += 1