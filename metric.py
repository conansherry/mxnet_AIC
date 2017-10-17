import mxnet as mx
import numpy as np

class AICRMSE(mx.metric.EvalMetric):
    def __init__(self, batch_size, name='AICMSE', stage=6, branch=1):
        self.stage = stage
        self.branch = branch
        self.batch_size = batch_size
        super(AICRMSE, self).__init__(name + '_stage_' + str(stage) + '_L' + str(branch))

    def update(self, labels, preds):
        if self.branch == 1:
            pred_l1 = preds[(self.stage - 1) * 2].asnumpy()

            self.sum_metric += (np.sum(pred_l1) / self.batch_size)
            self.num_inst += 1
        else:
            pred_l2 = preds[(self.stage - 1) * 2 + 1].asnumpy()

            self.sum_metric += (np.sum(pred_l2) / self.batch_size)
            self.num_inst += 1