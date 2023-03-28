import paddle
from paddle import nn
import paddle.nn.functional as F


class CELoss(nn.Layer):
    def __init__(self,
                 smoothing=False,
                 with_all=False,
                 ignore_index=-1,
                 **kwargs):
        super(CELoss, self).__init__()
        if ignore_index >= 0:
            self.loss_func = nn.CrossEntropyLoss(
                reduction='mean', ignore_index=ignore_index)
        else:
            self.loss_func = nn.CrossEntropyLoss(reduction='mean')
        self.smoothing = smoothing
        self.with_all = with_all

    def forward(self, pred, batch):

        if isinstance(pred, dict):  # for ABINet
            loss = {}
            loss_sum = []
            for name, logits in pred.items():
                if isinstance(logits, list):
                    logit_num = len(logits)
                    all_tgt = paddle.concat([batch[1]] * logit_num, 0)
                    all_logits = paddle.concat(logits, 0)
                    flt_logtis = all_logits.reshape([-1, all_logits.shape[2]])
                    flt_tgt = all_tgt.reshape([-1])
                else:
                    flt_logtis = logits.reshape([-1, logits.shape[2]])
                    flt_tgt = batch[1].reshape([-1])
                loss[name + '_loss'] = self.loss_func(flt_logtis, flt_tgt)
                loss_sum.append(loss[name + '_loss'])
            loss['loss'] = sum(loss_sum)
            return loss
        else:
            if self.with_all:  # for ViTSTR
                tgt = batch[1]
                pred = pred.reshape([-1, pred.shape[2]])
                tgt = tgt.reshape([-1])
                loss = self.loss_func(pred, tgt)
                return {'loss': loss}
            else:  # for NRTR
                max_len = batch[2].max()
                tgt = batch[1][:, 1:2 + max_len]
                pred = pred.reshape([-1, pred.shape[2]])
                tgt = tgt.reshape([-1])
                if self.smoothing:
                    eps = 0.1
                    n_class = pred.shape[1]
                    one_hot = F.one_hot(tgt, pred.shape[1])
                    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (
                        n_class - 1)
                    log_prb = F.log_softmax(pred, axis=1)
                    non_pad_mask = paddle.not_equal(
                        tgt, paddle.zeros(
                            tgt.shape, dtype=tgt.dtype))
                    loss = -(one_hot * log_prb).sum(axis=1)
                    loss = loss.masked_select(non_pad_mask).mean()
                else:
                    loss = self.loss_func(pred, tgt)
                return {'loss': loss}
