import paddle
from paddle import nn
import paddle.nn.functional as F


def cal_performance(pred, tgt):
    
    pred = pred.max(1)[1]
    tgt = tgt.contiguous().view(-1)
    non_pad_mask = tgt.ne(0)
    n_correct = pred.eq(tgt)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    return n_correct


class NRTRLoss(nn.Layer):
    def __init__(self,smoothing=True, **kwargs):
        super(NRTRLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction='mean',ignore_index=0)
        self.smoothing = smoothing

    def forward(self, pred, batch):
        pred = pred.reshape([-1, pred.shape[2]])
        max_len = batch[2].max()
        tgt = batch[1][:,1:2+max_len]
        tgt = tgt.reshape([-1] )
        if self.smoothing:
            eps = 0.1
            n_class = pred.shape[1]
            one_hot = F.one_hot(tgt, pred.shape[1])
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, axis=1)
            non_pad_mask = paddle.not_equal(tgt, paddle.zeros(tgt.shape,dtype='int64'))
            loss = -(one_hot * log_prb).sum(axis=1)
            loss = loss.masked_select(non_pad_mask).mean()
        else:
            loss = self.loss_func(pred, tgt)
        return {'loss': loss}
