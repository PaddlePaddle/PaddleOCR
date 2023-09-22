import paddle
from paddle import nn
import paddle.nn.functional as F


class CPPDLoss(nn.Layer):
    def __init__(self, smoothing=False, ignore_index=100, sideloss_weight=1.0, **kwargs):
        super(CPPDLoss, self).__init__()
        self.edge_ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)
        self.char_node_ce = nn.CrossEntropyLoss(reduction='mean')
        self.pos_node_ce = nn.BCEWithLogitsLoss(reduction='mean')
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.sideloss_weight = sideloss_weight

    
    def label_smoothing_ce(self, preds, targets):

        non_pad_mask = paddle.not_equal(targets, paddle.zeros(targets.shape, dtype=targets.dtype)+self.ignore_index)
        tgts = paddle.where(targets == (paddle.zeros(targets.shape, dtype=targets.dtype)+self.ignore_index), paddle.zeros(targets.shape, dtype=targets.dtype), targets)
        eps = 0.1
        n_class = preds.shape[1]
        one_hot = F.one_hot(tgts, preds.shape[1])
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(preds, axis=1)
        loss = -(one_hot * log_prb).sum(axis=1)
        loss = loss.masked_select(non_pad_mask).mean()
        return loss


    def forward(self, pred, batch):
        node_feats, edge_feats = pred
        node_tgt = batch[2]
        char_tgt = batch[1]

        loss_char_node = self.char_node_ce(node_feats[0].flatten(0,1), node_tgt[:,:-26].flatten(0,1))
        loss_pos_node = self.pos_node_ce(node_feats[1].flatten(0,1), node_tgt[:,-26:].flatten(0,1).cast('float32'))
        loss_node = loss_char_node + loss_pos_node

        edge_feats = edge_feats.flatten(0,1)
        char_tgt = char_tgt.flatten(0,1)
        if self.smoothing:
            loss_edge = self.label_smoothing_ce(edge_feats, char_tgt)
        else:
            loss_edge = self.edge_ce(edge_feats, char_tgt)

        return {'loss': self.sideloss_weight * loss_node + loss_edge, 'loss_node': self.sideloss_weight * loss_node, 'loss_edge': loss_edge}
