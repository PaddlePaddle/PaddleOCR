# reference from : https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/kie/losses/sdmgr_loss.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import nn
import paddle


class SDMGRLoss(nn.Layer):
    def __init__(self, node_weight=1.0, edge_weight=1.0, ignore=0):
        super().__init__()
        self.loss_node = nn.CrossEntropyLoss(ignore_index=ignore)
        self.loss_edge = nn.CrossEntropyLoss(ignore_index=-1)
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.ignore = ignore

    def pre_process(self, gts, tag):
        gts, tag = gts.numpy(), tag.numpy().tolist()
        temp_gts = []
        batch = len(tag)
        for i in range(batch):
            num, recoder_len = tag[i][0], tag[i][1]
            temp_gts.append(
                paddle.to_tensor(
                    gts[i, :num, :num + 1], dtype='int64'))
        return temp_gts

    def accuracy(self, pred, target, topk=1, thresh=None):
        """Calculate accuracy according to the prediction and target.

        Args:
            pred (torch.Tensor): The model prediction, shape (N, num_class)
            target (torch.Tensor): The target of each prediction, shape (N, )
            topk (int | tuple[int], optional): If the predictions in ``topk``
                matches the target, the predictions will be regarded as
                correct ones. Defaults to 1.
            thresh (float, optional): If not None, predictions with scores under
                this threshold are considered incorrect. Default to None.

        Returns:
            float | tuple[float]: If the input ``topk`` is a single integer,
                the function will return a single float as accuracy. If
                ``topk`` is a tuple containing multiple integers, the
                function will return a tuple containing accuracies of
                each ``topk`` number.
        """
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
            return_single = True
        else:
            return_single = False

        maxk = max(topk)
        if pred.shape[0] == 0:
            accu = [pred.new_tensor(0.) for i in range(len(topk))]
            return accu[0] if return_single else accu
        pred_value, pred_label = paddle.topk(pred, maxk, axis=1)
        pred_label = pred_label.transpose(
            [1, 0])  # transpose to shape (maxk, N)
        correct = paddle.equal(pred_label,
                               (target.reshape([1, -1]).expand_as(pred_label)))
        res = []
        for k in topk:
            correct_k = paddle.sum(correct[:k].reshape([-1]).astype('float32'),
                                   axis=0,
                                   keepdim=True)
            res.append(
                paddle.multiply(correct_k,
                                paddle.to_tensor(100.0 / pred.shape[0])))
        return res[0] if return_single else res

    def forward(self, pred, batch):
        node_preds, edge_preds = pred
        gts, tag = batch[4], batch[5]
        gts = self.pre_process(gts, tag)
        node_gts, edge_gts = [], []
        for gt in gts:
            node_gts.append(gt[:, 0])
            edge_gts.append(gt[:, 1:].reshape([-1]))
        node_gts = paddle.concat(node_gts)
        edge_gts = paddle.concat(edge_gts)

        node_valids = paddle.nonzero(node_gts != self.ignore).reshape([-1])
        edge_valids = paddle.nonzero(edge_gts != -1).reshape([-1])
        loss_node = self.loss_node(node_preds, node_gts)
        loss_edge = self.loss_edge(edge_preds, edge_gts)
        loss = self.node_weight * loss_node + self.edge_weight * loss_edge
        return dict(
            loss=loss,
            loss_node=loss_node,
            loss_edge=loss_edge,
            acc_node=self.accuracy(
                paddle.gather(node_preds, node_valids),
                paddle.gather(node_gts, node_valids)),
            acc_edge=self.accuracy(
                paddle.gather(edge_preds, edge_valids),
                paddle.gather(edge_gts, edge_valids)))
