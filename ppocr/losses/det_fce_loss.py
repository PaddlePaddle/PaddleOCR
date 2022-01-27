import numpy as np
from paddle import nn
import paddle
import paddle.nn.functional as F
from functools import partial


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class FCELoss(nn.Layer):
    """The class for implementing FCENet loss
    FCENet(CVPR2021): Fourier Contour Embedding for Arbitrary-shaped
        Text Detection

    [https://arxiv.org/abs/2104.10442]

    Args:
        fourier_degree (int) : The maximum Fourier transform degree k.
        num_sample (int) : The sampling points number of regression
            loss. If it is too small, fcenet tends to be overfitting.
        ohem_ratio (float): the negative/positive ratio in OHEM.
    """

    def __init__(self, fourier_degree, num_sample, ohem_ratio=3.):
        super().__init__()
        self.fourier_degree = fourier_degree
        self.num_sample = num_sample
        self.ohem_ratio = ohem_ratio

    def forward(self, preds, labels):
        assert isinstance(preds, dict)
        preds = preds['levels']

        p3_maps, p4_maps, p5_maps = labels[1:]
        assert p3_maps[0].shape[0] == 4 * self.fourier_degree + 5,\
            'fourier degree not equal in FCEhead and FCEtarget'

        # device = preds[0][0].device
        # to tensor
        gts = [p3_maps, p4_maps, p5_maps]
        for idx, maps in enumerate(gts):
            gts[idx] = paddle.to_tensor(np.stack(maps))

        losses = multi_apply(self.forward_single, preds, gts)

        loss_tr = paddle.to_tensor(0.).astype('float32')
        loss_tcl = paddle.to_tensor(0.).astype('float32')
        loss_reg_x = paddle.to_tensor(0.).astype('float32')
        loss_reg_y = paddle.to_tensor(0.).astype('float32')
        loss_all = paddle.to_tensor(0.).astype('float32')

        for idx, loss in enumerate(losses):
            loss_all += sum(loss)
            if idx == 0:
                loss_tr += sum(loss)
            elif idx == 1:
                loss_tcl += sum(loss)
            elif idx == 2:
                loss_reg_x += sum(loss)
            else:
                loss_reg_y += sum(loss)

        results = dict(
            loss=loss_all,
            loss_text=loss_tr,
            loss_center=loss_tcl,
            loss_reg_x=loss_reg_x,
            loss_reg_y=loss_reg_y, )
        return results

    def forward_single(self, pred, gt):
        cls_pred = paddle.transpose(pred[0], (0, 2, 3, 1))
        reg_pred = paddle.transpose(pred[1], (0, 2, 3, 1))
        gt = paddle.transpose(gt, (0, 2, 3, 1))

        k = 2 * self.fourier_degree + 1
        tr_pred = paddle.reshape(cls_pred[:, :, :, :2], (-1, 2))
        tcl_pred = paddle.reshape(cls_pred[:, :, :, 2:], (-1, 2))
        x_pred = paddle.reshape(reg_pred[:, :, :, 0:k], (-1, k))
        y_pred = paddle.reshape(reg_pred[:, :, :, k:2 * k], (-1, k))

        tr_mask = gt[:, :, :, :1].reshape([-1])
        tcl_mask = gt[:, :, :, 1:2].reshape([-1])
        train_mask = gt[:, :, :, 2:3].reshape([-1])
        x_map = paddle.reshape(gt[:, :, :, 3:3 + k], (-1, k))
        y_map = paddle.reshape(gt[:, :, :, 3 + k:], (-1, k))

        tr_train_mask = (train_mask * tr_mask).astype('bool')
        tr_train_mask2 = paddle.concat(
            [tr_train_mask.unsqueeze(1), tr_train_mask.unsqueeze(1)], axis=1)
        # tr loss
        loss_tr = self.ohem(tr_pred, tr_mask, train_mask)
        # import pdb; pdb.set_trace()
        # tcl loss
        loss_tcl = paddle.to_tensor(0.).astype('float32')
        tr_neg_mask = tr_train_mask.logical_not()
        tr_neg_mask2 = paddle.concat(
            [tr_neg_mask.unsqueeze(1), tr_neg_mask.unsqueeze(1)], axis=1)
        if tr_train_mask.sum().item() > 0:
            loss_tcl_pos = F.cross_entropy(
                tcl_pred.masked_select(tr_train_mask2).reshape([-1, 2]),
                tcl_mask.masked_select(tr_train_mask).astype('int64'))
            loss_tcl_neg = F.cross_entropy(
                tcl_pred.masked_select(tr_neg_mask2).reshape([-1, 2]),
                tcl_mask.masked_select(tr_neg_mask).astype('int64'))
            loss_tcl = loss_tcl_pos + 0.5 * loss_tcl_neg

        # regression loss
        loss_reg_x = paddle.to_tensor(0.).astype('float32')
        loss_reg_y = paddle.to_tensor(0.).astype('float32')
        if tr_train_mask.sum().item() > 0:
            weight = (tr_mask.masked_select(tr_train_mask.astype('bool'))
                      .astype('float32') + tcl_mask.masked_select(
                          tr_train_mask.astype('bool')).astype('float32')) / 2
            weight = weight.reshape([-1, 1])

            ft_x, ft_y = self.fourier2poly(x_map, y_map)
            ft_x_pre, ft_y_pre = self.fourier2poly(x_pred, y_pred)

            dim = ft_x.shape[1]

            tr_train_mask3 = paddle.concat(
                [tr_train_mask.unsqueeze(1) for i in range(dim)], axis=1)

            loss_reg_x = paddle.mean(weight * F.smooth_l1_loss(
                ft_x_pre.masked_select(tr_train_mask3).reshape([-1, dim]),
                ft_x.masked_select(tr_train_mask3).reshape([-1, dim]),
                reduction='none'))
            loss_reg_y = paddle.mean(weight * F.smooth_l1_loss(
                ft_y_pre.masked_select(tr_train_mask3).reshape([-1, dim]),
                ft_y.masked_select(tr_train_mask3).reshape([-1, dim]),
                reduction='none'))

        return loss_tr, loss_tcl, loss_reg_x, loss_reg_y

    def ohem(self, predict, target, train_mask):
        # device = train_mask.device

        pos = (target * train_mask).astype('bool')
        neg = ((1 - target) * train_mask).astype('bool')

        pos2 = paddle.concat([pos.unsqueeze(1), pos.unsqueeze(1)], axis=1)
        neg2 = paddle.concat([neg.unsqueeze(1), neg.unsqueeze(1)], axis=1)

        n_pos = pos.astype('float32').sum()

        if n_pos.item() > 0:
            loss_pos = F.cross_entropy(
                predict.masked_select(pos2).reshape([-1, 2]),
                target.masked_select(pos).astype('int64'),
                reduction='sum')
            loss_neg = F.cross_entropy(
                predict.masked_select(neg2).reshape([-1, 2]),
                target.masked_select(neg).astype('int64'),
                reduction='none')
            n_neg = min(
                int(neg.astype('float32').sum().item()),
                int(self.ohem_ratio * n_pos.astype('float32')))
        else:
            loss_pos = paddle.to_tensor(0.)
            loss_neg = F.cross_entropy(
                predict.masked_select(neg2).reshape([-1, 2]),
                target.masked_select(neg).astype('int64'),
                reduction='none')
            n_neg = 100
        if len(loss_neg) > n_neg:
            loss_neg, _ = paddle.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).astype('float32')

    def fourier2poly(self, real_maps, imag_maps):
        """Transform Fourier coefficient maps to polygon maps.

        Args:
            real_maps (tensor): A map composed of the real parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)
            imag_maps (tensor):A map composed of the imag parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)

        Returns
            x_maps (tensor): A map composed of the x value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
            y_maps (tensor): A map composed of the y value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
        """

        k_vect = paddle.arange(
            -self.fourier_degree, self.fourier_degree + 1,
            dtype='float32').reshape([-1, 1])
        i_vect = paddle.arange(
            0, self.num_sample, dtype='float32').reshape([1, -1])

        transform_matrix = 2 * np.pi / self.num_sample * paddle.matmul(k_vect,
                                                                       i_vect)

        x1 = paddle.einsum('ak, kn-> an', real_maps,
                           paddle.cos(transform_matrix))
        x2 = paddle.einsum('ak, kn-> an', imag_maps,
                           paddle.sin(transform_matrix))
        y1 = paddle.einsum('ak, kn-> an', real_maps,
                           paddle.sin(transform_matrix))
        y2 = paddle.einsum('ak, kn-> an', imag_maps,
                           paddle.cos(transform_matrix))

        x_maps = x1 - x2
        y_maps = y1 + y2

        return x_maps, y_maps
