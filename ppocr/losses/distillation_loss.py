# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import cv2

from .rec_ctc_loss import CTCLoss
from .rec_sar_loss import SARLoss
from .rec_ce_loss import CELoss
from .basic_loss import DMLLoss, KLDivLoss, DKDLoss
from .basic_loss import DistanceLoss
from .basic_loss import LossFromOutput
from .det_db_loss import DBLoss
from .det_basic_loss import BalanceLoss, MaskL1Loss, DiceLoss
from .vqa_token_layoutlm_loss import VQASerTokenLayoutLMLoss


def _sum_loss(loss_dict):
    if "loss" in loss_dict.keys():
        return loss_dict
    else:
        loss_dict["loss"] = 0.0
        for k, value in loss_dict.items():
            if k == "loss":
                continue
            else:
                loss_dict["loss"] += value
        return loss_dict


class DistillationDMLLoss(DMLLoss):
    """ """

    def __init__(
        self,
        model_name_pairs=[],
        act=None,
        use_log=False,
        key=None,
        multi_head=False,
        dis_head="ctc",
        maps_name=None,
        name="dml",
    ):
        super().__init__(act=act, use_log=use_log)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.multi_head = multi_head
        self.dis_head = dis_head
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.name = name
        self.maps_name = self._check_maps_name(maps_name)

    def _check_model_name_pairs(self, model_name_pairs):
        if not isinstance(model_name_pairs, list):
            return []
        elif isinstance(model_name_pairs[0], list) and isinstance(
            model_name_pairs[0][0], str
        ):
            return model_name_pairs
        else:
            return [model_name_pairs]

    def _check_maps_name(self, maps_name):
        if maps_name is None:
            return None
        elif isinstance(maps_name, str):
            return [maps_name]
        elif isinstance(maps_name, list):
            return [maps_name]
        else:
            return None

    def _slice_out(self, outs):
        new_outs = {}
        for k in self.maps_name:
            if k == "thrink_maps":
                new_outs[k] = outs[:, 0, :, :]
            elif k == "threshold_maps":
                new_outs[k] = outs[:, 1, :, :]
            elif k == "binary_maps":
                new_outs[k] = outs[:, 2, :, :]
            else:
                continue
        return new_outs

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            if self.maps_name is None:
                if self.multi_head:
                    loss = super().forward(out1[self.dis_head], out2[self.dis_head])
                else:
                    loss = super().forward(out1, out2)
                if isinstance(loss, dict):
                    for key in loss:
                        loss_dict["{}_{}_{}_{}".format(key, pair[0], pair[1], idx)] = (
                            loss[key]
                        )
                else:
                    loss_dict["{}_{}".format(self.name, idx)] = loss
            else:
                outs1 = self._slice_out(out1)
                outs2 = self._slice_out(out2)
                for _c, k in enumerate(outs1.keys()):
                    loss = super().forward(outs1[k], outs2[k])
                    if isinstance(loss, dict):
                        for key in loss:
                            loss_dict[
                                "{}_{}_{}_{}_{}".format(
                                    key, pair[0], pair[1], self.maps_name, idx
                                )
                            ] = loss[key]
                    else:
                        loss_dict[
                            "{}_{}_{}".format(self.name, self.maps_name[_c], idx)
                        ] = loss

        loss_dict = _sum_loss(loss_dict)

        return loss_dict


class DistillationKLDivLoss(KLDivLoss):
    """ """

    def __init__(
        self,
        model_name_pairs=[],
        key=None,
        multi_head=False,
        dis_head="ctc",
        maps_name=None,
        name="kl_div",
    ):
        super().__init__()
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.multi_head = multi_head
        self.dis_head = dis_head
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.name = name
        self.maps_name = self._check_maps_name(maps_name)

    def _check_model_name_pairs(self, model_name_pairs):
        if not isinstance(model_name_pairs, list):
            return []
        elif isinstance(model_name_pairs[0], list) and isinstance(
            model_name_pairs[0][0], str
        ):
            return model_name_pairs
        else:
            return [model_name_pairs]

    def _check_maps_name(self, maps_name):
        if maps_name is None:
            return None
        elif isinstance(maps_name, str):
            return [maps_name]
        elif isinstance(maps_name, list):
            return [maps_name]
        else:
            return None

    def _slice_out(self, outs):
        new_outs = {}
        for k in self.maps_name:
            if k == "thrink_maps":
                new_outs[k] = outs[:, 0, :, :]
            elif k == "threshold_maps":
                new_outs[k] = outs[:, 1, :, :]
            elif k == "binary_maps":
                new_outs[k] = outs[:, 2, :, :]
            else:
                continue
        return new_outs

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            if self.maps_name is None:
                if self.multi_head:
                    # for nrtr dml loss
                    max_len = batch[3].max()
                    tgt = batch[2][:, 1 : 2 + max_len]
                    tgt = tgt.reshape([-1])
                    non_pad_mask = paddle.not_equal(
                        tgt, paddle.zeros(tgt.shape, dtype=tgt.dtype)
                    )
                    loss = super().forward(
                        out1[self.dis_head], out2[self.dis_head], non_pad_mask
                    )
                else:
                    loss = super().forward(out1, out2)
                if isinstance(loss, dict):
                    for key in loss:
                        loss_dict["{}_{}_{}_{}".format(key, pair[0], pair[1], idx)] = (
                            loss[key]
                        )
                else:
                    loss_dict["{}_{}".format(self.name, idx)] = loss
            else:
                outs1 = self._slice_out(out1)
                outs2 = self._slice_out(out2)
                for _c, k in enumerate(outs1.keys()):
                    loss = super().forward(outs1[k], outs2[k])
                    if isinstance(loss, dict):
                        for key in loss:
                            loss_dict[
                                "{}_{}_{}_{}_{}".format(
                                    key, pair[0], pair[1], self.maps_name, idx
                                )
                            ] = loss[key]
                    else:
                        loss_dict[
                            "{}_{}_{}".format(self.name, self.maps_name[_c], idx)
                        ] = loss

        loss_dict = _sum_loss(loss_dict)

        return loss_dict


class DistillationDKDLoss(DKDLoss):
    """ """

    def __init__(
        self,
        model_name_pairs=[],
        key=None,
        multi_head=False,
        dis_head="ctc",
        maps_name=None,
        name="dkd",
        temperature=1.0,
        alpha=1.0,
        beta=1.0,
    ):
        super().__init__(temperature, alpha, beta)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.multi_head = multi_head
        self.dis_head = dis_head
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.name = name
        self.maps_name = self._check_maps_name(maps_name)

    def _check_model_name_pairs(self, model_name_pairs):
        if not isinstance(model_name_pairs, list):
            return []
        elif isinstance(model_name_pairs[0], list) and isinstance(
            model_name_pairs[0][0], str
        ):
            return model_name_pairs
        else:
            return [model_name_pairs]

    def _check_maps_name(self, maps_name):
        if maps_name is None:
            return None
        elif isinstance(maps_name, str):
            return [maps_name]
        elif isinstance(maps_name, list):
            return [maps_name]
        else:
            return None

    def _slice_out(self, outs):
        new_outs = {}
        for k in self.maps_name:
            if k == "thrink_maps":
                new_outs[k] = outs[:, 0, :, :]
            elif k == "threshold_maps":
                new_outs[k] = outs[:, 1, :, :]
            elif k == "binary_maps":
                new_outs[k] = outs[:, 2, :, :]
            else:
                continue
        return new_outs

    def forward(self, predicts, batch):
        loss_dict = dict()

        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            if self.maps_name is None:
                if self.multi_head:
                    # for nrtr dml loss
                    max_len = batch[3].max()
                    tgt = batch[2][:, 1 : 2 + max_len]  # [batch_size, max_len + 1]

                    tgt = tgt.reshape([-1])  # batch_size * (max_len + 1)
                    non_pad_mask = paddle.not_equal(
                        tgt, paddle.zeros(tgt.shape, dtype=tgt.dtype)
                    )  # batch_size * (max_len + 1)

                    loss = super().forward(
                        out1[self.dis_head], out2[self.dis_head], tgt, non_pad_mask
                    )  # [batch_size, max_len + 1, num_char]
                else:
                    loss = super().forward(out1, out2)
                if isinstance(loss, dict):
                    for key in loss:
                        loss_dict["{}_{}_{}_{}".format(key, pair[0], pair[1], idx)] = (
                            loss[key]
                        )
                else:
                    loss_dict["{}_{}".format(self.name, idx)] = loss
            else:
                outs1 = self._slice_out(out1)
                outs2 = self._slice_out(out2)
                for _c, k in enumerate(outs1.keys()):
                    loss = super().forward(outs1[k], outs2[k])
                    if isinstance(loss, dict):
                        for key in loss:
                            loss_dict[
                                "{}_{}_{}_{}_{}".format(
                                    key, pair[0], pair[1], self.maps_name, idx
                                )
                            ] = loss[key]
                    else:
                        loss_dict[
                            "{}_{}_{}".format(self.name, self.maps_name[_c], idx)
                        ] = loss

        loss_dict = _sum_loss(loss_dict)

        return loss_dict


class DistillationNRTRDMLLoss(DistillationDMLLoss):
    """ """

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]

            if self.multi_head:
                # for nrtr dml loss
                max_len = batch[3].max()
                tgt = batch[2][:, 1 : 2 + max_len]
                tgt = tgt.reshape([-1])
                non_pad_mask = paddle.not_equal(
                    tgt, paddle.zeros(tgt.shape, dtype=tgt.dtype)
                )
                loss = super().forward(
                    out1[self.dis_head], out2[self.dis_head], non_pad_mask
                )
            else:
                loss = super().forward(out1, out2)
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict["{}_{}_{}_{}".format(key, pair[0], pair[1], idx)] = loss[
                        key
                    ]
            else:
                loss_dict["{}_{}".format(self.name, idx)] = loss

        loss_dict = _sum_loss(loss_dict)

        return loss_dict


class DistillationKLDivLoss(KLDivLoss):
    """ """

    def __init__(
        self,
        model_name_pairs=[],
        key=None,
        multi_head=False,
        dis_head="ctc",
        maps_name=None,
        name="kl_div",
    ):
        super().__init__()
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.multi_head = multi_head
        self.dis_head = dis_head
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.name = name
        self.maps_name = self._check_maps_name(maps_name)

    def _check_model_name_pairs(self, model_name_pairs):
        if not isinstance(model_name_pairs, list):
            return []
        elif isinstance(model_name_pairs[0], list) and isinstance(
            model_name_pairs[0][0], str
        ):
            return model_name_pairs
        else:
            return [model_name_pairs]

    def _check_maps_name(self, maps_name):
        if maps_name is None:
            return None
        elif isinstance(maps_name, str):
            return [maps_name]
        elif isinstance(maps_name, list):
            return [maps_name]
        else:
            return None

    def _slice_out(self, outs):
        new_outs = {}
        for k in self.maps_name:
            if k == "thrink_maps":
                new_outs[k] = outs[:, 0, :, :]
            elif k == "threshold_maps":
                new_outs[k] = outs[:, 1, :, :]
            elif k == "binary_maps":
                new_outs[k] = outs[:, 2, :, :]
            else:
                continue
        return new_outs

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            if self.maps_name is None:
                if self.multi_head:
                    # for nrtr dml loss
                    max_len = batch[3].max()
                    tgt = batch[2][:, 1 : 2 + max_len]
                    tgt = tgt.reshape([-1])
                    non_pad_mask = paddle.not_equal(
                        tgt, paddle.zeros(tgt.shape, dtype=tgt.dtype)
                    )
                    loss = super().forward(
                        out1[self.dis_head], out2[self.dis_head], non_pad_mask
                    )
                else:
                    loss = super().forward(out1, out2)
                if isinstance(loss, dict):
                    for key in loss:
                        loss_dict["{}_{}_{}_{}".format(key, pair[0], pair[1], idx)] = (
                            loss[key]
                        )
                else:
                    loss_dict["{}_{}".format(self.name, idx)] = loss
            else:
                outs1 = self._slice_out(out1)
                outs2 = self._slice_out(out2)
                for _c, k in enumerate(outs1.keys()):
                    loss = super().forward(outs1[k], outs2[k])
                    if isinstance(loss, dict):
                        for key in loss:
                            loss_dict[
                                "{}_{}_{}_{}_{}".format(
                                    key, pair[0], pair[1], self.maps_name, idx
                                )
                            ] = loss[key]
                    else:
                        loss_dict[
                            "{}_{}_{}".format(self.name, self.maps_name[_c], idx)
                        ] = loss

        loss_dict = _sum_loss(loss_dict)

        return loss_dict


class DistillationDKDLoss(DKDLoss):
    """ """

    def __init__(
        self,
        model_name_pairs=[],
        key=None,
        multi_head=False,
        dis_head="ctc",
        maps_name=None,
        name="dkd",
        temperature=1.0,
        alpha=1.0,
        beta=1.0,
    ):
        super().__init__(temperature, alpha, beta)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.multi_head = multi_head
        self.dis_head = dis_head
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.name = name
        self.maps_name = self._check_maps_name(maps_name)

    def _check_model_name_pairs(self, model_name_pairs):
        if not isinstance(model_name_pairs, list):
            return []
        elif isinstance(model_name_pairs[0], list) and isinstance(
            model_name_pairs[0][0], str
        ):
            return model_name_pairs
        else:
            return [model_name_pairs]

    def _check_maps_name(self, maps_name):
        if maps_name is None:
            return None
        elif isinstance(maps_name, str):
            return [maps_name]
        elif isinstance(maps_name, list):
            return [maps_name]
        else:
            return None

    def _slice_out(self, outs):
        new_outs = {}
        for k in self.maps_name:
            if k == "thrink_maps":
                new_outs[k] = outs[:, 0, :, :]
            elif k == "threshold_maps":
                new_outs[k] = outs[:, 1, :, :]
            elif k == "binary_maps":
                new_outs[k] = outs[:, 2, :, :]
            else:
                continue
        return new_outs

    def forward(self, predicts, batch):
        loss_dict = dict()

        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            if self.maps_name is None:
                if self.multi_head:
                    # for nrtr dml loss
                    max_len = batch[3].max()
                    tgt = batch[2][:, 1 : 2 + max_len]  # [batch_size, max_len + 1]

                    tgt = tgt.reshape([-1])  # batch_size * (max_len + 1)
                    non_pad_mask = paddle.not_equal(
                        tgt, paddle.zeros(tgt.shape, dtype=tgt.dtype)
                    )  # batch_size * (max_len + 1)

                    loss = super().forward(
                        out1[self.dis_head], out2[self.dis_head], tgt, non_pad_mask
                    )  # [batch_size, max_len + 1, num_char]
                else:
                    loss = super().forward(out1, out2)
                if isinstance(loss, dict):
                    for key in loss:
                        loss_dict["{}_{}_{}_{}".format(key, pair[0], pair[1], idx)] = (
                            loss[key]
                        )
                else:
                    loss_dict["{}_{}".format(self.name, idx)] = loss
            else:
                outs1 = self._slice_out(out1)
                outs2 = self._slice_out(out2)
                for _c, k in enumerate(outs1.keys()):
                    loss = super().forward(outs1[k], outs2[k])
                    if isinstance(loss, dict):
                        for key in loss:
                            loss_dict[
                                "{}_{}_{}_{}_{}".format(
                                    key, pair[0], pair[1], self.maps_name, idx
                                )
                            ] = loss[key]
                    else:
                        loss_dict[
                            "{}_{}_{}".format(self.name, self.maps_name[_c], idx)
                        ] = loss

        loss_dict = _sum_loss(loss_dict)

        return loss_dict


class DistillationCTCLoss(CTCLoss):
    def __init__(self, model_name_list=[], key=None, multi_head=False, name="loss_ctc"):
        super().__init__()
        self.model_name_list = model_name_list
        self.key = key
        self.name = name
        self.multi_head = multi_head

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, model_name in enumerate(self.model_name_list):
            out = predicts[model_name]
            if self.key is not None:
                out = out[self.key]
            if self.multi_head:
                assert "ctc" in out, "multi head has multi out"
                loss = super().forward(out["ctc"], batch[:2] + batch[3:])
            else:
                loss = super().forward(out, batch)
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict["{}_{}_{}".format(self.name, model_name, idx)] = loss[key]
            else:
                loss_dict["{}_{}".format(self.name, model_name)] = loss
        return loss_dict


class DistillationSARLoss(SARLoss):
    def __init__(
        self, model_name_list=[], key=None, multi_head=False, name="loss_sar", **kwargs
    ):
        ignore_index = kwargs.get("ignore_index", 92)
        super().__init__(ignore_index=ignore_index)
        self.model_name_list = model_name_list
        self.key = key
        self.name = name
        self.multi_head = multi_head

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, model_name in enumerate(self.model_name_list):
            out = predicts[model_name]
            if self.key is not None:
                out = out[self.key]
            if self.multi_head:
                assert "sar" in out, "multi head has multi out"
                loss = super().forward(out["sar"], batch[:1] + batch[2:])
            else:
                loss = super().forward(out, batch)
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict["{}_{}_{}".format(self.name, model_name, idx)] = loss[key]
            else:
                loss_dict["{}_{}".format(self.name, model_name)] = loss
        return loss_dict


class DistillationNRTRLoss(CELoss):
    def __init__(
        self,
        model_name_list=[],
        key=None,
        multi_head=False,
        smoothing=True,
        name="loss_nrtr",
        **kwargs,
    ):
        super().__init__(smoothing=smoothing)
        self.model_name_list = model_name_list
        self.key = key
        self.name = name
        self.multi_head = multi_head

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, model_name in enumerate(self.model_name_list):
            out = predicts[model_name]
            if self.key is not None:
                out = out[self.key]
            if self.multi_head:
                assert "gtc" in out, "multi head has multi out"
                loss = super().forward(out["gtc"], batch[:1] + batch[2:])
            else:
                loss = super().forward(out, batch)
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict["{}_{}_{}".format(self.name, model_name, idx)] = loss[key]
            else:
                loss_dict["{}_{}".format(self.name, model_name)] = loss
        return loss_dict


class DistillationDBLoss(DBLoss):
    def __init__(
        self,
        model_name_list=[],
        balance_loss=True,
        main_loss_type="DiceLoss",
        alpha=5,
        beta=10,
        ohem_ratio=3,
        eps=1e-6,
        name="db",
        **kwargs,
    ):
        super().__init__()
        self.model_name_list = model_name_list
        self.name = name
        self.key = None

    def forward(self, predicts, batch):
        loss_dict = {}
        for idx, model_name in enumerate(self.model_name_list):
            out = predicts[model_name]
            if self.key is not None:
                out = out[self.key]
            loss = super().forward(out, batch)

            if isinstance(loss, dict):
                for key in loss.keys():
                    if key == "loss":
                        continue
                    name = "{}_{}_{}".format(self.name, model_name, key)
                    loss_dict[name] = loss[key]
            else:
                loss_dict["{}_{}".format(self.name, model_name)] = loss

        loss_dict = _sum_loss(loss_dict)
        return loss_dict


class DistillationDilaDBLoss(DBLoss):
    def __init__(
        self,
        model_name_pairs=[],
        key=None,
        balance_loss=True,
        main_loss_type="DiceLoss",
        alpha=5,
        beta=10,
        ohem_ratio=3,
        eps=1e-6,
        name="dila_dbloss",
    ):
        super().__init__()
        self.model_name_pairs = model_name_pairs
        self.name = name
        self.key = key

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            stu_outs = predicts[pair[0]]
            tch_outs = predicts[pair[1]]
            if self.key is not None:
                stu_preds = stu_outs[self.key]
                tch_preds = tch_outs[self.key]

            stu_shrink_maps = stu_preds[:, 0, :, :]
            stu_binary_maps = stu_preds[:, 2, :, :]

            # dilation to teacher prediction
            dilation_w = np.array([[1, 1], [1, 1]])
            th_shrink_maps = tch_preds[:, 0, :, :]
            if hasattr(paddle.Tensor, "contiguous"):
                th_shrink_maps = th_shrink_maps.contiguous()
            th_shrink_maps = th_shrink_maps.numpy() > 0.3  # thresh = 0.3
            dilate_maps = np.zeros_like(th_shrink_maps).astype(np.float32)
            for i in range(th_shrink_maps.shape[0]):
                dilate_maps[i] = cv2.dilate(
                    th_shrink_maps[i, :, :].astype(np.uint8), dilation_w
                )
            th_shrink_maps = paddle.to_tensor(dilate_maps)

            (
                label_threshold_map,
                label_threshold_mask,
                label_shrink_map,
                label_shrink_mask,
            ) = batch[1:]

            # calculate the shrink map loss
            bce_loss = self.alpha * self.bce_loss(
                stu_shrink_maps, th_shrink_maps, label_shrink_mask
            )
            loss_binary_maps = self.dice_loss(
                stu_binary_maps, th_shrink_maps, label_shrink_mask
            )

            # k = f"{self.name}_{pair[0]}_{pair[1]}"
            k = "{}_{}_{}".format(self.name, pair[0], pair[1])
            loss_dict[k] = bce_loss + loss_binary_maps

        loss_dict = _sum_loss(loss_dict)
        return loss_dict


class DistillationDistanceLoss(DistanceLoss):
    """ """

    def __init__(
        self, mode="l2", model_name_pairs=[], key=None, name="loss_distance", **kargs
    ):
        super().__init__(mode=mode, **kargs)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.model_name_pairs = model_name_pairs
        self.name = name + "_l2"

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            loss = super().forward(out1, out2)
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict["{}_{}_{}".format(self.name, key, idx)] = loss[key]
            else:
                loss_dict["{}_{}_{}_{}".format(self.name, pair[0], pair[1], idx)] = loss
        return loss_dict


class DistillationVQASerTokenLayoutLMLoss(VQASerTokenLayoutLMLoss):
    def __init__(self, num_classes, model_name_list=[], key=None, name="loss_ser"):
        super().__init__(num_classes=num_classes)
        self.model_name_list = model_name_list
        self.key = key
        self.name = name

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, model_name in enumerate(self.model_name_list):
            out = predicts[model_name]
            if self.key is not None:
                out = out[self.key]
            loss = super().forward(out, batch)
            loss_dict["{}_{}".format(self.name, model_name)] = loss["loss"]
        return loss_dict


class DistillationLossFromOutput(LossFromOutput):
    def __init__(
        self,
        reduction="none",
        model_name_list=[],
        dist_key=None,
        key="loss",
        name="loss_re",
    ):
        super().__init__(key=key, reduction=reduction)
        self.model_name_list = model_name_list
        self.name = name
        self.dist_key = dist_key

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, model_name in enumerate(self.model_name_list):
            out = predicts[model_name]
            if self.dist_key is not None:
                out = out[self.dist_key]
            loss = super().forward(out, batch)
            loss_dict["{}_{}".format(self.name, model_name)] = loss["loss"]
        return loss_dict


class DistillationSERDMLLoss(DMLLoss):
    """ """

    def __init__(
        self,
        act="softmax",
        use_log=True,
        num_classes=7,
        model_name_pairs=[],
        key=None,
        name="loss_dml_ser",
    ):
        super().__init__(act=act, use_log=use_log)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.name = name
        self.num_classes = num_classes
        self.model_name_pairs = model_name_pairs

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            out1 = out1.reshape([-1, out1.shape[-1]])
            out2 = out2.reshape([-1, out2.shape[-1]])

            attention_mask = batch[2]
            if attention_mask is not None:
                active_output = (
                    attention_mask.reshape(
                        [
                            -1,
                        ]
                    )
                    == 1
                )
                out1 = out1[active_output]
                out2 = out2[active_output]

            loss_dict["{}_{}".format(self.name, idx)] = super().forward(out1, out2)

        return loss_dict


class DistillationVQADistanceLoss(DistanceLoss):
    def __init__(
        self,
        mode="l2",
        model_name_pairs=[],
        key=None,
        index=None,
        name="loss_distance",
        **kargs,
    ):
        super().__init__(mode=mode, **kargs)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.index = index
        self.model_name_pairs = model_name_pairs
        self.name = name + "_l2"

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            attention_mask = batch[2]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
                if self.index is not None:
                    out1 = out1[:, self.index, :, :]
                    out2 = out2[:, self.index, :, :]
                if attention_mask is not None:
                    max_len = attention_mask.shape[-1]
                    out1 = out1[:, :max_len]
                    out2 = out2[:, :max_len]
                out1 = out1.reshape([-1, out1.shape[-1]])
                out2 = out2.reshape([-1, out2.shape[-1]])
            if attention_mask is not None:
                active_output = (
                    attention_mask.reshape(
                        [
                            -1,
                        ]
                    )
                    == 1
                )
                out1 = out1[active_output]
                out2 = out2[active_output]

            loss = super().forward(out1, out2)
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict["{}_{}nohu_{}".format(self.name, key, idx)] = loss[key]
            else:
                loss_dict["{}_{}_{}_{}".format(self.name, pair[0], pair[1], idx)] = loss
        return loss_dict


class CTCDKDLoss(nn.Layer):
    """
    KLDivLoss
    """

    def __init__(self, temperature=0.5, alpha=1.0, beta=1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-6
        self.t = temperature
        self.act = nn.Softmax(axis=-1)
        self.use_log = True

    def kl_loss(self, p1, p2):  # predict, label
        loss = paddle.multiply(
            p2, paddle.log((p2 + self.eps) / (p1 + self.eps) + self.eps)
        )
        bs = loss.shape[0]
        loss = paddle.sum(loss) / bs
        return loss

    def _cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(axis=1, keepdim=True)
        t2 = (t * mask2).sum(axis=1, keepdim=True)
        rt = paddle.concat([t1, t2], axis=1)
        return rt

    def multi_label_mask(self, targets):
        targets = targets.astype("int32")
        res = F.one_hot(targets, num_classes=11465)
        mask = paddle.clip(paddle.sum(res, axis=1), 0, 1)
        mask[:, 0] = 0  # ignore ctc blank label
        return mask

    def forward(self, logits_student, logits_teacher, targets, mask=None):
        gt_mask = self.multi_label_mask(targets)
        other_mask = paddle.ones_like(gt_mask) - gt_mask

        pred_student = F.softmax(logits_student / self.temperature, axis=-1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, axis=-1)

        # differences with dkd
        pred_student = paddle.mean(pred_student, axis=1)
        pred_teacher = paddle.mean(pred_teacher, axis=1)

        pred_student = self._cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self._cat_mask(pred_teacher, gt_mask, other_mask)

        # differences with dkd
        tckd_loss = self.kl_loss(pred_student, pred_teacher)

        gt_mask_ex = paddle.expand_as(gt_mask.unsqueeze(axis=1), logits_teacher)
        pred_teacher_part2 = F.softmax(
            logits_teacher / self.temperature - 1000.0 * gt_mask_ex, axis=-1
        )
        pred_student_part2 = F.softmax(
            logits_student / self.temperature - 1000.0 * gt_mask_ex, axis=-1
        )
        # differences with dkd
        pred_teacher_part2 = paddle.mean(pred_teacher_part2, axis=1)
        pred_student_part2 = paddle.mean(pred_student_part2, axis=1)

        # differences with dkd
        nckd_loss = self.kl_loss(pred_student_part2, pred_teacher_part2)
        loss = self.alpha * tckd_loss + self.beta * nckd_loss
        return loss


class KLCTCLogits(nn.Layer):
    def __init__(self, weight=1.0, reduction="mean", mode="mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.eps = 1e-6
        self.t = 0.5
        self.act = nn.Softmax(axis=-1)
        self.use_log = True
        self.mode = mode
        self.ctc_dkd_loss = CTCDKDLoss()

    def kl_loss(self, p1, p2):  # predict, label
        loss = paddle.multiply(
            p2, paddle.log((p2 + self.eps) / (p1 + self.eps) + self.eps)
        )
        bs = loss.shape[0]
        loss = paddle.sum(loss) / bs
        return loss

    def forward_meanmax(self, stu_out, tea_out):
        stu_out = paddle.mean(F.softmax(stu_out / self.t, axis=-1), axis=1)
        tea_out = paddle.mean(F.softmax(tea_out / self.t, axis=-1), axis=1)
        loss = self.kl_loss(stu_out, tea_out)

        return loss

    def forward_meanlog(self, stu_out, tea_out):
        stu_out = paddle.mean(F.softmax(stu_out / self.t, axis=-1), axis=1)
        tea_out = paddle.mean(F.softmax(tea_out / self.t, axis=-1), axis=1)
        if self.use_log is True:
            # for recognition distillation, log is needed for feature map
            log_out1 = paddle.log(stu_out)
            log_out2 = paddle.log(tea_out)
            loss = (
                self._kldiv(log_out1, tea_out) + self._kldiv(log_out2, stu_out)
            ) / 2.0

        return loss

    def forward_sum(self, stu_out, tea_out):
        stu_out = paddle.sum(F.softmax(stu_out / self.t, axis=-1), axis=1)
        tea_out = paddle.sum(F.softmax(tea_out / self.t, axis=-1), axis=1)
        stu_out = paddle.log(stu_out)
        bs = stu_out.shape[0]
        loss = tea_out * (paddle.log(tea_out + self.eps) - stu_out)
        loss = paddle.sum(loss, axis=1) / loss.shape[0]
        return loss

    def _kldiv(self, x, target):
        eps = 1.0e-10
        loss = target * (paddle.log(target + eps) - x)
        loss = paddle.sum(paddle.mean(loss, axis=1)) / loss.shape[0]
        return loss

    def forward(self, stu_out, tea_out, targets=None):
        if self.mode == "log":
            return self.forward_log(stu_out, tea_out)
        elif self.mode == "mean":
            blank_mask = paddle.ones_like(stu_out)
            blank_mask.stop_gradient = True
            blank_mask[:, :, 0] = -1
            stu_out *= blank_mask
            tea_out *= blank_mask
            return self.forward_meanmax(stu_out, tea_out)
        elif self.mode == "sum":
            return self.forward_sum(stu_out, tea_out)
        elif self.mode == "meanlog":
            blank_mask = paddle.ones_like(stu_out)
            blank_mask.stop_gradient = True
            blank_mask[:, :, 0] = -1
            stu_out *= blank_mask
            tea_out *= blank_mask
            return self.forward_meanlog(stu_out, tea_out)
        elif self.mode == "ctcdkd":
            # ignore ctc blank logits
            blank_mask = paddle.ones_like(stu_out)
            blank_mask.stop_gradient = True
            blank_mask[:, :, 0] = -1
            stu_out *= blank_mask
            tea_out *= blank_mask
            return self.ctc_dkd_loss(stu_out, tea_out, targets)
        else:
            raise ValueError("error!!!!!!")

    def forward_log(self, out1, out2):
        if self.act is not None:
            out1 = self.act(out1) + 1e-10
            out2 = self.act(out2) + 1e-10
        if self.use_log is True:
            # for recognition distillation, log is needed for feature map
            log_out1 = paddle.log(out1)
            log_out2 = paddle.log(out2)
            loss = (self._kldiv(log_out1, out2) + self._kldiv(log_out2, out1)) / 2.0

        return loss


class DistillCTCLogits(KLCTCLogits):
    def __init__(
        self, model_name_pairs=[], key=None, name="ctc_logits", reduction="mean"
    ):
        super().__init__(reduction=reduction)
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.key = key
        self.name = name

    def _check_model_name_pairs(self, model_name_pairs):
        if not isinstance(model_name_pairs, list):
            return []
        elif isinstance(model_name_pairs[0], list) and isinstance(
            model_name_pairs[0][0], str
        ):
            return model_name_pairs
        else:
            return [model_name_pairs]

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]

            if self.key is not None:
                out1 = out1[self.key]["ctc"]
                out2 = out2[self.key]["ctc"]

            ctc_label = batch[1]
            loss = super().forward(out1, out2, ctc_label)
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict[
                        "{}_{}_{}".format(self.name, self.model_name_pairs, idx)
                    ] = loss[key]
            else:
                loss_dict["{}_{}".format(self.name, idx)] = loss
        return loss_dict
