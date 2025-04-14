# copyright (c) 2025 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np
import paddle
import paddle.nn as nn
from ppocr.modeling.backbones.rec_svtrnet import DropPath, Identity, Mlp
from ppocr.modeling.heads.rec_nrtr_head import Embeddings


def dim2perm(ndim, dim0, dim1):
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm


init_TruncatedNormal = nn.initializer.TruncatedNormal(std=0.02)


class CrossAttention(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(in_features=dim, out_features=dim, bias_attr=qkv_bias)
        self.kv = nn.Linear(in_features=dim, out_features=dim * 2, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(in_features=dim, out_features=dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def forward(self, q, kv, key_mask=None):
        N, C = tuple(kv.shape)[1:]
        QN = tuple(q.shape)[1]
        q = (
            self.q(q)
            .reshape([-1, QN, self.num_heads, C // self.num_heads])
            .transpose([0, 2, 1, 3])
        )
        q = q * self.scale
        k, v = (
            self.kv(kv)
            .reshape([-1, N, 2, self.num_heads, C // self.num_heads])
            .transpose(perm=[2, 0, 3, 1, 4])
        )
        attn = q.matmul(y=k.transpose(perm=dim2perm(k.ndim, 2, 3)))
        if key_mask is not None:
            attn = attn + key_mask.unsqueeze(axis=1)
        attn = nn.functional.softmax(x=attn, axis=-1)
        if not self.training:
            self.attn_map = attn
        attn = self.attn_drop(attn)
        x = attn.matmul(y=v)
        x = x.transpose(perm=dim2perm(x.ndim, 1, 2)).reshape((-1, QN, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderLayer(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer="nn.LayerNorm",
        epsilon=1e-06,
    ):
        super().__init__()
        self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        self.normkv = eval(norm_layer)(dim, epsilon=epsilon)
        self.mixer = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, q, kv, key_mask=None):
        x1 = q + self.drop_path(self.mixer(self.norm1(q), self.normkv(kv), key_mask))
        x = x1 + self.drop_path(self.mlp(self.norm2(x1)))
        return x


class CMFFLayer(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        epsilon=1e-06,
    ):
        super().__init__()
        self.normq1 = nn.LayerNorm(normalized_shape=dim, epsilon=epsilon)
        self.normkv1 = nn.LayerNorm(normalized_shape=dim, epsilon=epsilon)
        self.images_to_question_cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.normq2 = nn.LayerNorm(normalized_shape=dim, epsilon=epsilon)
        self.normkv2 = nn.LayerNorm(normalized_shape=dim, epsilon=epsilon)
        self.question_to_images_cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.normmlp = nn.LayerNorm(normalized_shape=dim, epsilon=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, question_f, prompt_f, visual_f, mask=None):
        query_add = paddle.concat(x=[question_f, prompt_f, visual_f], axis=1)
        query_add = query_add + self.drop_path(
            self.images_to_question_cross_attn(
                self.normq1(query_add), self.normkv1(prompt_f), mask
            )
        )
        query_add = query_add + self.drop_path(
            self.question_to_images_cross_attn(
                self.normq2(query_add),
                self.normkv2(query_add[:, -tuple(visual_f.shape)[1] :, :]),
            )
        )
        query_updated = query_add + self.drop_path(self.mlp(self.normmlp(query_add)))
        question_f_updated = query_updated[:, : tuple(question_f.shape)[1], :]
        prompt_f_updated = query_updated[
            :, tuple(question_f.shape)[1] : -tuple(visual_f.shape)[1], :
        ]
        visual_f_updated = query_updated[:, -tuple(visual_f.shape)[1] :, :]
        return question_f_updated, prompt_f_updated, visual_f_updated


class IGTRHead(nn.Layer):
    def __init__(
        self,
        in_channels,
        dim,
        out_channels,
        num_layer=2,
        drop_path_rate=0.1,
        max_len=25,
        vis_seq=50,
        ch=False,
        ar=False,
        refine_iter=0,
        quesall=True,
        next_pred=False,
        ds=False,
        pos2d=False,
        check_search=False,
        max_size=[8, 32],
        **kwargs,
    ):
        super(IGTRHead, self).__init__()
        self.out_channels = out_channels
        self.dim = dim
        self.max_len = max_len + 3
        self.ch = ch
        self.char_embed = Embeddings(
            d_model=dim, vocab=self.out_channels, scale_embedding=True
        )
        self.ignore_index = out_channels - 1
        self.ar = ar
        self.refine_iter = refine_iter
        self.bos = self.out_channels - 2
        self.eos = 0
        self.next_pred = next_pred
        self.quesall = quesall
        self.check_search = check_search
        dpr = np.linspace(0, drop_path_rate, num_layer + 2)
        self.cmff_decoder = nn.LayerList(
            sublayers=[
                CMFFLayer(
                    dim=dim,
                    num_heads=dim // 32,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop_path=dpr[i],
                )
                for i in range(num_layer)
            ]
        )
        self.answer_to_question_layer = DecoderLayer(
            dim=dim,
            num_heads=dim // 32,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path=dpr[-2],
        )
        self.answer_to_image_layer = DecoderLayer(
            dim=dim,
            num_heads=dim // 32,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path=dpr[-1],
        )
        self.char_pos_embed = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=[self.max_len, dim], dtype="float32"),
            trainable=True,
        )
        self.appear_num_embed = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=[self.max_len, dim], dtype="float32"),
            trainable=True,
        )
        self.ds = ds
        self.pos2d = pos2d
        if not ds:
            self.vis_pos_embed = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.zeros(shape=[1, vis_seq, dim], dtype="float32"),
                trainable=True,
            )
            init_TruncatedNormal(self.vis_pos_embed)
        elif pos2d:
            pos_embed = paddle.zeros(
                shape=[1, max_size[0] * max_size[1], dim], dtype="float32"
            )
            init_TruncatedNormal(pos_embed)
            self.vis_pos_embed = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=pos_embed.transpose(perm=dim2perm(pos_embed.ndim, 1, 2)).reshape(
                    [1, dim, max_size[0], max_size[1]]
                ),
                trainable=True,
            )
        self.prompt_pos_embed = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=[1, 6, dim], dtype="float32"), trainable=True
        )
        self.answer_query = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=[1, 1, dim], dtype="float32"), trainable=True
        )
        self.norm_pred = nn.LayerNorm(normalized_shape=dim, epsilon=1e-06)
        self.ques1_head = nn.Linear(in_features=dim, out_features=self.out_channels - 2)
        self.ques2_head = nn.Linear(
            in_features=dim, out_features=self.max_len, bias_attr=False
        )
        self.ques3_head = nn.Linear(in_features=dim, out_features=self.max_len - 1)
        self.ques4_head = nn.Linear(in_features=dim, out_features=self.max_len - 1)

        init_TruncatedNormal(self.char_pos_embed)
        init_TruncatedNormal(self.appear_num_embed)
        init_TruncatedNormal(self.answer_query)
        init_TruncatedNormal(self.prompt_pos_embed)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init_TruncatedNormal(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init_Constant = nn.initializer.Constant(value=0.0)
                init_Constant(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init_Constant = nn.initializer.Constant(value=0.0)
            init_Constant(m.bias)
            init_Constant = nn.initializer.Constant(value=1.0)
            init_Constant(m.weight)

    def question_encoder(self, targets, train_i):
        (
            prompt_pos_idx,
            prompt_char_idx,
            ques_pos_idx,
            ques1_answer,
            ques2_char_idx,
            ques2_answer,
            ques4_char_num,
            ques_len,
            ques2_len,
            prompt_len,
        ) = targets
        max_ques_len = paddle.max(x=ques_len)
        max_ques2_len = paddle.max(x=ques2_len)
        max_prompt_len = paddle.max(x=prompt_len)
        if self.next_pred and (train_i == 2 or train_i == 3):
            prompt_pos = self.prompt_pos_embed
            prompt_char_idx = prompt_char_idx[:, :max_prompt_len]
        else:
            prompt_pos = nn.functional.embedding(
                x=prompt_pos_idx[:, :max_prompt_len], weight=self.char_pos_embed
            )
            prompt_char_idx = prompt_char_idx[:, :max_prompt_len]
        prompt_char = self.char_embed(prompt_char_idx)
        prompt = prompt_pos + prompt_char
        mask_1234 = paddle.where(
            condition=prompt_char_idx == self.ignore_index,
            x=float("-inf"),
            y=0.0,
        )
        mask_1234 = paddle.cast(mask_1234.unsqueeze(axis=1), paddle.float32)
        # mask_1234 = mask_1234
        ques1 = nn.functional.embedding(
            x=ques_pos_idx[:, :max_ques_len], weight=self.char_pos_embed
        )

        ques1_answer = ques1_answer[:, :max_ques_len]
        if self.quesall or train_i == 0:
            ques2_char = self.char_embed(ques2_char_idx[:, :max_ques2_len, (1)])
            ques2 = ques2_char + nn.functional.embedding(
                x=ques2_char_idx[:, :max_ques2_len, (0)], weight=self.char_pos_embed
            )
            ques2_answer = ques2_answer[:, :max_ques2_len]
            # print(ques2_char_idx[:, :max_ques2_len, (0)].shape, self.ques2_head.weight.shape)
            ques2_head = nn.functional.embedding(
                x=ques2_char_idx[:, :max_ques2_len, (0)],
                weight=self.ques2_head.weight.transpose([1, 0]),
            )
            # print(ques2_head)
            ques4_char = self.char_embed(ques1_answer)
            ques4_ap_num = nn.functional.embedding(
                x=ques4_char_num[:, :max_ques_len], weight=self.appear_num_embed
            )
            ques4 = ques4_char + ques4_ap_num
            ques4_answer = ques_pos_idx[:, :max_ques_len]
            return (
                prompt,
                ques1,
                ques2,
                ques2_head,
                ques4,
                ques1_answer,
                ques2_answer,
                ques4_answer,
                mask_1234,
            )
        else:
            return prompt, ques1, ques1_answer, mask_1234

    def forward(self, x, targets=None):
        if self.training:
            return self.forward_train(x, targets)
        else:
            return self.forward_test(x)

    def forward_test(self, x):
        if not self.ds:
            visual_f = x + self.vis_pos_embed
        elif self.pos2d:
            x = x + self.vis_pos_embed[:, :, : tuple(x.shape)[2], : tuple(x.shape)[3]]
            visual_f = x.flatten(start_axis=2).transpose(
                perm=dim2perm(x.flatten(start_axis=2).ndim, 1, 2)
            )
        else:
            visual_f = x
        bs = tuple(x.shape)[0]
        prompt_bos = self.char_embed(
            paddle.full(shape=[bs, 1], fill_value=self.bos, dtype="int64")
        ) + self.char_pos_embed[:1, :].unsqueeze(axis=0)
        ques_all = paddle.tile(
            x=self.char_pos_embed.unsqueeze(axis=0), repeat_times=(bs, 1, 1)
        )
        if not self.ar:
            if self.check_search:
                tgt_in = paddle.full(
                    shape=(bs, self.max_len),
                    fill_value=self.ignore_index,
                    dtype="int64",
                )
                tgt_in[:, (0)] = self.bos
                logits = []
                for j in range(1, self.max_len):
                    visual_f_check = visual_f
                    ques_check_i = ques_all[:, j : j + 1, :] + self.char_embed(
                        paddle.arange(end=self.out_channels - 2)
                    ).unsqueeze(axis=0)
                    prompt_check = ques_all[:, :j] + self.char_embed(tgt_in[:, :j])
                    mask = paddle.where(
                        condition=(tgt_in[:, :j] == self.eos)
                        .astype(dtype="int32")
                        .cumsum(axis=-1)
                        > 0,
                        x=float("-inf"),
                        y=0,
                    )
                    for layer in self.cmff_decoder:
                        ques_check_i, prompt_check, visual_f_check = layer(
                            ques_check_i,
                            prompt_check,
                            visual_f_check,
                            mask.unsqueeze(axis=1),
                        )
                    answer_query_i = self.answer_to_question_layer(
                        ques_check_i, prompt_check, mask.unsqueeze(axis=1)
                    )
                    answer_pred_i = self.norm_pred(
                        self.answer_to_image_layer(answer_query_i, visual_f_check)
                    )
                    fc_2 = self.ques2_head.weight[j : j + 1].unsqueeze(axis=0)
                    fc_2 = fc_2.tile(repeat_times=[bs, 1, 1])
                    p_i = fc_2 @ answer_pred_i.transpose(
                        perm=dim2perm(answer_pred_i.ndim, 1, 2)
                    )
                    logits.append(p_i)
                    if j < self.max_len - 1:
                        tgt_in[:, (j)] = p_i.squeeze().argmax(axis=-1)
                        if (
                            (tgt_in == self.eos)
                            .astype("bool")
                            .any(axis=-1)
                            .astype("bool")
                            .all()
                        ):
                            break
                logits = paddle.concat(x=logits, axis=1)
            else:
                ques_pd = ques_all[:, 1:, :]
                prompt_pd = prompt_bos
                visual_f_pd = visual_f
                for layer in self.cmff_decoder:
                    ques_pd, prompt_pd, visual_f_pd = layer(
                        ques_pd, prompt_pd, visual_f_pd
                    )
                answer_query_pd = self.answer_to_question_layer(ques_pd, prompt_pd)
                answer_feats_pd = self.norm_pred(
                    self.answer_to_image_layer(answer_query_pd, visual_f_pd)
                )
                logits = self.ques1_head(answer_feats_pd)
        elif self.next_pred:
            ques_pd_1 = ques_all[:, 1:2, :]
            prompt_pd = prompt_bos
            visual_f_pd = visual_f
            for layer in self.cmff_decoder:
                ques_pd_1, prompt_pd, visual_f_pd = layer(
                    ques_pd_1, prompt_pd, visual_f_pd
                )
            answer_query_pd = self.answer_to_question_layer(ques_pd_1, prompt_pd)
            answer_feats_pd = self.norm_pred(
                self.answer_to_image_layer(answer_query_pd, visual_f_pd)
            )
            logits_pd_1 = self.ques1_head(answer_feats_pd)
            ques_next = (
                self.char_pos_embed[-2:-1, :]
                .unsqueeze(axis=0)
                .tile(repeat_times=[bs, 1, 1])
            )
            prompt_next_bos = (
                self.char_embed(
                    paddle.full(shape=[bs, 1], fill_value=self.bos, dtype="int64")
                )
                + self.prompt_pos_embed[:, :1, :]
            )
            pred_prob, pred_id = nn.functional.softmax(x=logits_pd_1, axis=-1).max(-1)
            pred_prob_list = [pred_prob]
            pred_id_list = [pred_id]
            for j in range(1, 70):
                prompt_next_1 = (
                    self.char_embed(pred_id)
                    + self.prompt_pos_embed[:, -1 * tuple(pred_id.shape)[1] :, :]
                )
                prompt_next = paddle.concat(x=[prompt_next_bos, prompt_next_1], axis=1)
                ques_next_i = ques_next
                visual_f_i = visual_f
                for layer in self.cmff_decoder:
                    ques_next_i, prompt_next, visual_f_pd = layer(
                        ques_next_i, prompt_next, visual_f_i
                    )
                answer_query_next_i = self.answer_to_question_layer(
                    ques_next_i, prompt_next
                )
                answer_feats_next_i = self.norm_pred(
                    self.answer_to_image_layer(answer_query_next_i, visual_f_i)
                )
                logits_next_i = self.ques1_head(answer_feats_next_i)
                pred_prob_i, pred_id_i = nn.functional.softmax(
                    x=logits_next_i, axis=-1
                ).max(-1)
                pred_prob_list.append(pred_prob_i)
                pred_id_list.append(pred_id_i)
                if (
                    (paddle.concat(x=pred_id_list, axis=1) == self.eos)
                    .astype("bool")
                    .any(axis=-1)
                    .astype("bool")
                    .all()
                ):
                    break
                if tuple(pred_id.shape)[1] >= 5:
                    pred_id = paddle.concat(x=[pred_id[:, 1:], pred_id_i], axis=1)
                else:
                    pred_id = paddle.concat(x=[pred_id, pred_id_i], axis=1)
            return [
                paddle.concat(x=pred_id_list, axis=1),
                paddle.concat(x=pred_prob_list, axis=1),
            ]
        else:
            tgt_in = paddle.full(
                shape=(bs, self.max_len), fill_value=self.ignore_index, dtype="int64"
            )
            tgt_in[:, (0)] = self.bos
            logits = []
            for j in range(1, self.max_len):
                visual_f_ar = visual_f
                ques_i = ques_all[:, j : j + 1, :]
                prompt_ar = ques_all[:, :j] + self.char_embed(tgt_in[:, :j])
                mask = paddle.where(
                    condition=(tgt_in[:, :j] == self.eos)
                    .astype(dtype="int32")
                    .cumsum(axis=-1)
                    > 0,
                    x=float("-inf"),
                    y=0,
                )
                for layer in self.cmff_decoder:
                    ques_i, prompt_ar, visual_f_ar = layer(
                        ques_i, prompt_ar, visual_f_ar, mask.unsqueeze(axis=1)
                    )
                answer_query_i = self.answer_to_question_layer(
                    ques_i, prompt_ar, mask.unsqueeze(axis=1)
                )
                answer_pred_i = self.norm_pred(
                    self.answer_to_image_layer(answer_query_i, visual_f_ar)
                )
                p_i = self.ques1_head(answer_pred_i)
                logits.append(p_i)
                if j < self.max_len - 1:
                    tgt_in[:, (j)] = p_i.squeeze().argmax(axis=-1)
                    if (
                        (tgt_in == self.eos)
                        .astype("bool")
                        .any(axis=-1)
                        .astype("bool")
                        .all()
                    ):
                        break
            logits = paddle.concat(x=logits, axis=1)
        if self.refine_iter > 0:
            pred_probs, pred_idxs = nn.functional.softmax(x=logits, axis=-1).max(-1)
            for i in range(self.refine_iter):
                mask_check = (pred_idxs == self.eos).astype(dtype="int32").cumsum(
                    axis=-1
                ) <= 1
                ques_check_all = (
                    self.char_embed(pred_idxs)
                    + ques_all[:, 1 : tuple(pred_idxs.shape)[1] + 1, :]
                )
                prompt_check = prompt_bos
                visual_f_check = visual_f
                ques_check = ques_check_all
                for layer in self.cmff_decoder:
                    ques_check, prompt_check, visual_f_check = layer(
                        ques_check, prompt_check, visual_f_check
                    )
                answer_query_check = self.answer_to_question_layer(
                    ques_check, prompt_check
                )
                answer_pred_check = self.norm_pred(
                    self.answer_to_image_layer(answer_query_check, visual_f_check)
                )
                ques2_head = self.ques2_head.weight[
                    1 : tuple(pred_idxs.shape)[1] + 1, :
                ]
                ques2_head = paddle.tile(
                    x=ques2_head.unsqueeze(axis=0), repeat_times=[bs, 1, 1]
                )
                answer2_pred = answer_pred_check.matmul(
                    y=ques2_head.transpose(perm=dim2perm(ques2_head.ndim, 1, 2))
                )
                diag_mask = (
                    paddle.eye(num_rows=tuple(answer2_pred.shape)[1])
                    .unsqueeze(axis=0)
                    .tile(repeat_times=[bs, 1, 1])
                )
                answer2_pred = (
                    nn.functional.sigmoid(x=(answer2_pred * diag_mask).sum(axis=-1))
                    * mask_check
                )
                check_result = answer2_pred < 0.9
                prompt_refine = paddle.concat(x=[prompt_bos, ques_check_all], axis=1)
                mask_refine = paddle.where(
                    condition=check_result, x=float("-inf"), y=0
                ) + paddle.where(
                    condition=(pred_idxs == self.eos)
                    .astype(dtype="int32")
                    .cumsum(axis=-1)
                    < 1,
                    x=0,
                    y=float("-inf"),
                )
                mask_refine = paddle.concat(
                    x=[paddle.zeros(shape=[bs, 1]), mask_refine], axis=1
                ).unsqueeze(axis=1)
                ques_refine = ques_all[:, 1 : tuple(pred_idxs.shape)[1] + 1, :]
                visual_f_refine = visual_f
                for layer in self.cmff_decoder:
                    ques_refine, prompt_refine, visual_f_refine = layer(
                        ques_refine, prompt_refine, visual_f_refine, mask_refine
                    )
                answer_query_refine = self.answer_to_question_layer(
                    ques_refine, prompt_refine, mask_refine
                )
                answer_pred_refine = self.norm_pred(
                    self.answer_to_image_layer(answer_query_refine, visual_f_refine)
                )
                answer_refine = self.ques1_head(answer_pred_refine)
                refine_probs, refine_idxs = nn.functional.softmax(
                    x=answer_refine, axis=-1
                ).max(-1)
                pred_idxs_refine = paddle.where(
                    condition=check_result, x=refine_idxs, y=pred_idxs
                )
                pred_idxs = paddle.where(
                    condition=mask_check, x=pred_idxs_refine, y=pred_idxs
                )
                pred_probs_refine = paddle.where(
                    condition=check_result, x=refine_probs, y=pred_probs
                )
                pred_probs = paddle.where(
                    condition=mask_check, x=pred_probs_refine, y=pred_probs
                )
            return [pred_idxs, pred_probs]
        return nn.functional.softmax(x=logits, axis=-1)

    def forward_train(self, x, targets=None):
        bs = tuple(x.shape)[0]
        answer_token = paddle.tile(x=self.answer_query, repeat_times=(bs, 1, 1))
        if self.ch:
            ques3 = self.char_embed(targets[7][:, :, (0)]) + answer_token
            ques3_answer = targets[7][:, :, (1)]
        else:
            ques3 = (
                self.char_embed(paddle.arange(end=self.out_channels - 2)).unsqueeze(
                    axis=0
                )
                + answer_token
            )
            ques3_answer = targets[7]
        loss1_list = []
        loss2_list = []
        loss3_list = []
        loss4_list = []
        sampler1_num = 0
        sampler2_num = 0
        sampler3_num = 0
        sampler4_num = 0
        if not self.ds:
            visual_f = x + self.vis_pos_embed
        elif self.pos2d:
            x = x + self.vis_pos_embed[:, :, : tuple(x.shape)[2], : tuple(x.shape)[3]]
            visual_f = x.flatten(start_axis=2).transpose(
                perm=dim2perm(x.flatten(start_axis=2).ndim, 1, 2)
            )
        else:
            visual_f = x
        train_i = 0
        for target_ in zip(
            targets[1].transpose(perm=dim2perm(targets[1].ndim, 0, 1)),
            targets[2].transpose(perm=dim2perm(targets[2].ndim, 0, 1)),
            targets[3].transpose(perm=dim2perm(targets[3].ndim, 0, 1)),
            targets[4].transpose(perm=dim2perm(targets[4].ndim, 0, 1)),
            targets[5].transpose(perm=dim2perm(targets[5].ndim, 0, 1)),
            targets[6].transpose(perm=dim2perm(targets[6].ndim, 0, 1)),
            targets[8].transpose(perm=dim2perm(targets[8].ndim, 0, 1)),
            targets[9].transpose(perm=dim2perm(targets[9].ndim, 0, 1)),
            targets[10].transpose(perm=dim2perm(targets[10].ndim, 0, 1)),
            targets[11].transpose(perm=dim2perm(targets[11].ndim, 0, 1)),
        ):
            visual_f_1234 = visual_f
            if self.quesall or train_i == 0:
                (
                    prompt,
                    ques1,
                    ques2,
                    ques2_head,
                    ques4,
                    ques1_answer,
                    ques2_answer,
                    ques4_answer,
                    mask_1234,
                ) = self.question_encoder(target_, train_i)
                prompt_1234 = prompt
                ques_1234 = paddle.concat(x=[ques1, ques2, ques3, ques4], axis=1)
                for layer in self.cmff_decoder:
                    ques_1234, prompt_1234, visual_f_1234 = layer(
                        ques_1234, prompt_1234, visual_f_1234, mask_1234
                    )
                answer_query_1234 = self.answer_to_question_layer(
                    ques_1234, prompt_1234, mask_1234
                )
                answer_feats_1234 = self.norm_pred(
                    self.answer_to_image_layer(answer_query_1234, visual_f_1234)
                )
                answer_feats_1 = answer_feats_1234[:, : tuple(ques1.shape)[1], :]
                answer_feats_2 = answer_feats_1234[
                    :,
                    tuple(ques1.shape)[1] : tuple(ques1.shape)[1]
                    + tuple(ques2.shape)[1],
                    :,
                ]
                answer_feats_3 = answer_feats_1234[
                    :,
                    tuple(ques1.shape)[1]
                    + tuple(ques2.shape)[1] : -tuple(ques4.shape)[1],
                    :,
                ]
                answer_feats_4 = answer_feats_1234[:, -tuple(ques4.shape)[1] :, :]
                answer1_pred = self.ques1_head(answer_feats_1)
                if train_i == 0:
                    logits = answer1_pred
                n = (ques1_answer != self.ignore_index).sum().item()
                loss1 = n * nn.functional.cross_entropy(
                    input=answer1_pred.flatten(start_axis=0, stop_axis=1),
                    label=ques1_answer.flatten(start_axis=0, stop_axis=1),
                    ignore_index=self.ignore_index,
                    reduction="mean",
                )
                sampler1_num += n
                loss1_list.append(loss1)
                answer2_pred = answer_feats_2.matmul(
                    y=ques2_head.transpose(perm=dim2perm(ques2_head.ndim, 1, 2))
                )
                diag_mask = (
                    paddle.eye(num_rows=tuple(answer2_pred.shape)[1])
                    .unsqueeze(axis=0)
                    .tile(repeat_times=[bs, 1, 1])
                )
                answer2_pred = (answer2_pred * diag_mask).sum(axis=-1)
                ques2_answer = ques2_answer.flatten(start_axis=0, stop_axis=1)
                non_pad_mask = paddle.not_equal(
                    x=ques2_answer,
                    y=paddle.to_tensor(self.ignore_index, dtype=paddle.float32),
                )
                n = non_pad_mask.sum().item()
                ques2_answer = paddle.where(
                    condition=ques2_answer == self.ignore_index,
                    x=paddle.to_tensor(0.0, dtype=paddle.float32),
                    y=ques2_answer,
                )
                loss2_none = nn.functional.binary_cross_entropy_with_logits(
                    logit=answer2_pred.flatten(start_axis=0, stop_axis=1),
                    label=ques2_answer,
                    reduction="none",
                )
                loss2 = n * loss2_none.masked_select(mask=non_pad_mask).mean()
                sampler2_num += n
                loss2_list.append(loss2)
                answer3_pred = self.ques3_head(answer_feats_3)
                n = (ques3_answer != self.ignore_index).sum().item()
                loss3 = n * nn.functional.cross_entropy(
                    input=answer3_pred.flatten(start_axis=0, stop_axis=1),
                    label=ques3_answer.flatten(start_axis=0, stop_axis=1),
                    reduction="mean",
                )
                sampler3_num += n
                loss3_list.append(loss3)
                answer4_pred = self.ques4_head(answer_feats_4)
                n = (ques4_answer != self.max_len - 1).sum().item()
                loss4 = n * nn.functional.cross_entropy(
                    input=answer4_pred.flatten(start_axis=0, stop_axis=1),
                    label=ques4_answer.flatten(start_axis=0, stop_axis=1),
                    ignore_index=self.max_len - 1,
                    reduction="mean",
                )
                sampler4_num += n
                loss4_list.append(loss4)
            else:
                prompt, ques1, ques1_answer, mask_1234 = self.question_encoder(
                    target_, train_i
                )
                prompt_1234 = prompt
                for layer in self.cmff_decoder:
                    ques1, prompt_1234, visual_f_1234 = layer(
                        ques1, prompt_1234, visual_f_1234, mask_1234
                    )
                answer_query_1 = self.answer_to_question_layer(
                    ques1, prompt_1234, mask_1234
                )
                answer_feats_1 = self.norm_pred(
                    self.answer_to_image_layer(answer_query_1, visual_f_1234)
                )
                answer1_pred = self.ques1_head(answer_feats_1)
                n = (ques1_answer != self.ignore_index).sum().item()
                loss1 = n * nn.functional.cross_entropy(
                    input=answer1_pred.flatten(start_axis=0, stop_axis=1),
                    label=ques1_answer.flatten(start_axis=0, stop_axis=1),
                    ignore_index=self.ignore_index,
                    reduction="mean",
                )
                sampler1_num += n
                loss1_list.append(loss1)
            train_i += 1
        loss_list = [
            sum(loss1_list) / sampler1_num,
            sum(loss2_list) / sampler2_num,
            sum(loss3_list) / sampler3_num,
            sum(loss4_list) / sampler4_num,
        ]
        loss = {
            "loss": sum(loss_list),
            "loss1": loss_list[0],
            "loss2": loss_list[1],
            "loss3": loss_list[2],
            "loss4": loss_list[3],
        }
        return [loss, logits]
