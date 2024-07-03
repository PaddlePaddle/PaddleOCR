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
import numpy as np


class VQATokenPad(object):
    def __init__(
        self,
        max_seq_len=512,
        pad_to_max_seq_len=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        truncation_strategy="longest_first",
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        infer_mode=False,
        **kwargs,
    ):
        self.max_seq_len = max_seq_len
        self.pad_to_max_seq_len = max_seq_len
        self.return_attention_mask = return_attention_mask
        self.return_token_type_ids = return_token_type_ids
        self.truncation_strategy = truncation_strategy
        self.return_overflowing_tokens = return_overflowing_tokens
        self.return_special_tokens_mask = return_special_tokens_mask
        self.pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index
        self.infer_mode = infer_mode

    def __call__(self, data):
        needs_to_be_padded = (
            self.pad_to_max_seq_len and len(data["input_ids"]) < self.max_seq_len
        )

        if needs_to_be_padded:
            if "tokenizer_params" in data:
                tokenizer_params = data.pop("tokenizer_params")
            else:
                tokenizer_params = dict(
                    padding_side="right", pad_token_type_id=0, pad_token_id=1
                )

            difference = self.max_seq_len - len(data["input_ids"])
            if tokenizer_params["padding_side"] == "right":
                if self.return_attention_mask:
                    data["attention_mask"] = [1] * len(data["input_ids"]) + [
                        0
                    ] * difference
                if self.return_token_type_ids:
                    data["token_type_ids"] = (
                        data["token_type_ids"]
                        + [tokenizer_params["pad_token_type_id"]] * difference
                    )
                if self.return_special_tokens_mask:
                    data["special_tokens_mask"] = (
                        data["special_tokens_mask"] + [1] * difference
                    )
                data["input_ids"] = (
                    data["input_ids"] + [tokenizer_params["pad_token_id"]] * difference
                )
                if not self.infer_mode:
                    data["labels"] = (
                        data["labels"] + [self.pad_token_label_id] * difference
                    )
                data["bbox"] = data["bbox"] + [[0, 0, 0, 0]] * difference
            elif tokenizer_params["padding_side"] == "left":
                if self.return_attention_mask:
                    data["attention_mask"] = [0] * difference + [1] * len(
                        data["input_ids"]
                    )
                if self.return_token_type_ids:
                    data["token_type_ids"] = [
                        tokenizer_params["pad_token_type_id"]
                    ] * difference + data["token_type_ids"]
                if self.return_special_tokens_mask:
                    data["special_tokens_mask"] = [1] * difference + data[
                        "special_tokens_mask"
                    ]
                data["input_ids"] = [
                    tokenizer_params["pad_token_id"]
                ] * difference + data["input_ids"]
                if not self.infer_mode:
                    data["labels"] = [self.pad_token_label_id] * difference + data[
                        "labels"
                    ]
                data["bbox"] = [[0, 0, 0, 0]] * difference + data["bbox"]
        else:
            if self.return_attention_mask:
                data["attention_mask"] = [1] * len(data["input_ids"])

        for key in data:
            if key in [
                "input_ids",
                "labels",
                "token_type_ids",
                "bbox",
                "attention_mask",
            ]:
                if self.infer_mode:
                    if key != "labels":
                        length = min(len(data[key]), self.max_seq_len)
                        data[key] = data[key][:length]
                    else:
                        continue
                data[key] = np.array(data[key], dtype="int64")
        return data
