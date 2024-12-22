# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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
from datasets import load_dataset

import paddle
from paddle.io import Dataset
from .imaug.label_ops import MixTexLabelEncode
from .imaug import transform, create_operators

from paddlenlp.transformers.roberta.tokenizer import RobertaTokenizer


class MixTexDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(MixTexDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()

        global_config = config["Global"]
        dataset_config = config[mode]["dataset"]
        loader_config = config[mode]["loader"]

        self.data_dir = dataset_config["data_dir"]
        self.image_size = global_config["d2s_train_image_shape"]
        self.batchsize = dataset_config["batch_size_per_pair"]
        self.max_seq_len = global_config["max_seq_len"]
        self.rec_char_dict_path = global_config["rec_char_dict_path"]
        self.tokenizer = MixTexLabelEncode(self.rec_char_dict_path)

        self.dataframe = load_dataset(self.data_dir)

        self.ops = create_operators(dataset_config["transforms"], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx", 2)
        self.need_reset = True

    def __getitem__(self, idx):
        image = self.dataframe["train"][idx]["image"].convert("RGB")
        image = np.asarray(image)
        data = {"image": image}
        pixel_values = transform(data, self.ops)
        target_text = self.dataframe["train"][idx]["text"]
        target = self.tokenizer.tokenizer(
            target_text,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
        ).input_ids
        labels = [
            label if label != self.tokenizer.tokenizer.pad_token_id else 1
            for label in target
        ]
        labels = np.array(labels)

        pixel_values = np.array(pixel_values).reshape(
            (
                len(pixel_values),
                pixel_values[0].shape[0],
                pixel_values[0].shape[1],
                pixel_values[0].shape[2],
            )
        )
        return (pixel_values, labels)

    def __len__(self):
        return len(self.dataframe["train"])
