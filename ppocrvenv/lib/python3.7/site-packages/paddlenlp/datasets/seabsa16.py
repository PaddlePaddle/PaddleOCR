# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['SeAbsa16']


class SeAbsa16(DatasetBuilder):
    """
    SE-ABSA16_PHNS dataset for Aspect-level Sentiment Classification task.
    More information please refer to 
    https://aistudio.baidu.com/aistudio/competition/detail/50/?isFromLuge=1.

    """

    BUILDER_CONFIGS = {
        # phns is short for phones.
        'phns': {
            'url':
            "https://bj.bcebos.com/paddlenlp/datasets/SE-ABSA16_PHNS.zip",
            'md5': "f5a62548f2fcf73892cacf2cdf159671",
            'splits': {
                'train': [
                    os.path.join('SE-ABSA16_PHNS', 'train.tsv'),
                    'cb4f65aaee59fa76526a0c79b7c12689', (0, 1, 2), 1
                ],
                'test': [
                    os.path.join('SE-ABSA16_PHNS', 'test.tsv'),
                    '7ad80f284e0eccc059ece3ce3d3a173f', (1, 2), 1
                ],
            },
            'labels': ["0", "1"]
        },
        # came is short for cameras.
        'came': {
            'url':
            "https://bj.bcebos.com/paddlenlp/datasets/SE-ABSA16_CAME.zip",
            'md5': "3104e92217bbff80a1ed834230f1df51",
            'splits': {
                'train': [
                    os.path.join('SE-ABSA16_CAME', 'train.tsv'),
                    '8c661c0e83bb34b66c6fbf039c7fae80', (0, 1, 2), 1
                ],
                'test': [
                    os.path.join('SE-ABSA16_CAME', 'test.tsv'),
                    '8b80f77960be55adca1184d7a20501df', (1, 2), 1
                ],
            },
            'labels': ["0", "1"]
        }
    }

    def _get_data(self, mode, **kwargs):
        """Downloads dataset."""
        builder_config = self.BUILDER_CONFIGS[self.name]
        default_root = os.path.join(DATA_HOME, f'SE-ABSA16_{self.name.upper()}')
        filename, data_hash, _, _ = builder_config['splits'][mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            url = builder_config['url']
            md5 = builder_config['md5']
            get_path_from_url(url, DATA_HOME, md5)

        return fullname

    def _read(self, filename, split):
        """Reads data"""
        _, _, field_indices, num_discard_samples = self.BUILDER_CONFIGS[
            self.name]['splits'][split]
        with open(filename, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx < num_discard_samples:
                    continue
                line_stripped = line.strip().split('\t')
                if not line_stripped:
                    continue
                example = [line_stripped[indice] for indice in field_indices]
                if split == 'test':
                    yield {"text": example[0], "text_pair": example[1]}
                else:
                    yield {
                        "text": example[1],
                        "text_pair": example[2],
                        "label": example[0]
                    }

    def get_labels(self):
        """
        Return labels of the SE_ABSA16.
        """
        return self.BUILDER_CONFIGS[self.name]['labels']
