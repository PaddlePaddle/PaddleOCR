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


class Conll2002(DatasetBuilder):
    """
    Named entities are phrases that contain the names of persons, organizations,
    locations, times and quantities. Example: [PER Wolff] , currently a journalist
    in [LOC Argentina] , played with [PER Del Bosque] in the final years of the seventies in [ORG Real Madrid] .
    The shared task of CoNLL-2002 concerns language-independent named entity recognition.
    We will concentrate on four types of named entities: persons, locations, organizations and names of
    miscellaneous entities that do not belong to the previous three groups. The participants of the
    shared task will be offered training and test data for at least two languages.
    They will use the data for developing a named-entity recognition system that includes a machine learning component.
    Information sources other than the training data may be used in this shared task. We are especially interested
    in methods that can use additional unannotated data for improving their performance (for example co-training).
    For more details see https://www.clips.uantwerpen.be/conll2002/ner/
    and https://www.aclweb.org/anthology/W02-2024/
    """
    META_INFO = collections.namedtuple('META_INFO', ('file', 'url', 'md5'))
    BASE_URL = 'https://bj.bcebos.com/paddlenlp/datasets/conll2002/'
    BUILDER_CONFIGS = {
        'es': {
            'splits': {
                'train': META_INFO('esp.train', BASE_URL + 'esp.train',
                                   'c8c6b342371b9de2f83a93767d352c17'),
                'dev': META_INFO('esp.testa', BASE_URL + 'esp.testa',
                                 'de0578160dde26ec68cc580595587dde'),
                'test': META_INFO('esp.testb', BASE_URL + 'esp.testb',
                                  'c8d35f340685a2ce6559ee90d78f9e37')
            },
            'pos_tags': [
                "AO",
                "AQ",
                "CC",
                "CS",
                "DA",
                "DE",
                "DD",
                "DI",
                "DN",
                "DP",
                "DT",
                "Faa",
                "Fat",
                "Fc",
                "Fd",
                "Fe",
                "Fg",
                "Fh",
                "Fia",
                "Fit",
                "Fp",
                "Fpa",
                "Fpt",
                "Fs",
                "Ft",
                "Fx",
                "Fz",
                "I",
                "NC",
                "NP",
                "P0",
                "PD",
                "PI",
                "PN",
                "PP",
                "PR",
                "PT",
                "PX",
                "RG",
                "RN",
                "SP",
                "VAI",
                "VAM",
                "VAN",
                "VAP",
                "VAS",
                "VMG",
                "VMI",
                "VMM",
                "VMN",
                "VMP",
                "VMS",
                "VSG",
                "VSI",
                "VSM",
                "VSN",
                "VSP",
                "VSS",
                "Y",
                "Z",
            ]
        },
        'nl': {
            'splits': {
                'train': META_INFO('ned.train', BASE_URL + 'ned.train',
                                   'b6189d04eb34597d2a98ca5cec477605'),
                'dev': META_INFO('ned.testa', BASE_URL + 'ned.testa',
                                 '626900497823fdbc4f84335518cb85ce'),
                'test': META_INFO('ned.testb', BASE_URL + 'ned.testb',
                                  'c37de92da20c68c6418a73dd42e322dc')
            },
            'pos_tags': [
                "Adj", "Adv", "Art", "Conj", "Int", "Misc", "N", "Num", "Prep",
                "Pron", "Punc", "V"
            ]
        }
    }

    def _get_data(self, mode, **kwargs):
        builder_config = self.BUILDER_CONFIGS[self.name]
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, url, data_hash = builder_config['splits'][mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(url, default_root, data_hash)
        return fullname

    def _read(self, filename, *args):
        with open(filename, 'r', encoding="utf-8") as f:
            tokens = []
            ner_tags = []
            pos_tags = []
            for line in f.readlines():
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield {
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                            "pos_tags": pos_tags
                        }
                        tokens = []
                        ner_tags = []
                        pos_tags = []
                else:
                    # conll2002 tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    pos_tags.append(splits[1])
                    ner_tags.append(splits[2].rstrip())
            # last example
            yield {"tokens": tokens, "ner_tags": ner_tags, "pos_tags": pos_tags}

    def get_labels(self):
        """
        Returns labels of ner tags and pos tags.
        """
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"], \
               self.BUILDER_CONFIGS[self.name]['pos_tags']
