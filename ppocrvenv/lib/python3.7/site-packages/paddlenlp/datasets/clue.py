# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import json
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder


class Clue(DatasetBuilder):
    '''
    `ClUE <https://arxiv.org/abs/2004.05986>`_ is the first large-scale Chinese
    Language Understanding Evaluation(CLUE) benchmark. CLUE is an open-ended,
    community-driven project that brings together 9 tasks spanning several
    well-established single-sentence/sentence-pair classification tasks, as
    well as machine reading comprehension, all on original Chinese text.

    From https://github.com/CLUEbenchmark/CLUE

    AFQMC:
        AFQMC: The Ant Financial Question Matching Corpus3 comes from Ant
        Technology Exploration Conference (ATEC) Developer competition. It is
        a binary classification task that aims to predict whether two sentences
        are semantically similar.
    
    TNEWS:
        TouTiao Text Classification for News Titles2 consists of Chinese news
        published by TouTiao before May 2018, with a total of 73,360 titles.
        Each title is labeled with one of 15 news categories (finance,
        technology, sports, etc.) and the task is to predict which category the
        title belongs to.
    
    IFLYTEK:
        IFLYTEK contains 17,332 app descriptions. The task is to assign each
        description into one of 119 categories, such as food, car rental,
        education, etc. 
    
    OCNLI:
        Original Chinese Natural Language Inference is collected closely
        following procedures of MNLI. OCNLI is composed of 56k inference pairs
        from five genres: news, government, fiction, TV transcripts and
        Telephone transcripts, where the premises are collected from Chinese
        sources, and universities students in language majors are hired to
        write the hypotheses.
    
    CMNLI:
        Chinese Multi-Genre NLI.
    
    CLUEWSC2020:
        The Chinese Winograd Schema Challenge dataset is an anaphora/
        coreference resolution task where the model is asked to decide whether
        a pronoun and a noun (phrase) in a sentence co-refer (binary
        classification), built following similar datasets in English.

    CSL:
        Chinese Scientific Literature dataset contains Chinese paper abstracts
        and their keywords from core journals of China, covering multiple
        fields of natural sciences and social sciences.

    '''

    BUILDER_CONFIGS = {
        'afqmc': {
            'url': "https://bj.bcebos.com/paddlenlp/datasets/afqmc_public.zip",
            'md5': '3377b559bb4e61d03a35282550902ca0',
            'splits': {
                'train': [
                    os.path.join('afqmc_public', 'train.json'),
                    '319cf775353af9473140abca4052b89a',
                ],
                'dev': [
                    os.path.join('afqmc_public', 'dev.json'),
                    '307154b59cb6c3e68a0f39c310bbd364',
                ],
                'test': [
                    os.path.join('afqmc_public', 'test.json'),
                    '94b925f23a9615dd08199c4013f761f4',
                ]
            },
            'labels': ["0", "1"]
        },
        'tnews': {
            'url': "https://bj.bcebos.com/paddlenlp/datasets/tnews_public.zip",
            'md5': '38186ed0a751bc33e3ae0c1b59319777',
            'splits': {
                'train': [
                    os.path.join('tnews_public', 'train.json'),
                    '25c021725309a3330736380a230850fd',
                ],
                'dev': [
                    os.path.join('tnews_public', 'dev.json'),
                    'f0660a3339a32e764075c801b42ece3c',
                ],
                'test': [
                    os.path.join('tnews_public', 'test.json'),
                    '045a6c4f59bf1a066c4a0d7afe6cd2b4',
                ],
                'test1.0': [
                    os.path.join('tnews_public', 'test1.0.json'),
                    '2d1557c7548c72d5a84c47bbbd3a4e85',
                ],
                'labels': [
                    os.path.join('tnews_public', 'labels.json'),
                    'a1a7595e596b202556dedd2a20617769',
                ]
            },
            'labels': [
                "100", "101", "102", "103", "104", "106", "107", "108", "109",
                "110", "112", "113", "114", "115", "116"
            ]
        },
        'iflytek': {
            'url':
            'https://bj.bcebos.com/paddlenlp/datasets/iflytek_public.zip',
            'md5': "19e4b19947db126f69aae18db0da2b87",
            'splits': {
                'train': [
                    os.path.join('iflytek_public', 'train.json'),
                    'fc9a21700c32ee3efee3fc283e9ac560',
                ],
                'dev': [
                    os.path.join('iflytek_public', 'dev.json'),
                    '79b7d95bddeb11cd54198fd077992704',
                ],
                'test': [
                    os.path.join('iflytek_public', 'test.json'),
                    'ea764519ddb4369767d07664afde3325',
                ],
                'labels': [
                    os.path.join('iflytek_public', 'labels.json'),
                    '7f9e794688ffb37fbd42b58325579fdf',
                ]
            },
            'labels': [str(i) for i in range(119)]
        },
        'ocnli': {
            'url': 'https://bj.bcebos.com/paddlenlp/datasets/ocnli_public.zip',
            'md5': 'acb426f6f3345076c6ce79239e7bc307',
            'splits': {
                'train': [
                    os.path.join('ocnli_public', 'train.50k.json'),
                    'd38ec492ef086a894211590a18ab7596',
                ],
                'dev': [
                    os.path.join('ocnli_public', 'dev.json'),
                    '3481b456bee57a3c9ded500fcff6834c',
                ],
                'test': [
                    os.path.join('ocnli_public', 'test.json'),
                    '680ff24e6b3419ff8823859bc17936aa',
                ]
            },
            'labels': ["entailment", "contradiction", "neutral"]
        },
        'cmnli': {
            'url': 'https://bj.bcebos.com/paddlenlp/datasets/cmnli_public.zip',
            'md5': 'e0e8caefd9b3491220c18b466233f2ff',
            'splits': {
                'train': [
                    os.path.join('cmnli_public', 'train.json'),
                    '7d02308650cd2a0e183bf599ca9bb263',
                ],
                'dev': [
                    os.path.join('cmnli_public', 'dev.json'),
                    '0b16a50a297a9afb1ce5385ee4dd3d9c',
                ],
                'test': [
                    os.path.join('cmnli_public', 'test.json'),
                    '804cb0bb67266983d59d1c855e6b03b0',
                ]
            },
            'labels': ["contradiction", "entailment", "neutral"]
        },
        'cluewsc2020': {
            'url':
            'https://bj.bcebos.com/paddlenlp/datasets/cluewsc2020_public.zip',
            'md5': '2e387e20e93eeab0ffaded5b0d2dfd3d',
            'splits': {
                'train': [
                    os.path.join('cluewsc2020_public', 'train.json'),
                    'afd235dcf8cdb89ee1a21d0a4823eecc',
                ],
                'dev': [
                    os.path.join('cluewsc2020_public', 'dev.json'),
                    'bad8cd6fa0916fc37ac96b8ce316714a',
                ],
                'test': [
                    os.path.join('cluewsc2020_public', 'test.json'),
                    '27614454cc26be6fcab5bbd9a45967ff',
                ],
                'test1.0': [
                    os.path.join('cluewsc2020_public', 'test1.0.json'),
                    '0e9e8ffd8ee90ddf1f58d6dc2e02de7b',
                ]
            },
            'labels': ["true", "false"]
        },
        'csl': {
            'url': 'https://bj.bcebos.com/paddlenlp/datasets/csl_public.zip',
            'md5': '394a2ccbf6ddd7e331be4d5d7798f0f6',
            'splits': {
                'train': [
                    os.path.join('csl_public', 'train.json'),
                    'e927948b4e0eb4992fe9f45a77446bf5',
                ],
                'dev': [
                    os.path.join('csl_public', 'dev.json'),
                    '6c2ab8dd3b4785829ead94b05a1cb957',
                ],
                'test': [
                    os.path.join('csl_public', 'test.json'),
                    'ebfb89575355f00dcd9b18f8353547cd',
                ]
            },
            'labels': ["0", "1"]
        },
    }

    def _get_data(self, mode, **kwargs):
        builder_config = self.BUILDER_CONFIGS[self.name]
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash = builder_config['splits'][mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(builder_config['url'], default_root,
                              builder_config['md5'])
        return fullname

    def _read(self, filename, split):
        if self.name == 'cmnli' and split == 'dev' or self.name == 'ocnli' and split in [
                'train', 'dev'
        ]:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    example_dict = json.loads(line.rstrip())
                    if example_dict['label'] == "-":
                        continue
                    yield example_dict
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    yield json.loads(line.rstrip())

    def get_labels(self):
        """
        Returns labels of the Clue task.
        """
        return self.BUILDER_CONFIGS[self.name]['labels']
