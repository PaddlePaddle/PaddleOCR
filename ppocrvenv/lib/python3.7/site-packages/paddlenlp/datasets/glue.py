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


class Glue(DatasetBuilder):
    '''
    The General Language Understanding Evaluation (GLUE) benchmark is a collection
    of resources for training, evaluating, and analyzing natural language 
    understanding systems.
    From https://gluebenchmark.com/tasks

    CoLA:
        The Corpus of Linguistic Acceptability (Warstadt et al., 2018) consists of
        English acceptability judgments drawn from books and journal articles on
        linguistic theory.
        Each example is a sequence of words annotated with whether it is a
        grammatical English sentence. 
    
    SST2:
        The Stanford Sentiment Treebank (Socher et al., 2013) consists of sentences
        from movie reviews and human annotations of their sentiment.
    
    MRPC:
        The Microsoft Research Paraphrase Corpus dataset.
    
    STSB:
        The Semantic Textual Similarity Benchmark (Cer et al., 2017) is a
        collection of sentence pairs drawn from news headlines, video and image
        captions, and natural language inference data. Each pair is human-annotated
        with a similarity score from 1 to 5.
    
    QQP:
        The Quora Question Pairs dataset is a collection of question pairs from the
        community question-answering website Quora.

    MNLI:
        The Multi-Genre Natural Language Inference Corpus (Williams et al., 2018)
        is a crowdsourced collection of sentence pairs with textual entailment
        annotations.
    
    QNLI:
        The Question-answering NLI dataset converted from Stanford Question
        Answering Dataset (Rajpurkar et al. 2016).

    RTE:
        The Recognizing Textual Entailment (RTE) datasets come from a series of
        annual textual entailment challenges (RTE1, RTE2, RTE3, and RTE5).

    WNLI:
        The Winograd NLI dataset converted from the dataset in Winograd Schema
        Challenge (Levesque et al., 2011).
    '''

    BUILDER_CONFIGS = {
        'cola': {
            'url': "https://bj.bcebos.com/dataset/glue/CoLA.zip",
            'md5': 'b178a7c2f397b0433c39c7caf50a3543',
            'splits': {
                'train': [
                    os.path.join('CoLA', 'train.tsv'),
                    'c79d4693b8681800338aa044bf9e797b', (3, 1), 0
                ],
                'dev': [
                    os.path.join('CoLA', 'dev.tsv'),
                    'c5475ccefc9e7ca0917294b8bbda783c', (3, 1), 0
                ],
                'test': [
                    os.path.join('CoLA', 'test.tsv'),
                    'd8721b7dedda0dcca73cebb2a9f4259f', (1, ), 1
                ]
            },
            'labels': ["0", "1"]
        },
        'sst-2': {
            'url': "https://bj.bcebos.com/dataset/glue/SST.zip",
            'md5': '9f81648d4199384278b86e315dac217c',
            'splits': {
                'train': [
                    os.path.join('SST-2', 'train.tsv'),
                    'da409a0a939379ed32a470bc0f7fe99a', (0, 1), 1
                ],
                'dev': [
                    os.path.join('SST-2', 'dev.tsv'),
                    '268856b487b2a31a28c0a93daaff7288', (0, 1), 1
                ],
                'test': [
                    os.path.join('SST-2', 'test.tsv'),
                    '3230e4efec76488b87877a56ae49675a', (1, ), 1
                ]
            },
            'labels': ["0", "1"]
        },
        'sts-b': {
            'url': 'https://bj.bcebos.com/dataset/glue/STS.zip',
            'md5': 'd573676be38f1a075a5702b90ceab3de',
            'splits': {
                'train': [
                    os.path.join('STS-B', 'train.tsv'),
                    '4f7a86dde15fe4832c18e5b970998672', (7, 8, 9), 1
                ],
                'dev': [
                    os.path.join('STS-B', 'dev.tsv'),
                    '5f4d6b0d2a5f268b1b56db773ab2f1fe', (7, 8, 9), 1
                ],
                'test': [
                    os.path.join('STS-B', 'test.tsv'),
                    '339b5817e414d19d9bb5f593dd94249c', (7, 8), 1
                ]
            },
            'labels': None
        },
        'qqp': {
            'url': 'https://dataset.bj.bcebos.com/glue/QQP.zip',
            'md5': '884bf26e39c783d757acc510a2a516ef',
            'splits': {
                'train': [
                    os.path.join('QQP', 'train.tsv'),
                    'e003db73d277d38bbd83a2ef15beb442', (3, 4, 5), 1
                ],
                'dev': [
                    os.path.join('QQP', 'dev.tsv'),
                    'cff6a448d1580132367c22fc449ec214', (3, 4, 5), 1
                ],
                'test': [
                    os.path.join('QQP', 'test.tsv'),
                    '73de726db186b1b08f071364b2bb96d0', (1, 2), 1
                ]
            },
            'labels': ["0", "1"]
        },
        'mnli': {
            'url': 'https://bj.bcebos.com/dataset/glue/MNLI.zip',
            'md5': 'e343b4bdf53f927436d0792203b9b9ff',
            'splits': {
                'train': [
                    os.path.join('MNLI', 'train.tsv'),
                    '220192295e23b6705f3545168272c740', (8, 9, 11), 1
                ],
                'dev_matched': [
                    os.path.join('MNLI', 'dev_matched.tsv'),
                    'c3fa2817007f4cdf1a03663611a8ad23', (8, 9, 15), 1
                ],
                'dev_mismatched': [
                    os.path.join('MNLI', 'dev_mismatched.tsv'),
                    'b219e6fe74e4aa779e2f417ffe713053', (8, 9, 15), 1
                ],
                'test_matched': [
                    os.path.join('MNLI', 'test_matched.tsv'),
                    '33ea0389aedda8a43dabc9b3579684d9', (8, 9), 1
                ],
                'test_mismatched': [
                    os.path.join('MNLI', 'test_mismatched.tsv'),
                    '7d2f60a73d54f30d8a65e474b615aeb6', (8, 9), 1
                ]
            },
            'labels': ["contradiction", "entailment", "neutral"]
        },
        'qnli': {
            'url': 'https://bj.bcebos.com/dataset/glue/QNLI.zip',
            'md5': 'b4efd6554440de1712e9b54e14760e82',
            'splits': {
                'train': [
                    os.path.join('QNLI', 'train.tsv'),
                    '5e6063f407b08d1f7c7074d049ace94a', (1, 2, 3), 1
                ],
                'dev': [
                    os.path.join('QNLI', 'dev.tsv'),
                    '1e81e211959605f144ba6c0ad7dc948b', (1, 2, 3), 1
                ],
                'test': [
                    os.path.join('QNLI', 'test.tsv'),
                    'f2a29f83f3fe1a9c049777822b7fa8b0', (1, 2), 1
                ]
            },
            'labels': ["entailment", "not_entailment"]
        },
        'rte': {
            'url': 'https://bj.bcebos.com/dataset/glue/RTE.zip',
            'md5': 'bef554d0cafd4ab6743488101c638539',
            'splits': {
                'train': [
                    os.path.join('RTE', 'train.tsv'),
                    'd2844f558d111a16503144bb37a8165f', (1, 2, 3), 1
                ],
                'dev': [
                    os.path.join('RTE', 'dev.tsv'),
                    '973cb4178d4534cf745a01c309d4a66c', (1, 2, 3), 1
                ],
                'test': [
                    os.path.join('RTE', 'test.tsv'),
                    '6041008f3f3e48704f57ce1b88ad2e74', (1, 2), 1
                ]
            },
            'labels': ["entailment", "not_entailment"]
        },
        'wnli': {
            'url': 'https://bj.bcebos.com/dataset/glue/WNLI.zip',
            'md5': 'a1b4bd2861017d302d29e42139657a42',
            'splits': {
                'train': [
                    os.path.join('WNLI', 'train.tsv'),
                    '5cdc5a87b7be0c87a6363fa6a5481fc1', (1, 2, 3), 1
                ],
                'dev': [
                    os.path.join('WNLI', 'dev.tsv'),
                    'a79a6dd5d71287bcad6824c892e517ee', (1, 2, 3), 1
                ],
                'test': [
                    os.path.join('WNLI', 'test.tsv'),
                    'a18789ba4f60f6fdc8cb4237e4ba24b5', (1, 2), 1
                ]
            },
            'labels': ["0", "1"]
        },
        'mrpc': {
            'url': {
                'train_data':
                'https://bj.bcebos.com/dataset/glue/mrpc/msr_paraphrase_train.txt',
                'dev_id': 'https://bj.bcebos.com/dataset/glue/mrpc/dev_ids.tsv',
                'test_data':
                'https://bj.bcebos.com/dataset/glue/mrpc/msr_paraphrase_test.txt'
            },
            'md5': {
                'train_data': '793daf7b6224281e75fe61c1f80afe35',
                'dev_id': '7ab59a1b04bd7cb773f98a0717106c9b',
                'test_data': 'e437fdddb92535b820fe8852e2df8a49'
            },
            'splits': {
                'train': [
                    os.path.join('MRPC', 'train.tsv'),
                    'dc2dac669a113866a6480a0b10cd50bf', (3, 4, 0), 1
                ],
                'dev': [
                    os.path.join('MRPC', 'dev.tsv'),
                    '185958e46ba556b38c6a7cc63f3a2135', (3, 4, 0), 1
                ],
                'test': [
                    os.path.join('MRPC', 'test.tsv'),
                    '4825dab4b4832f81455719660b608de5', (3, 4), 1
                ]
            },
            'labels': ["0", "1"]
        }
    }

    def _get_data(self, mode, **kwargs):
        builder_config = self.BUILDER_CONFIGS[self.name]
        if self.name != 'mrpc':
            default_root = os.path.join(DATA_HOME, self.__class__.__name__)
            filename, data_hash, _, _ = builder_config['splits'][mode]
            fullname = os.path.join(default_root, filename)
            if not os.path.exists(fullname) or (
                    data_hash and not md5file(fullname) == data_hash):
                get_path_from_url(builder_config['url'], default_root,
                                  builder_config['md5'])

        else:
            default_root = os.path.join(DATA_HOME, self.__class__.__name__)
            filename, data_hash, _, _ = builder_config['splits'][mode]
            fullname = os.path.join(default_root, filename)
            if not os.path.exists(fullname) or (
                    data_hash and not md5file(fullname) == data_hash):
                if mode in ('train', 'dev'):
                    dev_id_path = get_path_from_url(
                        builder_config['url']['dev_id'],
                        os.path.join(default_root, 'MRPC'),
                        builder_config['md5']['dev_id'])
                    train_data_path = get_path_from_url(
                        builder_config['url']['train_data'],
                        os.path.join(default_root, 'MRPC'),
                        builder_config['md5']['train_data'])
                    # read dev data ids
                    dev_ids = []
                    print(dev_id_path)
                    with open(dev_id_path, encoding='utf-8') as ids_fh:
                        for row in ids_fh:
                            dev_ids.append(row.strip().split('\t'))

                    # generate train and dev set
                    train_path = os.path.join(default_root, 'MRPC', 'train.tsv')
                    dev_path = os.path.join(default_root, 'MRPC', 'dev.tsv')
                    with open(train_data_path, encoding='utf-8') as data_fh:
                        with open(
                                train_path, 'w', encoding='utf-8') as train_fh:
                            with open(dev_path, 'w', encoding='utf8') as dev_fh:
                                header = data_fh.readline()
                                train_fh.write(header)
                                dev_fh.write(header)
                                for row in data_fh:
                                    label, id1, id2, s1, s2 = row.strip().split(
                                        '\t')
                                    example = '%s\t%s\t%s\t%s\t%s\n' % (
                                        label, id1, id2, s1, s2)
                                    if [id1, id2] in dev_ids:
                                        dev_fh.write(example)
                                    else:
                                        train_fh.write(example)

                else:
                    test_data_path = get_path_from_url(
                        builder_config['url']['test_data'],
                        os.path.join(default_root, 'MRPC'),
                        builder_config['md5']['test_data'])
                    test_path = os.path.join(default_root, 'MRPC', 'test.tsv')
                    with open(test_data_path, encoding='utf-8') as data_fh:
                        with open(test_path, 'w', encoding='utf-8') as test_fh:
                            header = data_fh.readline()
                            test_fh.write(
                                'index\t#1 ID\t#2 ID\t#1 String\t#2 String\n')
                            for idx, row in enumerate(data_fh):
                                label, id1, id2, s1, s2 = row.strip().split(
                                    '\t')
                                test_fh.write('%d\t%s\t%s\t%s\t%s\n' %
                                              (idx, id1, id2, s1, s2))

        return fullname

    def _read(self, filename, split):
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
                if self.name in ['cola', 'sst-2']:
                    yield {
                        'sentence': example[0]
                    } if 'test' in split else {
                        'sentence': example[0],
                        'labels': example[-1]
                    }
                else:
                    yield {
                        'sentence1': example[0],
                        'sentence2': example[1]
                    } if 'test' in split else {
                        'sentence1': example[0],
                        'sentence2': example[1],
                        'labels': example[-1]
                    }

    def get_labels(self):
        """
        Returns labels of the Glue task.
        """
        return self.BUILDER_CONFIGS[self.name]['labels']
