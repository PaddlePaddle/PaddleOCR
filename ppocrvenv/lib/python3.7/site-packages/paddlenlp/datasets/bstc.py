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

import os

import json
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder


class BSTC(DatasetBuilder):
    """
    BSTC (Baidu Speech Translation Corpus), a large-scale Chinese-English
    speech translation dataset. This dataset is constructed based on a
    collection of licensed videos of talks or lectures, including about
    68 hours of Mandarin data, their manual transcripts and translations
    into English, as well as automated transcripts by an automatic speech
    recognition (ASR) model.
    Details: https://arxiv.org/pdf/2104.03575.pdf
    """
    lazy = False
    BUILDER_CONFIGS = {
        'transcription_translation': {
            'url':
            "https://bj.bcebos.com/paddlenlp/datasets/bstc_transcription_translation.tar.gz",
            'md5': '236800188e397c42a3251982aeee48ee',
            'splits': {
                'train':
                [os.path.join('bstc_transcription_translation', 'train')],
                'dev': [
                    os.path.join('bstc_transcription_translation', 'dev',
                                 'streaming_transcription'),
                    os.path.join('bstc_transcription_translation', 'dev',
                                 'ref_text')
                ]
            }
        },
        'asr': {
            'url': "https://bj.bcebos.com/paddlenlp/datasets/bstc_asr.tar.gz",
            'md5': '3a0cc5039f45e62e29485e27d3a5f5a7',
            'splits': {
                'train': [os.path.join('bstc_asr', 'train', 'asr_sentences')],
                'dev': [
                    os.path.join('bstc_asr', 'dev', 'streaming_asr'),
                    os.path.join('bstc_asr', 'dev', 'ref_text')
                ]
            }
        }
    }

    def _get_data(self, mode, **kwargs):
        ''' Check and download Dataset '''
        builder_config = self.BUILDER_CONFIGS[self.name]
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        source_file_dir = builder_config['splits'][mode][0]
        source_full_dir = os.path.join(default_root, source_file_dir)
        if not os.path.exists(source_full_dir):
            get_path_from_url(builder_config['url'], default_root,
                              builder_config['md5'])
        if mode == 'train':
            return source_full_dir
        elif mode == 'dev':
            target_file_dir = builder_config['splits'][mode][1]
            target_full_dir = os.path.join(default_root, target_file_dir)
            if not os.path.exists(target_full_dir):
                get_path_from_url(builder_config['url'], default_root,
                                  builder_config['md5'])
            return source_full_dir, target_full_dir

    def _read(self, data_dir, split):
        """Reads data."""
        if split == 'train':
            if self.name == 'transcription_translation':
                source_full_dir = data_dir
                filenames = [
                    f for f in os.listdir(source_full_dir)
                    if not f.startswith('.')
                ]
                filenames.sort(key=lambda x: int(x[:-5]))
                for filename in filenames:
                    with open(
                            os.path.join(source_full_dir, filename),
                            'r',
                            encoding='utf-8') as f:
                        for line in f.readlines():
                            line = line.strip()
                            if not line:
                                continue
                            yield json.loads(line)
            elif self.name == 'asr':
                source_full_dir = data_dir
                dir_list = [
                    f for f in os.listdir(source_full_dir)
                    if not f.startswith('.')
                ]
                dir_list.sort(key=lambda x: int(x))
                for dir_name in dir_list:
                    filenames = [
                        f
                        for f in os.listdir(
                            os.path.join(source_full_dir, dir_name))
                        if not f.startswith('.')
                    ]
                    filenames.sort(key=lambda x: int(x[x.find('-') + 1:-5]))
                    for filename in filenames:
                        with open(
                                os.path.join(source_full_dir, dir_name,
                                             filename),
                                'r',
                                encoding='utf-8') as f:
                            for line in f.readlines():
                                line = line.strip()
                                if not line:
                                    continue
                                yield json.loads(line)
            else:
                raise ValueError(
                    'Argument name should be one of [transcription_translation, asr].'
                )
        elif split == 'dev':
            source_full_dir, target_full_dir = data_dir
            source_filenames = [
                f for f in os.listdir(source_full_dir) if f.endswith('txt')
            ]
            target_filenames = [
                f for f in os.listdir(target_full_dir) if f.endswith('txt')
            ]
            assert len(source_filenames) == len(target_filenames)
            source_filenames.sort(
                key=lambda x: int(x[:-4]) if self.name == 'transcription_translation' else int(x[:-8])
            )
            target_filenames.sort(key=lambda x: int(x[:-4]))
            for src_file, tgt_file in zip(source_filenames, target_filenames):
                if self.name == 'transcription_translation':
                    src_list = []
                    with open(
                            os.path.join(source_full_dir, src_file),
                            'r',
                            encoding='utf-8') as src_f:
                        src_part = []
                        for src_line in src_f.readlines():
                            src_line = src_line.strip()
                            if not src_line:
                                continue
                            if len(src_part) != 0 and not src_line.startswith(
                                    src_part[-1]):
                                src_list.append(src_part)
                                src_part = [src_line]
                            else:
                                src_part.append(src_line)
                        if len(src_part) > 0:
                            src_list.append(src_part)
                elif self.name == 'asr':
                    src_list = []
                    with open(
                            os.path.join(source_full_dir, src_file),
                            'r',
                            encoding='utf-8') as src_f:
                        src_part = []
                        for src_line in src_f.readlines():
                            src_line = src_line.strip()
                            if not src_line:
                                continue
                            line = src_line.split(', ')
                            final = line[2].split(': ')[1] == 'final'
                            src_part.append(src_line)
                            if final:
                                src_list.append(src_part)
                                src_part = []
                else:
                    raise ValueError(
                        'Argument name should be one of [transcription_translation, asr].'
                    )
                tgt_list = []
                with open(
                        os.path.join(target_full_dir, tgt_file),
                        'r',
                        encoding='utf-8') as tgt_f:
                    lines = tgt_f.readlines()
                    for idx, tgt_line in enumerate(lines):
                        tgt_line = tgt_line.strip()
                        if not tgt_line:
                            continue
                        tgt_list.append(tgt_line)
                yield {'src': src_list, 'tgt': tgt_list}
