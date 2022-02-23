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
import csv
from contextlib import ExitStack
import shutil

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url, _decompress, _get_unique_endpoints
from paddle.distributed import ParallelEnv
from paddlenlp.utils.env import DATA_HOME
from paddlenlp.utils.log import logger
from . import DatasetBuilder

__all__ = ['XNLI']
ALL_LANGUAGES = ("ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw",
                 "th", "tr", "ur", "vi", "zh")


class XNLI(DatasetBuilder):
    """
    XNLI is a subset of a few thousand examples from MNLI which has been translated into
    a 14 different languages (some low-ish resource). As with MNLI, the goal is to predict
    textual entailment (does sentence A imply/contradict/neither sentence B) and is a
    classification task (given two sentences, predict one of three labels).

    For more information, please visit https://github.com/facebookresearch/XNLI
    """
    META_INFO = collections.namedtuple('META_INFO', ('file', 'data_md5', 'url',
                                                     'zipfile_md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join('XNLI-MT-1.0', 'XNLI-MT-1.0', 'multinli'), '',
            'https://bj.bcebos.com/paddlenlp/datasets/XNLI-MT-1.0.zip',
            'fa3d8d6c3d1866cedc45680ba93c296e'),
        'dev': META_INFO(
            os.path.join('XNLI-1.0', 'XNLI-1.0', 'xnli.dev.tsv'),
            '4c23601abba3e3e222e19d1c6851649e',
            'https://bj.bcebos.com/paddlenlp/datasets/XNLI-1.0.zip',
            '53393158739ec671c34f205efc7d1666'),
        'test': META_INFO(
            os.path.join('XNLI-1.0', 'XNLI-1.0', 'xnli.test.tsv'),
            'fbc26e90f7e892e24dde978a2bd8ece6',
            'https://bj.bcebos.com/paddlenlp/datasets/XNLI-1.0.zip',
            '53393158739ec671c34f205efc7d1666'),
    }

    def _get_data(self, mode, **kwargs):
        """Downloads dataset."""
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash, url, zipfile_hash = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)
        if mode == 'train':
            if not os.path.exists(fullname):
                get_path_from_url(url, default_root, zipfile_hash)
            unique_endpoints = _get_unique_endpoints(ParallelEnv()
                                                     .trainer_endpoints[:])
            if ParallelEnv().current_endpoint in unique_endpoints:
                file_num = len(os.listdir(fullname))
                if file_num != len(ALL_LANGUAGES):
                    logger.warning(
                        "Number of train files is %d != %d, decompress again." %
                        (file_num, len(ALL_LANGUAGES)))
                    shutil.rmtree(fullname)
                    _decompress(
                        os.path.join(default_root, os.path.basename(url)))
        else:
            if not os.path.exists(fullname) or (
                    data_hash and not md5file(fullname) == data_hash):
                get_path_from_url(url, default_root, zipfile_hash)

        return fullname

    def _read(self, filename, split):
        """Reads data."""
        language = self.name
        if language == "all_languages":
            languages = ALL_LANGUAGES
        else:
            languages = [language]
        if split == 'train':
            files = [
                os.path.join(filename, f"multinli.train.{lang}.tsv")
                for lang in languages
            ]
            if language == "all_languages":
                with ExitStack() as stack:
                    files = [
                        stack.enter_context(open(
                            file, 'r', encoding="utf-8")) for file in files
                    ]
                    readers = [
                        csv.DictReader(
                            file, delimiter="\t", quoting=csv.QUOTE_NONE)
                        for file in files
                    ]
                    for row_idx, rows in enumerate(zip(*readers)):
                        if not rows[0]["label"]:
                            continue
                        data = {
                            "premise": {},
                            "hypothesis": {},
                            "label": rows[0]["label"].replace("contradictory",
                                                              "contradiction")
                        }
                        for lang, row in zip(languages, rows):
                            if not row["premise"] or not row["hypo"]:
                                continue
                            data["premise"][lang] = row["premise"]
                            data["hypothesis"][lang] = row["hypo"]
                        yield data
            else:
                for idx, file in enumerate(files):
                    f = open(file, 'r', encoding="utf-8")
                    reader = csv.DictReader(
                        f, delimiter="\t", quoting=csv.QUOTE_NONE)
                    for row_idx, row in enumerate(reader):
                        if not row["premise"] or not row["hypo"] or not row[
                                "label"]:
                            continue
                        yield {
                            "premise": row["premise"],
                            "hypothesis": row["hypo"],
                            "label": row["label"].replace("contradictory",
                                                          "contradiction"),
                        }
        else:
            if language == "all_languages":
                rows_per_pair_id = collections.defaultdict(list)
                with open(filename, encoding="utf-8") as f:
                    reader = csv.DictReader(
                        f, delimiter="\t", quoting=csv.QUOTE_NONE)
                    for row in reader:
                        rows_per_pair_id[row["pairID"]].append(row)

                for rows in rows_per_pair_id.values():
                    if not rows[0]["gold_label"]:
                        continue
                    data = {
                        "premise": {},
                        "hypothesis": {},
                        "label": rows[0]["gold_label"]
                    }
                    for row in rows:
                        if not row["sentence1"] or not row["sentence2"]:
                            continue
                        data["premise"][row["language"]] = row["sentence1"]
                        data["hypothesis"][row["language"]] = row["sentence2"]
                    yield data
            else:
                with open(filename, encoding="utf-8") as f:
                    reader = csv.DictReader(
                        f, delimiter="\t", quoting=csv.QUOTE_NONE)
                    for row in reader:
                        if row["language"] == language:
                            if not row["sentence1"] or not row[
                                    "sentence2"] or not row["gold_label"]:
                                continue
                            yield {
                                "premise": row["sentence1"],
                                "hypothesis": row["sentence2"],
                                "label": row["gold_label"],
                            }

    def get_labels(self):
        """
        Return labels of XNLI dataset.
        """
        return ["entailment", "neutral", "contradiction"]
