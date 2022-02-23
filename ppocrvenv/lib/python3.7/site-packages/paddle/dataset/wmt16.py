# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
"""
ACL2016 Multimodal Machine Translation. Please see this website for more
details: http://www.statmt.org/wmt16/multimodal-task.html#task1

If you use the dataset created for your task, please cite the following paper:
Multi30K: Multilingual English-German Image Descriptions.

@article{elliott-EtAl:2016:VL16,
 author    = {{Elliott}, D. and {Frank}, S. and {Sima"an}, K. and {Specia}, L.},
 title     = {Multi30K: Multilingual English-German Image Descriptions},
 booktitle = {Proceedings of the 6th Workshop on Vision and Language},
 year      = {2016},
 pages     = {70--74},
 year      = 2016
}
"""

from __future__ import print_function

import os
import six
import tarfile
import gzip
from collections import defaultdict

import paddle
import paddle.compat as cpt
import paddle.utils.deprecated as deprecated

__all__ = []

DATA_URL = ("http://paddlemodels.bj.bcebos.com/wmt/wmt16.tar.gz")
DATA_MD5 = "0c38be43600334966403524a40dcd81e"

TOTAL_EN_WORDS = 11250
TOTAL_DE_WORDS = 19220

START_MARK = "<s>"
END_MARK = "<e>"
UNK_MARK = "<unk>"


def __build_dict(tar_file, dict_size, save_path, lang):
    word_dict = defaultdict(int)
    with tarfile.open(tar_file, mode="r") as f:
        for line in f.extractfile("wmt16/train"):
            line = cpt.to_text(line)
            line_split = line.strip().split("\t")
            if len(line_split) != 2: continue
            sen = line_split[0] if lang == "en" else line_split[1]
            for w in sen.split():
                word_dict[w] += 1

    with open(save_path, "wb") as fout:
        fout.write(
            cpt.to_bytes("%s\n%s\n%s\n" % (START_MARK, END_MARK, UNK_MARK)))
        for idx, word in enumerate(
                sorted(
                    six.iteritems(word_dict), key=lambda x: x[1],
                    reverse=True)):
            if idx + 3 == dict_size: break
            fout.write(cpt.to_bytes(word[0]))
            fout.write(cpt.to_bytes('\n'))


def __load_dict(tar_file, dict_size, lang, reverse=False):
    dict_path = os.path.join(paddle.dataset.common.DATA_HOME,
                             "wmt16/%s_%d.dict" % (lang, dict_size))
    if not os.path.exists(dict_path) or (
            len(open(dict_path, "rb").readlines()) != dict_size):
        __build_dict(tar_file, dict_size, dict_path, lang)

    word_dict = {}
    with open(dict_path, "rb") as fdict:
        for idx, line in enumerate(fdict):
            if reverse:
                word_dict[idx] = cpt.to_text(line.strip())
            else:
                word_dict[cpt.to_text(line.strip())] = idx
    return word_dict


def __get_dict_size(src_dict_size, trg_dict_size, src_lang):
    src_dict_size = min(src_dict_size, (TOTAL_EN_WORDS if src_lang == "en" else
                                        TOTAL_DE_WORDS))
    trg_dict_size = min(trg_dict_size, (TOTAL_DE_WORDS if src_lang == "en" else
                                        TOTAL_EN_WORDS))
    return src_dict_size, trg_dict_size


def reader_creator(tar_file, file_name, src_dict_size, trg_dict_size, src_lang):
    def reader():
        src_dict = __load_dict(tar_file, src_dict_size, src_lang)
        trg_dict = __load_dict(tar_file, trg_dict_size,
                               ("de" if src_lang == "en" else "en"))

        # the index for start mark, end mark, and unk are the same in source
        # language and target language. Here uses the source language
        # dictionary to determine their indices.
        start_id = src_dict[START_MARK]
        end_id = src_dict[END_MARK]
        unk_id = src_dict[UNK_MARK]

        src_col = 0 if src_lang == "en" else 1
        trg_col = 1 - src_col

        with tarfile.open(tar_file, mode="r") as f:
            for line in f.extractfile(file_name):
                line = cpt.to_text(line)
                line_split = line.strip().split("\t")
                if len(line_split) != 2:
                    continue
                src_words = line_split[src_col].split()
                src_ids = [start_id] + [
                    src_dict.get(w, unk_id) for w in src_words
                ] + [end_id]

                trg_words = line_split[trg_col].split()
                trg_ids = [trg_dict.get(w, unk_id) for w in trg_words]

                trg_ids_next = trg_ids + [end_id]
                trg_ids = [start_id] + trg_ids

                yield src_ids, trg_ids, trg_ids_next

    return reader


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.WMT16",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def train(src_dict_size, trg_dict_size, src_lang="en"):
    """
    WMT16 train set reader.

    This function returns the reader for train data. Each sample the reader
    returns is made up of three fields: the source language word index sequence,
    target language word index sequence and next word index sequence.


    NOTE:
    The original like for training data is:
    http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz

    paddle.dataset.wmt16 provides a tokenized version of the original dataset by
    using moses's tokenization script:
    https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl

    Args:
        src_dict_size(int): Size of the source language dictionary. Three
                            special tokens will be added into the dictionary:
                            <s> for start mark, <e> for end mark, and <unk> for
                            unknown word.
        trg_dict_size(int): Size of the target language dictionary. Three
                            special tokens will be added into the dictionary:
                            <s> for start mark, <e> for end mark, and <unk> for
                            unknown word.
        src_lang(string): A string indicating which language is the source
                          language. Available options are: "en" for English
                          and "de" for Germany.

    Returns:
        callable: The train reader.
    """

    if src_lang not in ["en", "de"]:
        raise ValueError("An error language type.  Only support: "
                         "en (for English); de(for Germany).")
    src_dict_size, trg_dict_size = __get_dict_size(src_dict_size, trg_dict_size,
                                                   src_lang)

    return reader_creator(
        tar_file=paddle.dataset.common.download(DATA_URL, "wmt16", DATA_MD5,
                                                "wmt16.tar.gz"),
        file_name="wmt16/train",
        src_dict_size=src_dict_size,
        trg_dict_size=trg_dict_size,
        src_lang=src_lang)


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.WMT16",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def test(src_dict_size, trg_dict_size, src_lang="en"):
    """
    WMT16 test set reader.

    This function returns the reader for test data. Each sample the reader
    returns is made up of three fields: the source language word index sequence,
    target language word index sequence and next word index sequence.

    NOTE:
    The original like for test data is:
    http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz

    paddle.dataset.wmt16 provides a tokenized version of the original dataset by
    using moses's tokenization script:
    https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl

    Args:
        src_dict_size(int): Size of the source language dictionary. Three
                            special tokens will be added into the dictionary:
                            <s> for start mark, <e> for end mark, and <unk> for
                            unknown word.
        trg_dict_size(int): Size of the target language dictionary. Three
                            special tokens will be added into the dictionary:
                            <s> for start mark, <e> for end mark, and <unk> for
                            unknown word.
        src_lang(string): A string indicating which language is the source
                          language. Available options are: "en" for English
                          and "de" for Germany.

    Returns:
        callable: The test reader.
    """

    if src_lang not in ["en", "de"]:
        raise ValueError("An error language type. "
                         "Only support: en (for English); de(for Germany).")

    src_dict_size, trg_dict_size = __get_dict_size(src_dict_size, trg_dict_size,
                                                   src_lang)

    return reader_creator(
        tar_file=paddle.dataset.common.download(DATA_URL, "wmt16", DATA_MD5,
                                                "wmt16.tar.gz"),
        file_name="wmt16/test",
        src_dict_size=src_dict_size,
        trg_dict_size=trg_dict_size,
        src_lang=src_lang)


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.WMT16",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def validation(src_dict_size, trg_dict_size, src_lang="en"):
    """
    WMT16 validation set reader.

    This function returns the reader for validation data. Each sample the reader
    returns is made up of three fields: the source language word index sequence,
    target language word index sequence and next word index sequence.

    NOTE:
    The original like for validation data is:
    http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz

    paddle.dataset.wmt16 provides a tokenized version of the original dataset by
    using moses's tokenization script:
    https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl

    Args:
        src_dict_size(int): Size of the source language dictionary. Three
                            special tokens will be added into the dictionary:
                            <s> for start mark, <e> for end mark, and <unk> for
                            unknown word.
        trg_dict_size(int): Size of the target language dictionary. Three
                            special tokens will be added into the dictionary:
                            <s> for start mark, <e> for end mark, and <unk> for
                            unknown word.
        src_lang(string): A string indicating which language is the source
                          language. Available options are: "en" for English
                          and "de" for Germany.

    Returns:
        callable: The validation reader.
    """
    if src_lang not in ["en", "de"]:
        raise ValueError("An error language type. "
                         "Only support: en (for English); de(for Germany).")
    src_dict_size, trg_dict_size = __get_dict_size(src_dict_size, trg_dict_size,
                                                   src_lang)

    return reader_creator(
        tar_file=paddle.dataset.common.download(DATA_URL, "wmt16", DATA_MD5,
                                                "wmt16.tar.gz"),
        file_name="wmt16/val",
        src_dict_size=src_dict_size,
        trg_dict_size=trg_dict_size,
        src_lang=src_lang)


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.WMT16",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def get_dict(lang, dict_size, reverse=False):
    """
    return the word dictionary for the specified language.

    Args:
        lang(string): A string indicating which language is the source
                      language. Available options are: "en" for English
                      and "de" for Germany.
        dict_size(int): Size of the specified language dictionary.
        reverse(bool): If reverse is set to False, the returned python
                       dictionary will use word as key and use index as value.
                       If reverse is set to True, the returned python
                       dictionary will use index as key and word as value.

    Returns:
        dict: The word dictionary for the specific language.
    """

    if lang == "en": dict_size = min(dict_size, TOTAL_EN_WORDS)
    else: dict_size = min(dict_size, TOTAL_DE_WORDS)

    dict_path = os.path.join(paddle.dataset.common.DATA_HOME,
                             "wmt16/%s_%d.dict" % (lang, dict_size))
    assert os.path.exists(dict_path), "Word dictionary does not exist. "
    "Please invoke paddle.dataset.wmt16.train/test/validation first "
    "to build the dictionary."
    tar_file = os.path.join(paddle.dataset.common.DATA_HOME, "wmt16.tar.gz")
    return __load_dict(tar_file, dict_size, lang, reverse)


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.WMT16",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def fetch():
    """download the entire dataset.
    """
    paddle.v4.dataset.common.download(DATA_URL, "wmt16", DATA_MD5,
                                      "wmt16.tar.gz")
