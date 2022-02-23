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
Conll05 dataset.
Paddle semantic role labeling Book and demo use this dataset as an example.
Because Conll05 is not free in public, the default downloaded URL is test set
of Conll05 (which is public). Users can change URL and MD5 to their Conll
dataset. And a pre-trained word vector model based on Wikipedia corpus is used
to initialize SRL model.
"""

from __future__ import print_function

import tarfile
import gzip
import itertools
import paddle.dataset.common
import paddle.compat as cpt
import paddle.utils.deprecated as deprecated
from six.moves import zip, range

__all__ = []

DATA_URL = 'http://paddlemodels.bj.bcebos.com/conll05st/conll05st-tests.tar.gz'
DATA_MD5 = '387719152ae52d60422c016e92a742fc'
WORDDICT_URL = 'http://paddlemodels.bj.bcebos.com/conll05st%2FwordDict.txt'
WORDDICT_MD5 = 'ea7fb7d4c75cc6254716f0177a506baa'
VERBDICT_URL = 'http://paddlemodels.bj.bcebos.com/conll05st%2FverbDict.txt'
VERBDICT_MD5 = '0d2977293bbb6cbefab5b0f97db1e77c'
TRGDICT_URL = 'http://paddlemodels.bj.bcebos.com/conll05st%2FtargetDict.txt'
TRGDICT_MD5 = 'd8c7f03ceb5fc2e5a0fa7503a4353751'
EMB_URL = 'http://paddlemodels.bj.bcebos.com/conll05st%2Femb'
EMB_MD5 = 'bf436eb0faa1f6f9103017f8be57cdb7'

UNK_IDX = 0


def load_label_dict(filename):
    d = dict()
    tag_dict = set()
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line.startswith("B-"):
                tag_dict.add(line[2:])
            elif line.startswith("I-"):
                tag_dict.add(line[2:])
        index = 0
        for tag in tag_dict:
            d["B-" + tag] = index
            index += 1
            d["I-" + tag] = index
            index += 1
        d["O"] = index
    return d


def load_dict(filename):
    d = dict()
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            d[line.strip()] = i
    return d


def corpus_reader(data_path, words_name, props_name):
    """
    Read one corpus. It returns an iterator. Each element of
    this iterator is a tuple including sentence and labels. The sentence is
    consist of a list of word IDs. The labels include a list of label IDs.
    :return: a iterator of data.
    :rtype: iterator
    """

    def reader():
        tf = tarfile.open(data_path)
        wf = tf.extractfile(words_name)
        pf = tf.extractfile(props_name)
        with gzip.GzipFile(fileobj=wf) as words_file, gzip.GzipFile(
                fileobj=pf) as props_file:
            sentences = []
            labels = []
            one_seg = []
            for word, label in zip(words_file, props_file):
                word = cpt.to_text(word.strip())
                label = cpt.to_text(label.strip().split())

                if len(label) == 0:  # end of sentence
                    for i in range(len(one_seg[0])):
                        a_kind_lable = [x[i] for x in one_seg]
                        labels.append(a_kind_lable)

                    if len(labels) >= 1:
                        verb_list = []
                        for x in labels[0]:
                            if x != '-':
                                verb_list.append(x)

                        for i, lbl in enumerate(labels[1:]):
                            cur_tag = 'O'
                            is_in_bracket = False
                            lbl_seq = []
                            verb_word = ''
                            for l in lbl:
                                if l == '*' and is_in_bracket == False:
                                    lbl_seq.append('O')
                                elif l == '*' and is_in_bracket == True:
                                    lbl_seq.append('I-' + cur_tag)
                                elif l == '*)':
                                    lbl_seq.append('I-' + cur_tag)
                                    is_in_bracket = False
                                elif l.find('(') != -1 and l.find(')') != -1:
                                    cur_tag = l[1:l.find('*')]
                                    lbl_seq.append('B-' + cur_tag)
                                    is_in_bracket = False
                                elif l.find('(') != -1 and l.find(')') == -1:
                                    cur_tag = l[1:l.find('*')]
                                    lbl_seq.append('B-' + cur_tag)
                                    is_in_bracket = True
                                else:
                                    raise RuntimeError('Unexpected label: %s' %
                                                       l)

                            yield sentences, verb_list[i], lbl_seq

                    sentences = []
                    labels = []
                    one_seg = []
                else:
                    sentences.append(word)
                    one_seg.append(label)

        pf.close()
        wf.close()
        tf.close()

    return reader


def reader_creator(corpus_reader,
                   word_dict=None,
                   predicate_dict=None,
                   label_dict=None):
    def reader():
        for sentence, predicate, labels in corpus_reader():

            sen_len = len(sentence)

            verb_index = labels.index('B-V')
            mark = [0] * len(labels)
            if verb_index > 0:
                mark[verb_index - 1] = 1
                ctx_n1 = sentence[verb_index - 1]
            else:
                ctx_n1 = 'bos'

            if verb_index > 1:
                mark[verb_index - 2] = 1
                ctx_n2 = sentence[verb_index - 2]
            else:
                ctx_n2 = 'bos'

            mark[verb_index] = 1
            ctx_0 = sentence[verb_index]

            if verb_index < len(labels) - 1:
                mark[verb_index + 1] = 1
                ctx_p1 = sentence[verb_index + 1]
            else:
                ctx_p1 = 'eos'

            if verb_index < len(labels) - 2:
                mark[verb_index + 2] = 1
                ctx_p2 = sentence[verb_index + 2]
            else:
                ctx_p2 = 'eos'

            word_idx = [word_dict.get(w, UNK_IDX) for w in sentence]

            ctx_n2_idx = [word_dict.get(ctx_n2, UNK_IDX)] * sen_len
            ctx_n1_idx = [word_dict.get(ctx_n1, UNK_IDX)] * sen_len
            ctx_0_idx = [word_dict.get(ctx_0, UNK_IDX)] * sen_len
            ctx_p1_idx = [word_dict.get(ctx_p1, UNK_IDX)] * sen_len
            ctx_p2_idx = [word_dict.get(ctx_p2, UNK_IDX)] * sen_len

            pred_idx = [predicate_dict.get(predicate)] * sen_len
            label_idx = [label_dict.get(w) for w in labels]

            yield word_idx, ctx_n2_idx, ctx_n1_idx, \
              ctx_0_idx, ctx_p1_idx, ctx_p2_idx, pred_idx, mark, label_idx

    return reader


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.Conll05st",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def get_dict():
    """
    Get the word, verb and label dictionary of Wikipedia corpus.
    """
    word_dict = load_dict(
        paddle.dataset.common.download(WORDDICT_URL, 'conll05st', WORDDICT_MD5))
    verb_dict = load_dict(
        paddle.dataset.common.download(VERBDICT_URL, 'conll05st', VERBDICT_MD5))
    label_dict = load_label_dict(
        paddle.dataset.common.download(TRGDICT_URL, 'conll05st', TRGDICT_MD5))
    return word_dict, verb_dict, label_dict


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.Conll05st",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def get_embedding():
    """
    Get the trained word vector based on Wikipedia corpus.
    """
    return paddle.dataset.common.download(EMB_URL, 'conll05st', EMB_MD5)


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.Conll05st",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def test():
    """
    Conll05 test set creator.

    Because the training dataset is not free, the test dataset is used for
    training. It returns a reader creator, each sample in the reader is nine
    features, including sentence sequence, predicate, predicate context,
    predicate context flag and tagged sequence.

    :return: Training reader creator
    :rtype: callable
    """
    word_dict, verb_dict, label_dict = get_dict()
    reader = corpus_reader(
        paddle.dataset.common.download(DATA_URL, 'conll05st', DATA_MD5),
        words_name='conll05st-release/test.wsj/words/test.wsj.words.gz',
        props_name='conll05st-release/test.wsj/props/test.wsj.props.gz')
    return reader_creator(reader, word_dict, verb_dict, label_dict)


@deprecated(
    since="2.0.0",
    update_to="paddle.text.datasets.Conll05st",
    level=1,
    reason="Please use new dataset API which supports paddle.io.DataLoader")
def fetch():
    paddle.dataset.common.download(WORDDICT_URL, 'conll05st', WORDDICT_MD5)
    paddle.dataset.common.download(VERBDICT_URL, 'conll05st', VERBDICT_MD5)
    paddle.dataset.common.download(TRGDICT_URL, 'conll05st', TRGDICT_MD5)
    paddle.dataset.common.download(EMB_URL, 'conll05st', EMB_MD5)
    paddle.dataset.common.download(DATA_URL, 'conll05st', DATA_MD5)
