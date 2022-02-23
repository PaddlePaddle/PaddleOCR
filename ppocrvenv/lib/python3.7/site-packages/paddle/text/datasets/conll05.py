#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import gzip
import tarfile
import numpy as np
import six
from six.moves import cPickle as pickle

from paddle.io import Dataset
import paddle.compat as cpt
from paddle.dataset.common import _check_exists_and_download

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


class Conll05st(Dataset):
    """
    Implementation of `Conll05st <https://www.cs.upc.edu/~srlconll/soft.html>`_
    test dataset.

    Note: only support download test dataset automatically for that
          only test dataset of Conll05st is public.

    Args:
        data_file(str): path to data tar file, can be set None if
            :attr:`download` is True. Default None
        word_dict_file(str): path to word dictionary file, can be set None if
            :attr:`download` is True. Default None
        verb_dict_file(str): path to verb dictionary file, can be set None if
            :attr:`download` is True. Default None
        target_dict_file(str): path to target dictionary file, can be set None if
            :attr:`download` is True. Default None
        emb_file(str): path to embedding dictionary file, only used for
            :code:`get_embedding` can be set None if :attr:`download` is
            True. Default None
        download(bool): whether to download dataset automatically if
            :attr:`data_file` :attr:`word_dict_file` :attr:`verb_dict_file`
            :attr:`target_dict_file` is not set. Default True

    Returns:
        Dataset: instance of conll05st dataset

    Examples:

        .. code-block:: python

            import paddle
            from paddle.text.datasets import Conll05st

            class SimpleNet(paddle.nn.Layer):
                def __init__(self):
                    super(SimpleNet, self).__init__()

                def forward(self, pred_idx, mark, label):
                    return paddle.sum(pred_idx), paddle.sum(mark), paddle.sum(label)


            conll05st = Conll05st()

            for i in range(10):
                pred_idx, mark, label= conll05st[i][-3:]
                pred_idx = paddle.to_tensor(pred_idx)
                mark = paddle.to_tensor(mark)
                label = paddle.to_tensor(label)

                model = SimpleNet()
                pred_idx, mark, label= model(pred_idx, mark, label)
                print(pred_idx.numpy(), mark.numpy(), label.numpy())

    """

    def __init__(self,
                 data_file=None,
                 word_dict_file=None,
                 verb_dict_file=None,
                 target_dict_file=None,
                 emb_file=None,
                 download=True):
        self.data_file = data_file
        if self.data_file is None:
            assert download, "data_file is not set and downloading automatically is disabled"
            self.data_file = _check_exists_and_download(
                data_file, DATA_URL, DATA_MD5, 'conll05st', download)

        self.word_dict_file = word_dict_file
        if self.word_dict_file is None:
            assert download, "word_dict_file is not set and downloading automatically is disabled"
            self.word_dict_file = _check_exists_and_download(
                word_dict_file, WORDDICT_URL, WORDDICT_MD5, 'conll05st',
                download)

        self.verb_dict_file = verb_dict_file
        if self.verb_dict_file is None:
            assert download, "verb_dict_file is not set and downloading automatically is disabled"
            self.verb_dict_file = _check_exists_and_download(
                verb_dict_file, VERBDICT_URL, VERBDICT_MD5, 'conll05st',
                download)

        self.target_dict_file = target_dict_file
        if self.target_dict_file is None:
            assert download, "target_dict_file is not set and downloading automatically is disabled"
            self.target_dict_file = _check_exists_and_download(
                target_dict_file, TRGDICT_URL, TRGDICT_MD5, 'conll05st',
                download)

        self.emb_file = emb_file
        if self.emb_file is None:
            assert download, "emb_file is not set and downloading automatically is disabled"
            self.emb_file = _check_exists_and_download(
                emb_file, EMB_URL, EMB_MD5, 'conll05st', download)

        self.word_dict = self._load_dict(self.word_dict_file)
        self.predicate_dict = self._load_dict(self.verb_dict_file)
        self.label_dict = self._load_label_dict(self.target_dict_file)

        # read dataset into memory
        self._load_anno()

    def _load_label_dict(self, filename):
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

    def _load_dict(self, filename):
        d = dict()
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                d[line.strip()] = i
        return d

    def _load_anno(self):
        tf = tarfile.open(self.data_file)
        wf = tf.extractfile(
            "conll05st-release/test.wsj/words/test.wsj.words.gz")
        pf = tf.extractfile(
            "conll05st-release/test.wsj/props/test.wsj.props.gz")
        self.sentences = []
        self.predicates = []
        self.labels = []
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

                            self.sentences.append(sentences)
                            self.predicates.append(verb_list[i])
                            self.labels.append(lbl_seq)

                    sentences = []
                    labels = []
                    one_seg = []
                else:
                    sentences.append(word)
                    one_seg.append(label)

        pf.close()
        wf.close()
        tf.close()

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        predicate = self.predicates[idx]
        labels = self.labels[idx]

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

        word_idx = [self.word_dict.get(w, UNK_IDX) for w in sentence]

        ctx_n2_idx = [self.word_dict.get(ctx_n2, UNK_IDX)] * sen_len
        ctx_n1_idx = [self.word_dict.get(ctx_n1, UNK_IDX)] * sen_len
        ctx_0_idx = [self.word_dict.get(ctx_0, UNK_IDX)] * sen_len
        ctx_p1_idx = [self.word_dict.get(ctx_p1, UNK_IDX)] * sen_len
        ctx_p2_idx = [self.word_dict.get(ctx_p2, UNK_IDX)] * sen_len

        pred_idx = [self.predicate_dict.get(predicate)] * sen_len
        label_idx = [self.label_dict.get(w) for w in labels]

        return (np.array(word_idx), np.array(ctx_n2_idx), np.array(ctx_n1_idx),
                np.array(ctx_0_idx), np.array(ctx_p1_idx), np.array(ctx_p2_idx),
                np.array(pred_idx), np.array(mark), np.array(label_idx))

    def __len__(self):
        return len(self.sentences)

    def get_dict(self):
        """
        Get the word, verb and label dictionary of Wikipedia corpus.

        Examples:
    
            .. code-block:: python
    
            	from paddle.text.datasets import Conll05st

            	conll05st = Conll05st()
            	word_dict, predicate_dict, label_dict = conll05st.get_dict()
        """
        return self.word_dict, self.predicate_dict, self.label_dict

    def get_embedding(self):
        """
        Get the embedding dictionary file.

        Examples:
    
            .. code-block:: python
    
            	from paddle.text.datasets import Conll05st

            	conll05st = Conll05st()
            	emb_file = conll05st.get_embedding()
        """
        return self.emb_file
