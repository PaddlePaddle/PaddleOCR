# coding:utf-8
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import copy
import os
import itertools

import numpy as np
import paddle
from ..data import Vocab, Pad
from .utils import download_file, dygraph_mode_guard
from .task import Task
from .models import BiAffineParser

usage = r"""
           from paddlenlp import Taskflow 

           ddp = Taskflow("dependency_parsing")
           ddp("三亚是一座美丽的城市")
           '''
           [{'word': ['三亚', '是', '一座', '美丽', '的', '城市'], 'head': [2, 0, 6, 6, 4, 2], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'MT', 'VOB']}]
           '''
           ddp(["三亚是一座美丽的城市", "他送了一本书"])
           '''
           [{'word': ['三亚', '是', '一座', '美丽', '的', '城市'], 'head': [2, 0, 6, 6, 4, 2], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'MT', 'VOB']}, {'word': ['他', '送', '了', '一本', '书'], 'head': [2, 0, 2, 5, 2], 'deprel': ['SBV', 'HED', 'MT', 'ATT', 'VOB']}]
           '''       

           ddp = Taskflow("dependency_parsing", prob=True, use_pos=True)
           ddp("三亚是一座美丽的城市")
           '''
           [{'word': ['三亚', '是', '一座', '美丽的城市'], 'head': [2, 0, 4, 2], 'deprel': ['SBV', 'HED', 'ATT', 'VOB'], 'postag': ['LOC', 'v', 'm', 'n'], 'prob': [1.0, 1.0, 1.0, 1.0]}]
           '''

           ddp = Taskflow("dependency_parsing", model="ddparser-ernie-1.0")
           ddp("三亚是一座美丽的城市")
           '''
           [{'word': ['三亚', '是', '一座', '美丽', '的', '城市'], 'head': [2, 0, 6, 6, 4, 2], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'MT', 'VOB']}]
           '''

           ddp = Taskflow("dependency_parsing", model="ddparser-ernie-gram-zh")
           ddp("三亚是一座美丽的城市")
           '''
           [{'word': ['三亚', '是', '一座', '美丽', '的', '城市'], 'head': [2, 0, 6, 6, 4, 2], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'MT', 'VOB']}]
           '''

           # 已分词输入
           ddp = Taskflow("dependency_parsing", segmented=True)
           ddp.from_segments([["三亚", "是", "一座", "美丽", "的", "城市"]])
           '''
           [{'word': ['三亚', '是', '一座', '美丽', '的', '城市'], 'head': [2, 0, 6, 6, 4, 2], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'MT', 'VOB']}]
           '''
           ddp.from_segments([['三亚', '是', '一座', '美丽', '的', '城市'], ['他', '送', '了', '一本', '书']])
           '''
           [{'word': ['三亚', '是', '一座', '美丽', '的', '城市'], 'head': [2, 0, 6, 6, 4, 2], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'MT', 'VOB']}, {'word': ['他', '送', '了', '一本', '书'], 'head': [2, 0, 2, 5, 2], 'deprel': ['SBV', 'HED', 'MT', 'ATT', 'VOB']}]
           '''   
         """


class DDParserTask(Task):
    """
    DDParser task to analyze the dependency relationship between words in a sentence 
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        tree(bool): Ensure the output conforms to the tree structure.
        prob(bool): Whether to return the probability of predicted heads.
        use_pos(bool): Whether to return the postag.
        batch_size(int): Numbers of examples a batch.
        return_visual(bool): If True, the result will contain the dependency visualization.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "word_vocab": "vocab.json",
        "rel_vocab": "rel_vocab.json",
    }
    resource_files_urls = {
        "ddparser": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/dependency_parsing/ddparser/model_state.pdparams",
                "f388c91e85b5b4d0db40157a4ee28c08"
            ],
            "word_vocab": [
                "https://bj.bcebos.com/paddlenlp/taskflow/dependency_parsing/ddparser/word_vocab.json",
                "594694033b149cbb724cac0975df07e4"
            ],
            "rel_vocab": [
                "https://bj.bcebos.com/paddlenlp/taskflow/dependency_parsing/ddparser/rel_vocab.json",
                "0decf1363278705f885184ff8681f4cd"
            ],
        },
        "ddparser-ernie-1.0": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/dependency_parsing/ddparser-ernie-1.0/model_state.pdparams",
                "78a4d5c2add642a88f6fdbee3574f617"
            ],
            "word_vocab": [
                "https://bj.bcebos.com/paddlenlp/taskflow/dependency_parsing/ddparser-ernie-1.0/word_vocab.json",
                "17ed37b5b7ebb8475d4bff1ff8dac4b7"
            ],
            "rel_vocab": [
                "https://bj.bcebos.com/paddlenlp/taskflow/dependency_parsing/ddparser-ernie-1.0/rel_vocab.json",
                "0decf1363278705f885184ff8681f4cd"
            ],
        },
        "ddparser-ernie-gram-zh": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/dependency_parsing/ddparser-ernie-gram-zh/model_state.pdparams",
                "9d0a49026feb97fac22c8eec3e88f5c3"
            ],
            "word_vocab": [
                "https://bj.bcebos.com/paddlenlp/taskflow/dependency_parsing/ddparser-ernie-gram-zh/word_vocab.json",
                "38120123d39876337975cc616901c8b9"
            ],
            "rel_vocab": [
                "https://bj.bcebos.com/paddlenlp/taskflow/dependency_parsing/ddparser-ernie-gram-zh/rel_vocab.json",
                "0decf1363278705f885184ff8681f4cd"
            ],
        },
        "font_file": {
            "font_file": [
                "https://bj.bcebos.com/paddlenlp/taskflow/dependency_parsing/SourceHanSansCN-Regular.ttf",
                "cecb7328bc0b9412b897fb3fc61edcdb"
            ]
        }
    }

    def __init__(self,
                 task,
                 model,
                 tree=True,
                 prob=False,
                 use_pos=False,
                 use_cuda=False,
                 batch_size=1,
                 return_visual=False,
                 **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._usage = usage
        self.model = model

        if self.model == "ddparser":
            self.encoding_model = "lstm-pe"
        elif self.model == "ddparser-ernie-1.0":
            self.encoding_model = "ernie-1.0"
        elif self.model == "ddparser-ernie-gram-zh":
            self.encoding_model = "ernie-gram-zh"
        else:
            raise ValueError("The encoding model should be one of \
                ddparser, ddparser-ernie-1.0 and ddoarser-ernie-gram-zh")
        self._check_task_files()
        self._construct_vocabs()
        self.font_file_path = download_file(
            self._task_path, "SourceHanSansCN-Regular.ttf",
            self.resource_files_urls["font_file"]["font_file"][0],
            self.resource_files_urls["font_file"]["font_file"][1])
        self.tree = tree
        self.prob = prob
        self.use_pos = use_pos
        self.batch_size = batch_size
        self.return_visual = return_visual

        try:
            from LAC import LAC
        except:
            raise ImportError(
                "Please install the dependencies first, pip install LAC --upgrade"
            )

        self.use_cuda = use_cuda
        self.lac = LAC(mode="lac" if self.use_pos else "seg",
                       use_cuda=self.use_cuda)
        self._get_inference_model()

    def _check_segmented_words(self, inputs):
        inputs = inputs[0]
        if not all([isinstance(i, list) and i and all(i) for i in inputs]):
            raise TypeError("Invalid input format.")
        return inputs

    def from_segments(self, segmented_words):
        segmented_words = self._check_segmented_words(segmented_words)
        inputs = {}
        inputs['words'] = segmented_words
        inputs = self._preprocess_words(inputs)
        outputs = self._run_model(inputs)
        results = self._postprocess(outputs)
        return results

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64"),
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64"),
        ]

    def _construct_vocabs(self):
        word_vocab_path = os.path.join(self._task_path, "word_vocab.json")
        rel_vocab_path = os.path.join(self._task_path, "rel_vocab.json")
        self.word_vocab = Vocab.from_json(word_vocab_path)
        self.rel_vocab = Vocab.from_json(rel_vocab_path)
        self.word_pad_index = self.word_vocab.to_indices("[PAD]")
        self.word_bos_index = self.word_vocab.to_indices("[CLS]")
        self.word_eos_index = self.word_vocab.to_indices("[SEP]")

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        model_instance = BiAffineParser(
            encoding_model=self.encoding_model,
            n_rels=len(self.rel_vocab),
            n_words=len(self.word_vocab),
            pad_index=self.word_pad_index,
            bos_index=self.word_bos_index,
            eos_index=self.word_eos_index, )
        model_path = os.path.join(self._task_path, "model_state.pdparams")
        # Load the model parameter for the predict
        state_dict = paddle.load(model_path)
        model_instance.set_dict(state_dict)
        model_instance.eval()
        self._model = model_instance

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        return None

    def _preprocess_words(self, inputs):
        examples = []
        for text in inputs['words']:
            example = {"FORM": text}
            example = convert_example(
                example, vocabs=[self.word_vocab, self.rel_vocab])
            examples.append(example)

        batches = [
            examples[idx:idx + self.batch_size]
            for idx in range(0, len(examples), self.batch_size)
        ]

        def batchify_fn(batch):
            raw_batch = [raw for raw in zip(*batch)]
            batch = [pad_sequence(data) for data in raw_batch]
            return batch

        batches = [flat_words(batchify_fn(batch)[0]) for batch in batches]

        inputs['data_loader'] = batches
        return inputs

    def _preprocess(self, inputs):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """

        # Get the config from the kwargs
        num_workers = self.kwargs[
            'num_workers'] if 'num_workers' in self.kwargs else 0
        lazy_load = self.kwargs[
            'lazy_load'] if 'lazy_load' in self.kwargs else False

        outputs = {}

        lac_results = []
        position = 0

        inputs = self._check_input_text(inputs)
        while position < len(inputs):
            lac_results += self.lac.run(inputs[position:position +
                                               self.batch_size])
            position += self.batch_size

        if not self.use_pos:
            outputs['words'] = lac_results
        else:
            outputs['words'], outputs[
                'postags'] = [raw for raw in zip(*lac_results)]

        outputs = self._preprocess_words(outputs)
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function. 
        """

        arcs, rels, probs = [], [], []
        for batch in inputs['data_loader']:
            words, wp = batch
            self.input_handles[0].copy_from_cpu(words)
            self.input_handles[1].copy_from_cpu(wp)
            self.predictor.run()
            arc_preds = self.output_handle[0].copy_to_cpu()
            rel_preds = self.output_handle[1].copy_to_cpu()
            s_arc = self.output_handle[2].copy_to_cpu()
            mask = self.output_handle[3].copy_to_cpu().astype('bool')

            arc_preds, rel_preds = decode(arc_preds, rel_preds, s_arc, mask,
                                          self.tree)

            arcs.extend([arc_pred[m] for arc_pred, m in zip(arc_preds, mask)])
            rels.extend([rel_pred[m] for rel_pred, m in zip(rel_preds, mask)])
            if self.prob:
                arc_probs = probability(s_arc, arc_preds)
                probs.extend(
                    [arc_prob[m] for arc_prob, m in zip(arc_probs, mask)])
        inputs['arcs'] = arcs
        inputs['rels'] = rels
        inputs['probs'] = probs
        return inputs

    def _postprocess(self, inputs):

        arcs = inputs['arcs']
        rels = inputs['rels']
        words = inputs['words']
        arcs = [[s for s in seq] for seq in arcs]
        rels = [self.rel_vocab.to_tokens(seq) for seq in rels]

        results = []

        for word, arc, rel in zip(words, arcs, rels):
            result = {
                'word': word,
                'head': arc,
                'deprel': rel,
            }
            results.append(result)

        if self.use_pos:
            postags = inputs['postags']
            for result, postag in zip(results, postags):
                result['postag'] = postag

        if self.prob:
            probs = inputs['probs']
            probs = [[round(p, 2) for p in seq.tolist()] for seq in probs]
            for result, prob in zip(results, probs):
                result['prob'] = prob

        if self.return_visual:
            for result in results:
                result['visual'] = self._visualize(result)

        return results

    def _visualize(self, data):
        """
        Visualize the dependency.
        Args:
            data(dict): A dict contains the word, head and dep
         Returns:
            data: a numpy array, use cv2.imshow to show it or cv2.imwrite to save it.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as font_manager
        except:
            raise ImportError(
                "Please install the dependencies first, pip install matplotlib --upgrade"
            )

        self.plt = plt
        self.font = font_manager.FontProperties(fname=self.font_file_path)
        word, head, deprel = data['word'], data['head'], data['deprel']

        nodes = ['ROOT'] + word
        x = list(range(len(nodes)))
        y = [0] * (len(nodes))
        fig, ax = self.plt.subplots()
        # Control the picture size
        max_span = max([abs(i + 1 - j) for i, j in enumerate(head)])
        fig.set_size_inches((len(nodes), max_span / 2))
        # Set the points
        self.plt.scatter(x, y, c='w')

        for i in range(len(nodes)):
            txt = nodes[i]
            xytext = (i, 0)
            if i == 0:
                # Set 'ROOT'
                ax.annotate(
                    txt,
                    xy=xytext,
                    xycoords='data',
                    xytext=xytext,
                    textcoords='data', )
            else:
                xy = (head[i - 1], 0)
                rad = 0.5 if head[i - 1] < i else -0.5
                # Set the word
                ax.annotate(
                    txt,
                    xy=xy,
                    xycoords='data',
                    xytext=(xytext[0] - 0.1, xytext[1]),
                    textcoords='data',
                    fontproperties=self.font)
                # Draw the curve
                ax.annotate(
                    "",
                    xy=xy,
                    xycoords='data',
                    xytext=xytext,
                    textcoords='data',
                    arrowprops=dict(
                        arrowstyle="<-",
                        shrinkA=12,
                        shrinkB=12,
                        color='blue',
                        connectionstyle="arc3,rad=%s" % rad, ), )
                # Set the deprel label. Calculate its position by the radius
                text_x = min(i, head[i - 1]) + abs((i - head[i - 1])) / 2 - 0.2
                text_y = abs((i - head[i - 1])) / 4
                ax.annotate(
                    deprel[i - 1],
                    xy=xy,
                    xycoords='data',
                    xytext=[text_x, text_y],
                    textcoords='data')

        # Control the axis
        self.plt.axis('equal')
        self.plt.axis('off')

        # Save to numpy array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (
            3, ))[:, :, ::-1]
        return data


def pad_sequence(sequences, padding_value=0, fix_len=None):
    """Fill sequences(np.ndarray) into a fixed-length matrix."""
    max_size = sequences[0].shape
    trailing_dims = max_size[1:]
    max_len = max([s.shape[0] for s in sequences])
    if fix_len is not None:
        assert fix_len >= max_len, "fix_len is too small."
        max_len = fix_len
    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = np.full(out_dims, padding_value, dtype=sequences[0].dtype)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        out_tensor[i, :length, ...] = tensor
    return out_tensor


def convert_example(example, vocabs, fix_len=20):
    word_vocab, rel_vocab = vocabs

    word_bos_index = word_vocab.to_indices("[CLS]")
    word_eos_index = word_vocab.to_indices("[SEP]")

    words = [[word_vocab.to_indices(char) for char in word]
             for word in example["FORM"]]
    words = [[word_bos_index]] + words + [[word_eos_index]]
    return [
        pad_sequence(
            [np.array(
                ids[:fix_len], dtype=np.int64) for ids in words],
            fix_len=fix_len)
    ]


def flat_words(words, pad_index=0):
    mask = words != pad_index
    lens = np.sum(mask.astype(np.int64), axis=-1)
    position = np.cumsum(lens + (lens == 0).astype(np.int64), axis=1) - 1
    lens = np.sum(lens, -1)
    words = words.ravel()[np.flatnonzero(words)]

    sequences = []
    idx = 0
    for l in lens:
        sequences.append(words[idx:idx + l])
        idx += l
    words = Pad(pad_val=pad_index)(sequences)

    max_len = words.shape[1]

    mask = (position >= max_len).astype(np.int64)
    position = position * np.logical_not(mask) + mask * (max_len - 1)
    return words, position


def probability(s_arc, arc_preds):
    s_arc = s_arc - s_arc.max(axis=-1).reshape(list(s_arc.shape)[:-1] + [1])
    s_arc = np.exp(s_arc) / np.exp(s_arc).sum(
        axis=-1).reshape(list(s_arc.shape)[:-1] + [1])

    arc_probs = [
        s[np.arange(len(arc_pred)), arc_pred]
        for s, arc_pred in zip(s_arc, arc_preds)
    ]
    return arc_probs


def decode(arc_preds, rel_preds, s_arc, mask, tree):
    """decode"""
    lens = np.sum(mask, -1)

    bad = [not istree(seq[:i + 1]) for i, seq in zip(lens, arc_preds)]
    if tree and any(bad):
        arc_preds[bad] = eisner(s_arc[bad], mask[bad])
    rel_preds = [
        rel_pred[np.arange(len(arc_pred)), arc_pred]
        for arc_pred, rel_pred in zip(arc_preds, rel_preds)
    ]
    return arc_preds, rel_preds


def eisner(scores, mask):
    """
    Eisner algorithm is a general dynamic programming decoding algorithm for bilexical grammar.

    Args：
        scores: Adjacency matrix，shape=(batch, seq_len, seq_len)
        mask: mask matrix，shape=(batch, sql_len)

    Returns:
        output，shape=(batch, seq_len)，the index of the parent node corresponding to the token in the query

    """
    lens = mask.sum(1)
    batch_size, seq_len, _ = scores.shape
    scores = scores.transpose(2, 1, 0)
    # Score for incomplete span
    s_i = np.full_like(scores, float('-inf'))
    # Score for complete span
    s_c = np.full_like(scores, float('-inf'))
    # Incompelte span position for backtrack
    p_i = np.zeros((seq_len, seq_len, batch_size), dtype=np.int64)
    # Compelte span position for backtrack
    p_c = np.zeros((seq_len, seq_len, batch_size), dtype=np.int64)
    # Set 0 to s_c.diagonal
    s_c = fill_diagonal(s_c, 0)
    # Contiguous
    s_c = np.ascontiguousarray(s_c)
    s_i = np.ascontiguousarray(s_i)
    for w in range(1, seq_len):
        n = seq_len - w
        starts = np.arange(n, dtype=np.int64)[np.newaxis, :]
        # ilr = C(i->r) + C(j->r+1)
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        # Shape: [batch_size, n, w]
        ilr = ilr.transpose(2, 0, 1)
        # scores.diagonal(-w).shape:[batch, n]
        il = ilr + scores.diagonal(-w)[..., np.newaxis]
        # I(j->i) = max(C(i->r) + C(j->r+1) + s(j->i)), i <= r < j
        il_span, il_path = il.max(-1), il.argmax(-1)
        s_i = fill_diagonal(s_i, il_span, offset=-w)
        p_i = fill_diagonal(p_i, il_path + starts, offset=-w)

        ir = ilr + scores.diagonal(w)[..., np.newaxis]
        # I(i->j) = max(C(i->r) + C(j->r+1) + s(i->j)), i <= r < j
        ir_span, ir_path = ir.max(-1), ir.argmax(-1)
        s_i = fill_diagonal(s_i, ir_span, offset=w)
        p_i = fill_diagonal(p_i, ir_path + starts, offset=w)

        # C(j->i) = max(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        cl = cl.transpose(2, 0, 1)
        cl_span, cl_path = cl.max(-1), cl.argmax(-1)
        s_c = fill_diagonal(s_c, cl_span, offset=-w)
        p_c = fill_diagonal(p_c, cl_path + starts, offset=-w)

        # C(i->j) = max(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr = cr.transpose(2, 0, 1)
        cr_span, cr_path = cr.max(-1), cr.argmax(-1)
        s_c = fill_diagonal(s_c, cr_span, offset=w)
        s_c[0, w][np.not_equal(lens, w)] = float('-inf')
        p_c = fill_diagonal(p_c, cr_path + starts + 1, offset=w)

    predicts = []
    p_c = p_c.transpose(2, 0, 1)
    p_i = p_i.transpose(2, 0, 1)
    for i, length in enumerate(lens.tolist()):
        heads = np.ones(length + 1, dtype=np.int64)
        backtrack(p_i[i], p_c[i], heads, 0, length, True)
        predicts.append(heads)

    return pad_sequence(predicts, fix_len=seq_len)


def fill_diagonal(x, value, offset=0, dim1=0, dim2=1):
    """
    Fill value into the diagoanl of x that offset is ${offset} 
    and the coordinate system is (dim1, dim2).
    """
    strides = x.strides
    shape = x.shape
    if dim1 > dim2:
        dim1, dim2 = dim2, dim1
    assert 0 <= dim1 < dim2 <= 2
    assert len(x.shape) == 3
    assert shape[dim1] == shape[dim2]

    dim_sum = dim1 + dim2
    dim3 = 3 - dim_sum
    if offset >= 0:
        diagonal = np.lib.stride_tricks.as_strided(
            x[:, offset:] if dim_sum == 1 else x[:, :, offset:],
            shape=(shape[dim3], shape[dim1] - offset),
            strides=(strides[dim3], strides[dim1] + strides[dim2]))
    else:
        diagonal = np.lib.stride_tricks.as_strided(
            x[-offset:, :] if dim_sum in [1, 2] else x[:, -offset:],
            shape=(shape[dim3], shape[dim1] + offset),
            strides=(strides[dim3], strides[dim1] + strides[dim2]))

    diagonal[...] = value
    return x


def backtrack(p_i, p_c, heads, i, j, complete):
    """
    Backtrack the position matrix of eisner to generate the tree
    """
    if i == j:
        return
    if complete:
        r = p_c[i, j]
        backtrack(p_i, p_c, heads, i, r, False)
        backtrack(p_i, p_c, heads, r, j, True)
    else:
        r, heads[j] = p_i[i, j], i
        i, j = sorted((i, j))
        backtrack(p_i, p_c, heads, i, r, True)
        backtrack(p_i, p_c, heads, j, r + 1, True)


def stripe(x, n, w, offset=(0, 0), dim=1):
    """
    Returns a diagonal stripe of the tensor.

    Args:
        x (Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 0 if returns a horizontal stripe; 1 else.

    Example:
    >>> x = np.arange(25).reshape(5, 5)
    >>> x
    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    >>> stripe(x, 2, 3, (1, 1))
    tensor([[ 6,  7,  8],
            [12, 13, 14]])
    >>> stripe(x, 2, 3, dim=0)
    tensor([[ 0,  5, 10],
            [ 6, 11, 16]])
    """
    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x)
    strides = x.strides
    m = strides[0] + strides[1]
    k = strides[1] if dim == 1 else strides[0]
    return np.lib.stride_tricks.as_strided(
        x[offset[0]:, offset[1]:],
        shape=[n, w] + list(x.shape[2:]),
        strides=[m, k] + list(strides[2:]))


class Node:
    """Node class"""

    def __init__(self, id=None, parent=None):
        self.lefts = []
        self.rights = []
        self.id = int(id)
        self.parent = parent if parent is None else int(parent)


class DepTree:
    """
    DepTree class, used to check whether the prediction result is a project Tree.
    A projective tree means that you can project the tree without crossing arcs.
    """

    def __init__(self, sentence):
        # set root head to -1
        sentence = copy.deepcopy(sentence)
        sentence[0] = -1
        self.sentence = sentence
        self.build_tree()
        self.visit = [False] * len(sentence)

    def build_tree(self):
        """Build the tree"""
        self.nodes = [
            Node(index, p_index) for index, p_index in enumerate(self.sentence)
        ]
        # set root
        self.root = self.nodes[0]
        for node in self.nodes[1:]:
            self.add(self.nodes[node.parent], node)

    def add(self, parent, child):
        """Add a child node"""
        if parent.id is None or child.id is None:
            raise Exception("id is None")
        if parent.id < child.id:
            parent.rights = sorted(parent.rights + [child.id])
        else:
            parent.lefts = sorted(parent.lefts + [child.id])

    def judge_legal(self):
        """Determine whether it is a project tree"""
        target_seq = list(range(len(self.nodes)))
        if len(self.root.lefts + self.root.rights) != 1:
            return False
        cur_seq = self.inorder_traversal(self.root)
        if target_seq != cur_seq:
            return False
        else:
            return True

    def inorder_traversal(self, node):
        """Inorder traversal"""
        if self.visit[node.id]:
            return []
        self.visit[node.id] = True
        lf_list = []
        rf_list = []
        for ln in node.lefts:
            lf_list += self.inorder_traversal(self.nodes[ln])
        for rn in node.rights:
            rf_list += self.inorder_traversal(self.nodes[rn])

        return lf_list + [node.id] + rf_list


def istree(sequence):
    """Is the sequence a project tree"""
    return DepTree(sequence).judge_legal()
