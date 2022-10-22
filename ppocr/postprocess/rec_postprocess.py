# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
from paddle.nn import functional as F
import re


class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if 'arabic' in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ''
        for c in pred:
            if not bool(re.search('[a-zA-Z0-9 :*./%+-]', c)):
                if c_current != '':
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ''
            else:
                c_current += c
        if c_current != '':
            pred_re.append(c_current)

        return ''.join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


class DistillationCTCLabelDecode(CTCLabelDecode):
    """
    Convert 
    Convert between text-label and text-index
    """

    def __init__(self,
                 character_dict_path=None,
                 use_space_char=False,
                 model_name=["student"],
                 key=None,
                 multi_head=False,
                 **kwargs):
        super(DistillationCTCLabelDecode, self).__init__(character_dict_path,
                                                         use_space_char)
        if not isinstance(model_name, list):
            model_name = [model_name]
        self.model_name = model_name

        self.key = key
        self.multi_head = multi_head

    def __call__(self, preds, label=None, *args, **kwargs):
        output = dict()
        for name in self.model_name:
            pred = preds[name]
            if self.key is not None:
                pred = pred[self.key]
            if self.multi_head and isinstance(pred, dict):
                pred = pred['ctc']
            output[name] = super().__call__(pred, label=label, *args, **kwargs)
        return output


class AttnLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(AttnLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = dict_character
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        [beg_idx, end_idx] = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(end_idx):
                    break
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        """
        text = self.decode(text)
        if label is None:
            return text
        else:
            label = self.decode(label, is_remove_duplicate=False)
            return text, label
        """
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        return text, label

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx


class RFLLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(RFLLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = dict_character
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        [beg_idx, end_idx] = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(end_idx):
                    break
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        # if seq_outputs is not None:
        if isinstance(preds, tuple) or isinstance(preds, list):
            cnt_outputs, seq_outputs = preds
            if isinstance(seq_outputs, paddle.Tensor):
                seq_outputs = seq_outputs.numpy()
            preds_idx = seq_outputs.argmax(axis=2)
            preds_prob = seq_outputs.max(axis=2)
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)

            if label is None:
                return text
            label = self.decode(label, is_remove_duplicate=False)
            return text, label

        else:
            cnt_outputs = preds
            if isinstance(cnt_outputs, paddle.Tensor):
                cnt_outputs = cnt_outputs.numpy()
            cnt_length = []
            for lens in cnt_outputs:
                length = round(np.sum(lens))
                cnt_length.append(length)
            if label is None:
                return cnt_length
            label = self.decode(label, is_remove_duplicate=False)
            length = [len(res[0]) for res in label]
            return cnt_length, length

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx


class SEEDLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(SEEDLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

    def add_special_char(self, dict_character):
        self.padding_str = "padding"
        self.end_str = "eos"
        self.unknown = "unknown"
        dict_character = dict_character + [
            self.end_str, self.padding_str, self.unknown
        ]
        return dict_character

    def get_ignored_tokens(self):
        end_idx = self.get_beg_end_flag_idx("eos")
        return [end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "sos":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "eos":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupport type %s in get_beg_end_flag_idx" % beg_or_end
        return idx

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        [end_idx] = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if int(text_index[batch_idx][idx]) == int(end_idx):
                    break
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        """
        text = self.decode(text)
        if label is None:
            return text
        else:
            label = self.decode(label, is_remove_duplicate=False)
            return text, label
        """
        preds_idx = preds["rec_pred"]
        if isinstance(preds_idx, paddle.Tensor):
            preds_idx = preds_idx.numpy()
        if "rec_pred_scores" in preds:
            preds_idx = preds["rec_pred"]
            preds_prob = preds["rec_pred_scores"]
        else:
            preds_idx = preds["rec_pred"].argmax(axis=2)
            preds_prob = preds["rec_pred"].max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        return text, label


class SRNLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(SRNLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)
        self.max_text_length = kwargs.get('max_text_length', 25)

    def __call__(self, preds, label=None, *args, **kwargs):
        pred = preds['predict']
        char_num = len(self.character_str) + 2
        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        pred = np.reshape(pred, [-1, char_num])

        preds_idx = np.argmax(pred, axis=1)
        preds_prob = np.max(pred, axis=1)

        preds_idx = np.reshape(preds_idx, [-1, self.max_text_length])

        preds_prob = np.reshape(preds_prob, [-1, self.max_text_length])

        text = self.decode(preds_idx, preds_prob)

        if label is None:
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
            return text
        label = self.decode(label)
        return text, label

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)

            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def add_special_char(self, dict_character):
        dict_character = dict_character + [self.beg_str, self.end_str]
        return dict_character

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx


class SARLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(SARLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

        self.rm_symbol = kwargs.get('rm_symbol', False)

    def add_special_char(self, dict_character):
        beg_end_str = "<BOS/EOS>"
        unknown_str = "<UKN>"
        padding_str = "<PAD>"
        dict_character = dict_character + [unknown_str]
        self.unknown_idx = len(dict_character) - 1
        dict_character = dict_character + [beg_end_str]
        self.start_idx = len(dict_character) - 1
        self.end_idx = len(dict_character) - 1
        dict_character = dict_character + [padding_str]
        self.padding_idx = len(dict_character) - 1
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()

        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(self.end_idx):
                    if text_prob is None and idx == 0:
                        continue
                    else:
                        break
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            if self.rm_symbol:
                comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
                text = text.lower()
                text = comp.sub('', text)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)

        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        return text, label

    def get_ignored_tokens(self):
        return [self.padding_idx]


class DistillationSARLabelDecode(SARLabelDecode):
    """
    Convert 
    Convert between text-label and text-index
    """

    def __init__(self,
                 character_dict_path=None,
                 use_space_char=False,
                 model_name=["student"],
                 key=None,
                 multi_head=False,
                 **kwargs):
        super(DistillationSARLabelDecode, self).__init__(character_dict_path,
                                                         use_space_char)
        if not isinstance(model_name, list):
            model_name = [model_name]
        self.model_name = model_name

        self.key = key
        self.multi_head = multi_head

    def __call__(self, preds, label=None, *args, **kwargs):
        output = dict()
        for name in self.model_name:
            pred = preds[name]
            if self.key is not None:
                pred = pred[self.key]
            if self.multi_head and isinstance(pred, dict):
                pred = pred['sar']
            output[name] = super().__call__(pred, label=label, *args, **kwargs)
        return output


class PRENLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(PRENLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

    def add_special_char(self, dict_character):
        padding_str = '<PAD>'  # 0 
        end_str = '<EOS>'  # 1
        unknown_str = '<UNK>'  # 2

        dict_character = [padding_str, end_str, unknown_str] + dict_character
        self.padding_idx = 0
        self.end_idx = 1
        self.unknown_idx = 2

        return dict_character

    def decode(self, text_index, text_prob=None):
        """ convert text-index into text-label. """
        result_list = []
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] == self.end_idx:
                    break
                if text_index[batch_idx][idx] in \
                    [self.padding_idx, self.unknown_idx]:
                    continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)

            text = ''.join(char_list)
            if len(text) > 0:
                result_list.append((text, np.mean(conf_list).tolist()))
            else:
                # here confidence of empty recog result is 1
                result_list.append(('', 1))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob)
        if label is None:
            return text
        label = self.decode(label)
        return text, label


class NRTRLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=True, **kwargs):
        super(NRTRLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):

        if len(preds) == 2:
            preds_id = preds[0]
            preds_prob = preds[1]
            if isinstance(preds_id, paddle.Tensor):
                preds_id = preds_id.numpy()
            if isinstance(preds_prob, paddle.Tensor):
                preds_prob = preds_prob.numpy()
            if preds_id[0][0] == 2:
                preds_idx = preds_id[:, 1:]
                preds_prob = preds_prob[:, 1:]
            else:
                preds_idx = preds_id
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
            if label is None:
                return text
            label = self.decode(label[:, 1:])
        else:
            if isinstance(preds, paddle.Tensor):
                preds = preds.numpy()
            preds_idx = preds.argmax(axis=2)
            preds_prob = preds.max(axis=2)
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
            if label is None:
                return text
            label = self.decode(label[:, 1:])
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank', '<unk>', '<s>', '</s>'] + dict_character
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                try:
                    char_idx = self.character[int(text_index[batch_idx][idx])]
                except:
                    continue
                if char_idx == '</s>':  # end
                    break
                char_list.append(char_idx)
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text.lower(), np.mean(conf_list).tolist()))
        return result_list


class ViTSTRLabelDecode(NRTRLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(ViTSTRLabelDecode, self).__init__(character_dict_path,
                                                use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, paddle.Tensor):
            preds = preds[:, 1:].numpy()
        else:
            preds = preds[:, 1:]
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None:
            return text
        label = self.decode(label[:, 1:])
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['<s>', '</s>'] + dict_character
        return dict_character


class ABINetLabelDecode(NRTRLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(ABINetLabelDecode, self).__init__(character_dict_path,
                                                use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, dict):
            preds = preds['align'][-1].numpy()
        elif isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        else:
            preds = preds

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['</s>'] + dict_character
        return dict_character


class SPINLabelDecode(AttnLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(SPINLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = dict_character
        dict_character = [self.beg_str] + [self.end_str] + dict_character
        return dict_character


class VLLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(VLLabelDecode, self).__init__(character_dict_path, use_space_char)
        self.max_text_length = kwargs.get('max_text_length', 25)
        self.nclass = len(self.character) + 1

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id - 1]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, label=None, length=None, *args, **kwargs):
        if len(preds) == 2:  # eval mode
            text_pre, x = preds
            b = text_pre.shape[1]
            lenText = self.max_text_length
            nsteps = self.max_text_length

            if not isinstance(text_pre, paddle.Tensor):
                text_pre = paddle.to_tensor(text_pre, dtype='float32')

            out_res = paddle.zeros(
                shape=[lenText, b, self.nclass], dtype=x.dtype)
            out_length = paddle.zeros(shape=[b], dtype=x.dtype)
            now_step = 0
            for _ in range(nsteps):
                if 0 in out_length and now_step < nsteps:
                    tmp_result = text_pre[now_step, :, :]
                    out_res[now_step] = tmp_result
                    tmp_result = tmp_result.topk(1)[1].squeeze(axis=1)
                    for j in range(b):
                        if out_length[j] == 0 and tmp_result[j] == 0:
                            out_length[j] = now_step + 1
                    now_step += 1
            for j in range(0, b):
                if int(out_length[j]) == 0:
                    out_length[j] = nsteps
            start = 0
            output = paddle.zeros(
                shape=[int(out_length.sum()), self.nclass], dtype=x.dtype)
            for i in range(0, b):
                cur_length = int(out_length[i])
                output[start:start + cur_length] = out_res[0:cur_length, i, :]
                start += cur_length
            net_out = output
            length = out_length

        else:  # train mode
            net_out = preds[0]
            length = length
            net_out = paddle.concat([t[:l] for t, l in zip(net_out, length)])
        text = []
        if not isinstance(net_out, paddle.Tensor):
            net_out = paddle.to_tensor(net_out, dtype='float32')
        net_out = F.softmax(net_out, axis=1)
        for i in range(0, length.shape[0]):
            preds_idx = net_out[int(length[:i].sum()):int(length[:i].sum(
            ) + length[i])].topk(1)[1][:, 0].tolist()
            preds_text = ''.join([
                self.character[idx - 1]
                if idx > 0 and idx <= len(self.character) else ''
                for idx in preds_idx
            ])
            preds_prob = net_out[int(length[:i].sum()):int(length[:i].sum(
            ) + length[i])].topk(1)[0][:, 0]
            preds_prob = paddle.exp(
                paddle.log(preds_prob).sum() / (preds_prob.shape[0] + 1e-6))
            text.append((preds_text, preds_prob.numpy()[0]))
        if label is None:
            return text
        label = self.decode(label)
        return text, label


class CANLabelDecode(BaseRecLabelDecode):
    """ Convert between latex-symbol and symbol-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(CANLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

    def decode(self, text_index, preds_prob=None):
        result_list = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            seq_end = text_index[batch_idx].argmin(0)
            idx_list = text_index[batch_idx][:seq_end].tolist()
            symbol_list = [self.character[idx] for idx in idx_list]
            probs = []
            if preds_prob is not None:
                probs = preds_prob[batch_idx][:len(symbol_list)].tolist()

            result_list.append([' '.join(symbol_list), probs])
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        pred_prob, _, _, _ = preds
        preds_idx = pred_prob.argmax(axis=2)

        text = self.decode(preds_idx)
        if label is None:
            return text
        label = self.decode(label)
        return text, label
