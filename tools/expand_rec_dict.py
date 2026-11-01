# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
Remap the dict-sized head layers of a pretrained rec model to a new dict.

Without this, load_pretrained_params skips these layers when the dict size
changes and they start from random init. Each character keeps its pretrained
row; new characters start from the mean of the existing rows, so the
converted model decodes the same as the original before training.

Supports CTCHead, MultiHead (CTC + NRTR / CTC + SAR) and distillation models.

    python3 tools/expand_rec_dict.py \
        -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml \
        -o Global.pretrained_model=./PP-OCRv5_server_rec_pretrained.pdparams \
           Global.character_dict_path=./my_dict.txt \
        --old_dict_path ./ppocr/utils/dict/ppocrv5_dict.txt \
        --save_path ./output/PP-OCRv5_server_rec_my_dict
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from collections import defaultdict

import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

# Class order of each head, built from `chars` (the dictionary, plus " " when
# use_space_char is true). They follow the label decoders in
# ppocr/postprocess/rec_postprocess.py and the out_channels set in
# tools/train.py:
#   CTC:  CTCLabelDecode adds "blank" in front.
#   NRTR: NRTRLabelEncode adds 4 special tokens in front; train.py sets
#         out_channels to len(CTC classes) + 3 and NRTRHead adds 1 more.
#   SAR:  SARLabelDecode appends 3 tokens; the prediction layer drops <PAD>.
_NRTR_PAD = "<nrtr_extra>"


def head_classes(head, chars):
    if head == "ctc":
        return ["blank"] + chars
    if head == "nrtr":
        return ["blank", "<unk>", "<s>", "</s>"] + chars + [_NRTR_PAD]
    if head == "sar":
        return chars + ["<UKN>", "<BOS/EOS>", "<PAD>"]
    raise ValueError("unknown head type: {}".format(head))


def head_type(key):
    """Which head a parameter belongs to, from its name; None if not a head."""
    if "head." not in key:
        return None
    if "sar_head" in key:
        return "sar"
    if "gtc_head" in key:
        return "nrtr"
    if "ctc_encoder" in key:
        return None
    return "ctc"


def row_mapping(old_classes, new_classes):
    """For each new class, the row index in the old layer, or -1 if new.

    Classes are matched by name. A class listed more than once (some
    dictionaries contain duplicates) is matched occurrence by occurrence, so
    an unchanged dictionary maps to the identity.
    """
    rows = defaultdict(list)
    for i, c in enumerate(old_classes):
        rows[c].append(i)
    used = defaultdict(int)
    mapping = []
    for c in new_classes:
        k = used[c]
        used[c] += 1
        mapping.append(rows[c][k] if k < len(rows[c]) else -1)
    return np.array(mapping, dtype=np.int64)


def remap_rows(value, axis, mapping):
    """Gather rows of `value` along `axis`; rows mapped to -1 take the mean
    of all existing rows."""
    moved = np.moveaxis(value, axis, 0)
    out = moved[np.maximum(mapping, 0)].copy()
    new = mapping < 0
    if new.any():
        out[new] = moved.astype(np.float64).mean(axis=0)
    return np.moveaxis(out, 0, axis)


def class_axis(key, value):
    """The axis that would hold the classes: paddle.nn.Linear stores weights
    as [in, out] and nn.Embedding as [num_embeddings, dim]."""
    if value.ndim == 1 or "embedding" in key:
        return 0
    return value.ndim - 1


def _match(state_dict, chars):
    """Find (key, head, axis, classes) for every head layer sized for `chars`."""
    found = []
    for key, value in state_dict.items():
        head = head_type(key)
        if head is None or value.ndim == 0:
            continue
        classes = head_classes(head, chars)
        sizes = {len(classes): classes}
        if head == "sar":
            sizes[len(classes) - 1] = classes[:-1]
        axis = class_axis(key, value)
        if value.shape[axis] in sizes:
            found.append((key, head, axis, sizes[value.shape[axis]]))
    return found


def expand_state_dict(state_dict, old_dict, new_chars):
    """Remap the head layers of `state_dict` from `old_dict` to `new_chars`.

    Args:
        state_dict: dict of name -> numpy array (the pretrained weights).
        old_dict: list of characters the pretrained model was trained with,
            without the space appended by use_space_char; whether the
            pretrained model used it is inferred from the layer sizes.
        new_chars: list of classes of the new model, i.e. the new dictionary
            plus " " if use_space_char is true.
    Returns:
        (new state dict, list of (key, head, old shape, new shape),
         number of kept, added and removed classes of the CTC/main head).
    """
    candidates = [_match(state_dict, old_dict + [" "]), _match(state_dict, old_dict)]
    found = max(candidates, key=len)
    if not found:
        raise ValueError(
            "No head layer of the pretrained model matches the size of the old "
            "dictionary ({} characters). Check --old_dict_path.".format(len(old_dict))
        )
    old_chars = old_dict + [" "] if found is candidates[0] else old_dict

    out = dict(state_dict)
    report = []
    for key, head, axis, old_classes in found:
        new_classes = head_classes(head, new_chars)
        if len(old_classes) < len(head_classes(head, old_chars)):
            new_classes = new_classes[:-1]  # SAR prediction layer, no <PAD>
        mapping = row_mapping(old_classes, new_classes)
        out[key] = remap_rows(state_dict[key], axis, mapping)
        report.append((key, head, list(state_dict[key].shape), list(out[key].shape)))

    mapping = row_mapping(old_chars, new_chars)
    kept = int((mapping >= 0).sum())
    stats = (kept, len(new_chars) - kept, len(old_chars) - kept)
    return out, report, stats


def main():
    import paddle

    import tools.program as program
    from ppocr.postprocess.rec_postprocess import BaseRecLabelDecode
    from ppocr.utils.network import maybe_download_params

    parser = program.ArgsParser()
    parser.add_argument(
        "--old_dict_path",
        required=True,
        help="dictionary the pretrained model was trained with",
    )
    parser.add_argument(
        "--save_path",
        required=True,
        help="where to save the remapped weights (.pdparams added if missing)",
    )
    args = parser.parse_args()
    config = program.merge_config(program.load_config(args.config), args.opt)
    global_config = config["Global"]

    src = global_config.get("pretrained_model")
    assert src, "Please set Global.pretrained_model to the pretrained weights."
    src = maybe_download_params(src)
    if not src.endswith(".pdparams"):
        src += ".pdparams"
    use_space_char = global_config.get("use_space_char", False)
    new_chars = BaseRecLabelDecode(
        global_config.get("character_dict_path"), use_space_char
    ).character
    old_dict = BaseRecLabelDecode(args.old_dict_path, False).character

    params = paddle.load(src)
    state_dict = {k: v.numpy() for k, v in params.items()}
    out, report, (kept, added, removed) = expand_state_dict(
        state_dict, old_dict, new_chars
    )
    for key, head, old_shape, new_shape in report:
        print("{:5s}{}: {} -> {}".format(head.upper(), key, old_shape, new_shape))
    print("classes kept: {}, added: {}, removed: {}".format(kept, added, removed))
    if removed:
        print(
            "Warning: {} classes of the old dictionary are not in the new one; "
            "their rows are dropped.".format(removed)
        )

    save_path = args.save_path
    if not save_path.endswith(".pdparams"):
        save_path += ".pdparams"
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    paddle.save({k: paddle.to_tensor(v) for k, v in out.items()}, save_path)
    print("saved to {}".format(save_path))


if __name__ == "__main__":
    main()
