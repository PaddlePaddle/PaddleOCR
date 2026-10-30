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

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "tools" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool("expand_rec_dict")

OLD = list("abcde")
D_MODEL = 8


def _multihead_state_dict(n_ctc, rng):
    """Head layers of a CTC + NRTR MultiHead model with n_ctc CTC classes."""
    n_nrtr = n_ctc + 4
    return {
        "backbone.conv.weight": rng.standard_normal((4, 3, 3, 3)),
        "head.ctc_encoder.encoder.fc.weight": rng.standard_normal((D_MODEL, n_ctc)),
        "head.ctc_head.fc.weight": rng.standard_normal((D_MODEL, n_ctc)),
        "head.ctc_head.fc.bias": rng.standard_normal((n_ctc,)),
        "head.gtc_head.embedding.embedding.weight": rng.standard_normal(
            (n_nrtr, D_MODEL)
        ),
        "head.gtc_head.tgt_word_prj.weight": rng.standard_normal((D_MODEL, n_nrtr)),
    }


def _rows(sd, key, axis):
    return np.moveaxis(sd[key], axis, 0)


def test_same_dict_is_identity():
    rng = np.random.default_rng(0)
    sd = _multihead_state_dict(len(OLD) + 2, rng)
    out, report, stats = tool.expand_state_dict(sd, OLD, OLD + [" "])
    assert len(report) == 4
    assert stats == (len(OLD) + 1, 0, 0)
    for k in sd:
        np.testing.assert_array_equal(out[k], sd[k])


def test_appended_chars_keep_old_rows():
    rng = np.random.default_rng(0)
    sd = _multihead_state_dict(len(OLD) + 2, rng)
    new = OLD + ["x", "y"] + [" "]
    out, report, stats = tool.expand_state_dict(sd, OLD, new)
    assert stats == (6, 2, 0)

    # CTC: blank + a..e + x y + space
    w_old, w_new = _rows(sd, "head.ctc_head.fc.weight", 1), _rows(
        out, "head.ctc_head.fc.weight", 1
    )
    assert w_new.shape[0] == 9
    np.testing.assert_array_equal(w_new[:6], w_old[:6])
    np.testing.assert_array_equal(w_new[8], w_old[6])
    bias = out["head.ctc_head.fc.bias"]
    assert np.allclose(bias[6:8], sd["head.ctc_head.fc.bias"].mean())
    np.testing.assert_allclose(w_new[6], w_old.mean(axis=0))

    # NRTR: 4 special tokens + a..e + x y + space + extra row
    for key, axis in [
        ("head.gtc_head.embedding.embedding.weight", 0),
        ("head.gtc_head.tgt_word_prj.weight", 1),
    ]:
        old_rows, new_rows = _rows(sd, key, axis), _rows(out, key, axis)
        assert new_rows.shape[0] == 13
        np.testing.assert_array_equal(new_rows[:9], old_rows[:9])
        np.testing.assert_array_equal(new_rows[11:], old_rows[9:])

    # untouched parameters, including one whose size only looks like a head
    for k in ["backbone.conv.weight", "head.ctc_encoder.encoder.fc.weight"]:
        np.testing.assert_array_equal(out[k], sd[k])


def test_new_dict_may_insert_and_reorder():
    rng = np.random.default_rng(0)
    sd = _multihead_state_dict(len(OLD) + 2, rng)
    new = ["e", "x", "a", "b", "c", "d", " "]
    out, _, stats = tool.expand_state_dict(sd, OLD, new)
    assert stats == (6, 1, 0)
    w_old, w_new = _rows(sd, "head.ctc_head.fc.weight", 1), _rows(
        out, "head.ctc_head.fc.weight", 1
    )
    for i, c in enumerate(["blank"] + new):
        if c != "x":
            j = (["blank"] + OLD + [" "]).index(c)
            np.testing.assert_array_equal(w_new[i], w_old[j])


def test_space_char_inferred_and_dropped():
    rng = np.random.default_rng(0)
    sd = _multihead_state_dict(len(OLD) + 2, rng)  # pretrained with space
    out, _, stats = tool.expand_state_dict(sd, OLD, OLD + ["x"])
    assert stats == (5, 1, 1)
    assert out["head.ctc_head.fc.bias"].shape == (7,)

    sd = _multihead_state_dict(len(OLD) + 1, rng)  # pretrained without space
    out, _, stats = tool.expand_state_dict(sd, OLD, OLD + [" "])
    assert stats == (5, 1, 0)
    assert out["head.gtc_head.tgt_word_prj.weight"].shape == (D_MODEL, 11)


def test_ctc_only_and_sar_heads():
    rng = np.random.default_rng(0)
    n = len(OLD) + 1  # dict + space
    sd = {
        "Student.head.fc.weight": rng.standard_normal((D_MODEL, n + 1)),
        "Student.head.fc.bias": rng.standard_normal((n + 1,)),
        "head.sar_head.decoder.embedding.weight": rng.standard_normal((n + 3, D_MODEL)),
        "head.sar_head.decoder.prediction.weight": rng.standard_normal(
            (D_MODEL, n + 2)
        ),
    }
    new = OLD + ["x", " "]
    out, report, _ = tool.expand_state_dict(sd, OLD, new)
    assert len(report) == 4
    assert out["Student.head.fc.weight"].shape == (D_MODEL, n + 2)
    emb_old = sd["head.sar_head.decoder.embedding.weight"]
    emb = out["head.sar_head.decoder.embedding.weight"]
    assert emb.shape == (n + 4, D_MODEL)
    # a..e + space keep their rows, x is new, then <UKN> <BOS/EOS> <PAD>
    np.testing.assert_array_equal(emb[:5], emb_old[:5])
    np.testing.assert_array_equal(emb[6], emb_old[5])
    np.testing.assert_array_equal(emb[7:], emb_old[6:])
    pred = out["head.sar_head.decoder.prediction.weight"]
    assert pred.shape == (D_MODEL, n + 3)
    np.testing.assert_array_equal(
        pred[:, 7:], sd["head.sar_head.decoder.prediction.weight"][:, 6:]
    )


def test_duplicate_chars_map_in_order():
    mapping = tool.row_mapping(["a", "b", "a"], ["a", "b", "a", "a"])
    assert mapping.tolist() == [0, 1, 2, -1]


def test_wrong_old_dict_raises():
    rng = np.random.default_rng(0)
    sd = _multihead_state_dict(len(OLD) + 2, rng)
    with pytest.raises(ValueError, match="old_dict_path"):
        tool.expand_state_dict(sd, OLD + ["z", "w"], OLD + [" "])


def test_new_classes_never_win_before_training():
    rng = np.random.default_rng(0)
    sd = _multihead_state_dict(len(OLD) + 2, rng)
    out, _, _ = tool.expand_state_dict(sd, OLD, OLD + list("xyz") + [" "])
    feats = rng.standard_normal((1000, D_MODEL))
    old_logits = feats @ sd["head.ctc_head.fc.weight"] + sd["head.ctc_head.fc.bias"]
    new_logits = feats @ out["head.ctc_head.fc.weight"] + out["head.ctc_head.fc.bias"]
    np.testing.assert_allclose(new_logits[:, :6], old_logits[:, :6])
    assert (new_logits[:, 6:9].max(axis=1) <= old_logits.max(axis=1) + 1e-9).all()
