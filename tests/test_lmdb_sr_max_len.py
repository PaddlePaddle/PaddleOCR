"""Regression test for LMDBDataSetSR.get_lmdb_sample_info max-length handling.

Loads ppocr/data/lmdb_dataset.py in isolation with the heavy optional
dependencies (paddle, cv2, lmdb, PIL, numpy) stubbed, so the test runs without
the full PaddlePaddle stack.

It guards the fix for the broken ``except IOError or len(word) > self.max_len``
clause: because ``A or B`` short-circuits to the truthy ``IOError`` class, the
length check used to be dead code and over-long words were never skipped. The
clause is now split into a real ``except IOError`` plus an explicit length
guard, so an over-long sample is correctly skipped to the next index.
"""

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _stub_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_lmdb_dataset_module(monkeypatch):
    _stub_module(monkeypatch, "cv2")
    _stub_module(
        monkeypatch, "numpy", random=types.SimpleNamespace(randint=lambda upper: 0)
    )
    _stub_module(monkeypatch, "lmdb", open=lambda *args, **kwargs: None)
    _stub_module(monkeypatch, "PIL")
    _stub_module(monkeypatch, "PIL.Image")

    paddle_module = _stub_module(monkeypatch, "paddle")
    paddle_io_module = _stub_module(
        monkeypatch, "paddle.io", Dataset=type("Dataset", (), {})
    )
    paddle_module.io = paddle_io_module

    ppocr_module = _stub_module(monkeypatch, "ppocr")
    ppocr_data_module = _stub_module(monkeypatch, "ppocr.data")
    ppocr_data_module.__path__ = []
    ppocr_module.data = ppocr_data_module
    _stub_module(
        monkeypatch,
        "ppocr.data.imaug",
        transform=lambda data, ops: data,
        create_operators=lambda *args, **kwargs: [],
    )

    spec = importlib.util.spec_from_file_location(
        "ppocr.data.lmdb_dataset", REPO_ROOT / "ppocr" / "data" / "lmdb_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "ppocr.data.lmdb_dataset", module)
    spec.loader.exec_module(module)
    return module


class _Txn:
    """Minimal lmdb transaction returning a fixed label for the label key."""

    def __init__(self, label):
        self._label = label

    def get(self, key):
        return self._label.encode()


def _make_sr_dataset(monkeypatch, module):
    dataset = module.LMDBDataSetSR.__new__(module.LMDBDataSetSR)
    # Avoid real image decoding (PIL is stubbed).
    monkeypatch.setattr(dataset, "buf2PIL", lambda txn, key, type="RGB": "IMG")
    # Capture the recursion target instead of running a real __getitem__.
    monkeypatch.setattr(
        module.LMDBDataSetSR, "__getitem__", lambda self, idx: ("SKIPPED", idx)
    )
    return dataset


def test_sr_sample_skips_overlong_word(monkeypatch):
    module = _load_lmdb_dataset_module(monkeypatch)
    dataset = _make_sr_dataset(monkeypatch, module)
    # max_len is hard-coded to 100 inside get_lmdb_sample_info.
    txn = _Txn("a" * 150)
    result = dataset.get_lmdb_sample_info(txn, 0)
    assert result == ("SKIPPED", 1)


def test_sr_sample_accepts_normal_word(monkeypatch):
    module = _load_lmdb_dataset_module(monkeypatch)
    dataset = _make_sr_dataset(monkeypatch, module)
    txn = _Txn("HELLO")
    img_hr, img_lr, label = dataset.get_lmdb_sample_info(txn, 0)
    assert (img_hr, img_lr) == ("IMG", "IMG")
    assert label == "HELLO"
