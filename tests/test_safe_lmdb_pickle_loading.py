import importlib.util
import os
import pickle
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _stub_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_lmdb_dataset_module(monkeypatch, lmdb_open):
    _stub_module(monkeypatch, "cv2")
    _stub_module(
        monkeypatch, "numpy", random=types.SimpleNamespace(randint=lambda upper: 0)
    )
    _stub_module(monkeypatch, "lmdb", open=lmdb_open)
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


class _Exploit:
    def __reduce__(self):
        return (os.system, ("echo SHOULD_NOT_RUN > /tmp/paddleocr_lmdb_pickle_test",))


class _Txn:
    def __init__(self, values):
        self._values = values

    def get(self, key):
        return self._values.get(key)


class _Env:
    def __init__(self, txn):
        self._txn = txn

    def begin(self, write=False):
        return self._txn


@pytest.fixture(autouse=True)
def _cleanup_marker():
    marker = Path("/tmp/paddleocr_lmdb_pickle_test")
    marker.unlink(missing_ok=True)
    yield
    marker.unlink(missing_ok=True)


def test_tablemaster_sample_info_rejects_pickle_rce_payload(monkeypatch):
    module = _load_lmdb_dataset_module(
        monkeypatch, lmdb_open=lambda *args, **kwargs: None
    )
    payload = pickle.dumps(_Exploit())
    txn = _Txn({b"1": payload})
    dataset = module.LMDBDataSetTableMaster.__new__(module.LMDBDataSetTableMaster)

    assert dataset.get_lmdb_sample_info(txn, 1) is None
    assert not Path("/tmp/paddleocr_lmdb_pickle_test").exists()


def test_tablemaster_sample_info_accepts_expected_basic_pickle_data(monkeypatch):
    module = _load_lmdb_dataset_module(
        monkeypatch, lmdb_open=lambda *args, **kwargs: None
    )
    safe_payload = pickle.dumps(("sample.png", b"img-bytes", "raw-name\ntext\n1,2,3,4"))
    txn = _Txn({b"1": safe_payload})
    dataset = module.LMDBDataSetTableMaster.__new__(module.LMDBDataSetTableMaster)

    sample = dataset.get_lmdb_sample_info(txn, 1)

    assert sample["file_name"] == "sample.png"
    assert sample["image"] == b"img-bytes"
    assert sample["structure"] == ["text"]
    assert sample["cells"] == [{"bbox": [1, 2, 3, 4], "tokens": ["1", "2"]}]


def test_tablemaster_length_metadata_rejects_pickle_rce_payload(monkeypatch):
    payload = pickle.dumps(_Exploit())
    txn = _Txn({b"__len__": payload})
    module = _load_lmdb_dataset_module(
        monkeypatch, lmdb_open=lambda *args, **kwargs: _Env(txn)
    )
    dataset = module.LMDBDataSetTableMaster.__new__(module.LMDBDataSetTableMaster)

    with pytest.raises(pickle.UnpicklingError):
        dataset.load_hierarchical_lmdb_dataset("/tmp/ignored")

    assert not Path("/tmp/paddleocr_lmdb_pickle_test").exists()


def test_tablemaster_length_metadata_accepts_expected_integer_pickle(monkeypatch):
    txn = _Txn({b"__len__": pickle.dumps(7)})
    module = _load_lmdb_dataset_module(
        monkeypatch, lmdb_open=lambda *args, **kwargs: _Env(txn)
    )
    dataset = module.LMDBDataSetTableMaster.__new__(module.LMDBDataSetTableMaster)

    lmdb_sets = dataset.load_hierarchical_lmdb_dataset("/tmp/ignored")

    assert lmdb_sets[0]["num_samples"] == 7
