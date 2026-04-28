import importlib.util
import io
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _load_latexocr_module(monkeypatch):
    module_name = "ppocr.data.latexocr_dataset"
    module_path = (
        Path(__file__).resolve().parents[1] / "ppocr" / "data" / "latexocr_dataset.py"
    )

    ppocr_pkg = types.ModuleType("ppocr")
    ppocr_pkg.__path__ = []
    data_pkg = types.ModuleType("ppocr.data")
    data_pkg.__path__ = []
    imaug_pkg = types.ModuleType("ppocr.data.imaug")
    imaug_pkg.__path__ = []
    label_ops_mod = types.ModuleType("ppocr.data.imaug.label_ops")

    class _Tokenizer:
        def __init__(self, *_args, **_kwargs):
            pass

    label_ops_mod.LatexOCRLabelEncode = _Tokenizer
    imaug_pkg.transform = lambda data, ops: data
    imaug_pkg.create_operators = lambda transforms, global_config: []

    paddle_mod = types.ModuleType("paddle")
    paddle_io_mod = types.ModuleType("paddle.io")

    class _Dataset:
        pass

    paddle_io_mod.Dataset = _Dataset
    paddle_mod.io = paddle_io_mod
    paddle_mod.arange = lambda n: list(range(n))
    paddle_mod.randperm = lambda n: list(range(n))

    for name, module in {
        "ppocr": ppocr_pkg,
        "ppocr.data": data_pkg,
        "ppocr.data.imaug": imaug_pkg,
        "ppocr.data.imaug.label_ops": label_ops_mod,
        "paddle": paddle_mod,
        "paddle.io": paddle_io_mod,
        "cv2": types.ModuleType("cv2"),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _Payload:
    def __init__(self, marker_path: Path):
        self.marker_path = marker_path

    def __reduce__(self):
        expression = (
            f"(__import__('pathlib').Path({self.marker_path.as_posix()!r}).write_text('pwned'), "
            "{(32, 32): [('eq', 'img.png')]})[1]"
        )
        return (eval, (expression,))


@pytest.fixture
def latexocr_config(tmp_path):
    return {
        "Global": {
            "max_seq_len": 16,
            "rec_char_dict_path": str(tmp_path / "dict.txt"),
        },
        "Train": {
            "dataset": {
                "data": str(tmp_path / "dataset.pkl"),
                "data_dir": str(tmp_path),
                "min_dimensions": [1, 1],
                "max_dimensions": [128, 128],
                "batch_size_per_pair": 1,
                "keep_smaller_batches": True,
                "transforms": [],
            },
            "loader": {"shuffle": False},
        },
    }


def test_latexocr_dataset_rejects_executable_pickle_payload(
    tmp_path, latexocr_config, monkeypatch
):
    module = _load_latexocr_module(monkeypatch)
    marker_path = tmp_path / "marker.txt"
    payload_path = Path(latexocr_config["Train"]["dataset"]["data"])
    payload_path.write_bytes(pickle.dumps(_Payload(marker_path)))

    with pytest.raises(pickle.UnpicklingError):
        module.LaTeXOCRDataSet(latexocr_config, "Train", logger=types.SimpleNamespace())

    assert not marker_path.exists()


def test_latexocr_dataset_accepts_basic_dict_payload(
    tmp_path, latexocr_config, monkeypatch
):
    module = _load_latexocr_module(monkeypatch)
    payload_path = Path(latexocr_config["Train"]["dataset"]["data"])
    payload_path.write_bytes(pickle.dumps({(32, 32): [("eq", "img.png")]}))

    dataset = module.LaTeXOCRDataSet(
        latexocr_config, "Train", logger=types.SimpleNamespace()
    )

    assert dataset.size == 1
    assert dataset.data == {(32, 32): [("eq", "img.png")]}
