import importlib.util
import sys
import types
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _stub_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_tools_program_module(monkeypatch):
    _stub_module(monkeypatch, "paddle")
    _stub_module(monkeypatch, "paddle.distributed")
    _stub_module(monkeypatch, "cv2")
    _stub_module(monkeypatch, "numpy")
    _stub_module(monkeypatch, "tqdm", tqdm=lambda *args, **kwargs: None)
    _stub_module(monkeypatch, "ppocr")
    _stub_module(monkeypatch, "ppocr.utils")
    _stub_module(monkeypatch, "ppocr.utils.stats", TrainingStats=object)
    _stub_module(
        monkeypatch, "ppocr.utils.save_load", save_model=lambda *args, **kwargs: None
    )
    _stub_module(
        monkeypatch,
        "ppocr.utils.utility",
        print_dict=lambda *args, **kwargs: None,
        AverageMeter=object,
    )
    _stub_module(
        monkeypatch, "ppocr.utils.logging", get_logger=lambda *args, **kwargs: None
    )
    _stub_module(monkeypatch, "ppocr.utils.loggers", WandbLogger=object, Loggers=object)
    _stub_module(monkeypatch, "ppocr.utils.profiler")
    _stub_module(
        monkeypatch, "ppocr.data", build_dataloader=lambda *args, **kwargs: None
    )
    _stub_module(
        monkeypatch, "ppocr.utils.export_model", export=lambda *args, **kwargs: None
    )

    module_name = "paddleocr_tools_program"
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / "tools" / "program.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_tools_program_load_config_rejects_python_object_tags(tmp_path, monkeypatch):
    module = _load_tools_program_module(monkeypatch)
    payload = (
        '!!python/object/apply:os.system ["echo SHOULD_NOT_RUN > '
        '/tmp/paddleocr_tools_program_test"]\n'
    )
    config_path = tmp_path / "malicious.yml"
    config_path.write_text(payload, encoding="utf-8")

    with pytest.raises(yaml.constructor.ConstructorError):
        module.load_config(str(config_path))


def test_tools_program_parse_opt_rejects_python_object_tags(monkeypatch):
    parser = _load_tools_program_module(monkeypatch).ArgsParser()
    malicious_opt = [
        'Global.debug=!!python/object/apply:os.system ["echo SHOULD_NOT_RUN > '
        '/tmp/paddleocr_opt_test"]'
    ]

    with pytest.raises(yaml.constructor.ConstructorError):
        parser._parse_opt(malicious_opt)
