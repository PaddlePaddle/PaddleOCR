import importlib.util
import sys
import types
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _load_tools_program_module():
    _stub_module("paddle")
    _stub_module("paddle.distributed")
    _stub_module("cv2")
    _stub_module("numpy")
    _stub_module("tqdm", tqdm=lambda *args, **kwargs: None)
    _stub_module("ppocr")
    _stub_module("ppocr.utils")
    _stub_module("ppocr.utils.stats", TrainingStats=object)
    _stub_module("ppocr.utils.save_load", save_model=lambda *args, **kwargs: None)
    _stub_module(
        "ppocr.utils.utility",
        print_dict=lambda *args, **kwargs: None,
        AverageMeter=object,
    )
    _stub_module("ppocr.utils.logging", get_logger=lambda *args, **kwargs: None)
    _stub_module("ppocr.utils.loggers", WandbLogger=object, Loggers=object)
    _stub_module("ppocr.utils.profiler")
    _stub_module("ppocr.data", build_dataloader=lambda *args, **kwargs: None)
    _stub_module("ppocr.utils.export_model", export=lambda *args, **kwargs: None)

    spec = importlib.util.spec_from_file_location(
        "paddleocr_tools_program", REPO_ROOT / "tools" / "program.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tools_program_load_config_rejects_python_object_tags(tmp_path):
    module = _load_tools_program_module()
    payload = (
        '!!python/object/apply:os.system ["echo SHOULD_NOT_RUN > '
        '/tmp/paddleocr_tools_program_test"]\n'
    )
    config_path = tmp_path / "malicious.yml"
    config_path.write_text(payload, encoding="utf-8")

    with pytest.raises(yaml.constructor.ConstructorError):
        module.load_config(str(config_path))


def test_tools_program_parse_opt_rejects_python_object_tags():
    parser = _load_tools_program_module().ArgsParser()
    malicious_opt = [
        'Global.debug=!!python/object/apply:os.system ["echo SHOULD_NOT_RUN > '
        '/tmp/paddleocr_opt_test"]'
    ]

    with pytest.raises(yaml.constructor.ConstructorError):
        parser._parse_opt(malicious_opt)
