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

import importlib
import importlib.machinery
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, TypeAlias


SDKObject: TypeAlias = Any
SDKException: TypeAlias = type[BaseException]
ImportModule: TypeAlias = Callable[[str], ModuleType]


@dataclass(frozen=True)
class PaddleOCRAPISDK:
    AsyncPaddleOCRClient: SDKObject
    Model: SDKObject
    OCROptions: SDKObject
    PPStructureV3Options: SDKObject
    PaddleOCRVLOptions: SDKObject
    APIError: SDKException
    AuthError: SDKException
    JobFailedError: SDKException
    RequestTimeoutError: SDKException
    ResponseFormatError: SDKException
    ResultParseError: SDKException
    ServiceUnavailableError: SDKException


_NAMES = (
    "AsyncPaddleOCRClient",
    "Model",
    "OCROptions",
    "PPStructureV3Options",
    "PaddleOCRVLOptions",
    "APIError",
    "AuthError",
    "JobFailedError",
    "RequestTimeoutError",
    "ResponseFormatError",
    "ResultParseError",
    "ServiceUnavailableError",
)


def _sdk_object(module: ModuleType, name: str) -> SDKObject:
    return getattr(module, name)


def _sdk_exception(module: ModuleType, name: str) -> SDKException:
    value = getattr(module, name)
    if not isinstance(value, type) or not issubclass(value, BaseException):
        raise TypeError(f"{module.__name__}.{name} must be an exception class")
    return value


def _build_sdk_from_top_level(paddleocr: ModuleType) -> PaddleOCRAPISDK:
    missing = [name for name in _NAMES if not hasattr(paddleocr, name)]
    if missing:
        raise AttributeError(f"paddleocr is missing API SDK exports: {missing}")
    return PaddleOCRAPISDK(
        AsyncPaddleOCRClient=_sdk_object(paddleocr, "AsyncPaddleOCRClient"),
        Model=_sdk_object(paddleocr, "Model"),
        OCROptions=_sdk_object(paddleocr, "OCROptions"),
        PPStructureV3Options=_sdk_object(paddleocr, "PPStructureV3Options"),
        PaddleOCRVLOptions=_sdk_object(paddleocr, "PaddleOCRVLOptions"),
        APIError=_sdk_exception(paddleocr, "APIError"),
        AuthError=_sdk_exception(paddleocr, "AuthError"),
        JobFailedError=_sdk_exception(paddleocr, "JobFailedError"),
        RequestTimeoutError=_sdk_exception(paddleocr, "RequestTimeoutError"),
        ResponseFormatError=_sdk_exception(paddleocr, "ResponseFormatError"),
        ResultParseError=_sdk_exception(paddleocr, "ResultParseError"),
        ServiceUnavailableError=_sdk_exception(paddleocr, "ServiceUnavailableError"),
    )


def _build_sdk_from_private_modules(import_module: ImportModule) -> PaddleOCRAPISDK:
    async_client = import_module("paddleocr._api_client.async_client")
    models = import_module("paddleocr._api_client.models")
    errors = import_module("paddleocr._api_client.errors")
    return PaddleOCRAPISDK(
        AsyncPaddleOCRClient=_sdk_object(async_client, "AsyncPaddleOCRClient"),
        Model=_sdk_object(models, "Model"),
        OCROptions=_sdk_object(models, "OCROptions"),
        PPStructureV3Options=_sdk_object(models, "PPStructureV3Options"),
        PaddleOCRVLOptions=_sdk_object(models, "PaddleOCRVLOptions"),
        APIError=_sdk_exception(errors, "APIError"),
        AuthError=_sdk_exception(errors, "AuthError"),
        JobFailedError=_sdk_exception(errors, "JobFailedError"),
        RequestTimeoutError=_sdk_exception(errors, "RequestTimeoutError"),
        ResponseFormatError=_sdk_exception(errors, "ResponseFormatError"),
        ResultParseError=_sdk_exception(errors, "ResultParseError"),
        ServiceUnavailableError=_sdk_exception(errors, "ServiceUnavailableError"),
    )


def _find_paddleocr_package_dir() -> Path:
    spec = importlib.machinery.PathFinder.find_spec("paddleocr")
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError("Cannot find paddleocr package")
    return Path(next(iter(spec.submodule_search_locations)))


def _register_api_client_package_from_files() -> None:
    package_dir = _find_paddleocr_package_dir()
    api_client_dir = package_dir / "_api_client"
    init_file = api_client_dir / "__init__.py"
    if not init_file.exists():
        raise ImportError(f"Cannot find PaddleOCR API client at {init_file}")

    if "paddleocr" not in sys.modules:
        parent = ModuleType("paddleocr")
        parent.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
        parent.__package__ = "paddleocr"
        parent.__spec__ = importlib.machinery.ModuleSpec(
            "paddleocr", loader=None, is_package=True
        )
        sys.modules["paddleocr"] = parent

    if "paddleocr._api_client" in sys.modules:
        return

    spec = importlib.util.spec_from_file_location(
        "paddleocr._api_client",
        init_file,
        submodule_search_locations=[str(api_client_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PaddleOCR API client from {init_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["paddleocr._api_client"] = module
    spec.loader.exec_module(module)


def _build_sdk_from_private_module_files() -> PaddleOCRAPISDK:
    _register_api_client_package_from_files()
    return _build_sdk_from_private_modules(importlib.import_module)


def load_paddleocr_api_sdk(
    import_module: ImportModule = importlib.import_module,
) -> PaddleOCRAPISDK:
    try:
        return _build_sdk_from_top_level(import_module("paddleocr"))
    except (AttributeError, ImportError):
        try:
            return _build_sdk_from_private_modules(import_module)
        except ImportError:
            return _build_sdk_from_private_module_files()


_SDK = load_paddleocr_api_sdk()

AsyncPaddleOCRClient = _SDK.AsyncPaddleOCRClient
Model = _SDK.Model
OCROptions = _SDK.OCROptions
PPStructureV3Options = _SDK.PPStructureV3Options
PaddleOCRVLOptions = _SDK.PaddleOCRVLOptions
APIError = _SDK.APIError
AuthError = _SDK.AuthError
JobFailedError = _SDK.JobFailedError
RequestTimeoutError = _SDK.RequestTimeoutError
ResponseFormatError = _SDK.ResponseFormatError
ResultParseError = _SDK.ResultParseError
ServiceUnavailableError = _SDK.ServiceUnavailableError

from paddleocr._api_client._core import (  # noqa: E402
    resolve_document_model,
    resolve_ocr_model,
)
