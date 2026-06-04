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

import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import requests

from .errors import InvalidRequestError, NetworkError, RequestTimeoutError
from .results import DocParsingResult, OCRResult


def save_resource(
    resource_url: str,
    destination: str,
    *,
    overwrite: bool = False,
    filename: Optional[str] = None,
    timeout: float = 300.0,
) -> str:
    if not resource_url:
        raise InvalidRequestError("resource_url is required.")
    if not destination:
        raise InvalidRequestError("destination is required.")

    parsed_url = urlparse(resource_url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        raise InvalidRequestError(f"Invalid resource URL: {resource_url}")

    target = _resolve_destination(parsed_url.path, destination, filename)
    _require_writable_target(target, overwrite)

    try:
        response = requests.get(resource_url, timeout=timeout)
    except requests.Timeout as e:
        raise RequestTimeoutError(f"Request timed out: {e}") from e
    except requests.ConnectionError as e:
        raise NetworkError(f"Connection failed: {e}") from e

    try:
        response.raise_for_status()
    except requests.RequestException as e:
        raise NetworkError(f"Failed to download resource: {e}") from e

    _atomic_write(target, response.content, overwrite)
    return str(target)


def save_ocr_result_resources(
    result: OCRResult,
    destination: str,
    *,
    overwrite: bool = False,
    timeout: float = 300.0,
) -> List[str]:
    if result is None:
        raise InvalidRequestError("OCR result is required.")
    dest_dir = _require_existing_directory(destination)
    saved_paths = []
    for index, page in enumerate(result.pages):
        if not page.ocr_image_url:
            continue
        filename = f"ocr-page-{index + 1}{_safe_resource_extension(page.ocr_image_url)}"
        saved_paths.append(
            save_resource(
                page.ocr_image_url,
                str(dest_dir / filename),
                overwrite=overwrite,
                timeout=timeout,
            )
        )
    return saved_paths


def save_document_parsing_result_resources(
    result: DocParsingResult,
    destination: str,
    *,
    overwrite: bool = False,
    timeout: float = 300.0,
) -> List[str]:
    if result is None:
        raise InvalidRequestError("document parsing result is required.")
    dest_dir = _require_existing_directory(destination)
    saved_paths = []
    for page in result.pages:
        for filename, resource_url in _iter_named_resources(page.markdown_images):
            saved_paths.append(
                save_resource(
                    resource_url,
                    str(dest_dir / filename),
                    overwrite=overwrite,
                    timeout=timeout,
                )
            )
        for filename, resource_url in _iter_named_resources(page.output_images):
            saved_paths.append(
                save_resource(
                    resource_url,
                    str(dest_dir / filename),
                    overwrite=overwrite,
                    timeout=timeout,
                )
            )
    return saved_paths


def _resolve_destination(
    url_path: str, destination: str, filename: Optional[str]
) -> Path:
    destination_path = Path(destination)
    if filename is not None:
        _validate_result_resource_filename(filename)
        target = destination_path / filename
    elif destination_path.exists() and destination_path.is_dir():
        target = destination_path / _safe_url_basename(url_path)
    else:
        target = destination_path

    parent = target.parent
    if not parent.exists():
        raise FileNotFoundError(str(parent))
    if not parent.is_dir():
        raise InvalidRequestError(f"Destination parent must be a directory: {parent}")
    return target


def _require_existing_directory(destination: str) -> Path:
    dest_dir = Path(destination)
    if not dest_dir.exists():
        raise FileNotFoundError(destination)
    if not dest_dir.is_dir():
        raise InvalidRequestError(
            f"Destination must be an existing directory: {destination}"
        )
    return dest_dir


def _require_writable_target(target: Path, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        raise InvalidRequestError(f"Destination already exists: {target}")


def _atomic_write(target: Path, content: bytes, overwrite: bool) -> None:
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "wb") as temp_file:
            temp_file.write(content)
        if overwrite:
            os.replace(temp_path, target)
        else:
            os.link(temp_path, target)
            os.remove(temp_path)
    except FileExistsError as e:
        raise InvalidRequestError(f"Destination already exists: {target}") from e
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _iter_named_resources(resources: Dict[str, str]) -> Iterable[Tuple[str, str]]:
    for key in sorted(resources):
        resource_url = resources[key]
        if not resource_url:
            continue
        _validate_result_resource_filename(key)
        yield key, resource_url


def _safe_url_basename(url_path: str) -> str:
    name = Path(unquote(url_path)).name
    if not name or name in (".", ".."):
        return "resource"
    _validate_result_resource_filename(name)
    return name


def _safe_resource_extension(resource_url: str) -> str:
    parsed = urlparse(resource_url)
    suffix = Path(unquote(parsed.path)).suffix
    if not suffix:
        return ""
    try:
        _validate_result_resource_filename(f"resource{suffix}")
    except InvalidRequestError:
        return ""
    return suffix


def _validate_result_resource_filename(name: str) -> None:
    if not name:
        raise InvalidRequestError("Resource filename must not be empty.")
    path = Path(name)
    if path.name != name or "/" in name or "\\" in name or name in (".", ".."):
        raise InvalidRequestError(f"Unsafe resource filename: {name}")
