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
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import requests

from ._http import DEFAULT_BASE_URL, HTTPClient, _extract_api_message
from ._poller import Poller, parse_doc_parsing_result, parse_ocr_result
from .errors import (
    APIError,
    AuthError,
    FileNotFoundError,
    InvalidRequestError,
    NetworkError,
    RequestTimeoutError,
)
from .models import (
    DocParsingOptions,
    Model,
    OCROptions,
    is_document_parsing_model,
    is_ocr_model,
)
from .results import (
    DocParsingResult,
    Job,
    JobStatus,
    OCRResult,
    ResourceSaveSummary,
)


class APIClient:
    """Synchronous blocking client for PaddleOCR official API.

    Wraps the async job API internally: submit → poll → fetch result.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        request_timeout: float = 300.0,
        poll_timeout: float = 600.0,
    ):
        self._token = token or os.environ.get("PADDLEOCR_ACCESS_TOKEN", "")
        if not self._token:
            raise AuthError(
                "Token is required. Set PADDLEOCR_ACCESS_TOKEN or pass token=."
            )
        self._http = HTTPClient(self._token, base_url, request_timeout)
        self._poller = Poller(self._http, max_wait_time=poll_timeout)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._http.close()

    def ocr(
        self,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[OCROptions] = None,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
        model: Union[Model, str] = Model.PP_OCRV5,
    ) -> OCRResult:
        model = self._ocr_model(model)
        job_id = self._submit(
            model,
            file_url,
            file_path,
            options,
            page_ranges,
            batch_id,
        )
        jsonl_data, _ = self._poller.poll_until_done(job_id)
        return parse_ocr_result(job_id, jsonl_data)

    def parse_document(
        self,
        model: Union[Model, str] = Model.PADDLE_OCR_VL_15,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[DocParsingOptions] = None,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> DocParsingResult:
        model = self._document_model(model)
        job_id = self._submit(
            model, file_url, file_path, options, page_ranges, batch_id
        )
        jsonl_data, _ = self._poller.poll_until_done(job_id)
        return parse_doc_parsing_result(job_id, jsonl_data)

    def submit_ocr(
        self,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[OCROptions] = None,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
        model: Union[Model, str] = Model.PP_OCRV5,
    ) -> Job:
        model = self._ocr_model(model)
        job_id = self._submit(
            model,
            file_url,
            file_path,
            options,
            page_ranges,
            batch_id,
        )
        return Job(job_id=job_id, model=model.value, task="ocr")

    def submit_document_parsing(
        self,
        model: Union[Model, str] = Model.PADDLE_OCR_VL_15,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[DocParsingOptions] = None,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> Job:
        model = self._document_model(model)
        job_id = self._submit(
            model, file_url, file_path, options, page_ranges, batch_id
        )
        return Job(job_id=job_id, model=model.value, task="document_parsing")

    def get_status(self, job_id: str) -> JobStatus:
        return self._poller.get_status(job_id)

    def wait_ocr_result(self, job: Union[Job, str]) -> OCRResult:
        job_id = self._job_id_for_task(job, "ocr")
        jsonl_data, _ = self._poller.poll_until_done(job_id)
        return parse_ocr_result(job_id, jsonl_data)

    def wait_document_parsing_result(self, job: Union[Job, str]) -> DocParsingResult:
        job_id = self._job_id_for_task(job, "document_parsing")
        jsonl_data, _ = self._poller.poll_until_done(job_id)
        return parse_doc_parsing_result(job_id, jsonl_data)

    def save_resource(
        self,
        resource: Union[str, OCRResult, DocParsingResult],
        destination: Union[str, os.PathLike],
        overwrite: bool = False,
    ) -> Union[str, ResourceSaveSummary]:
        destination_path = Path(destination)
        if isinstance(resource, str):
            return str(self._download_resource(resource, destination_path, overwrite))

        if not destination_path.exists():
            raise FileNotFoundError(str(destination_path))
        if not destination_path.is_dir():
            raise InvalidRequestError(
                "destination must be a directory when saving result resources."
            )

        summary = ResourceSaveSummary()
        for relative_name, url in self._iter_result_resources(resource):
            target_path = destination_path / relative_name
            saved_path = self._download_resource(url, target_path, overwrite)
            summary.saved_paths.append(str(saved_path))
        return summary

    def _submit(
        self,
        model: Model,
        file_url: Optional[str],
        file_path: Optional[str],
        options,
        page_ranges: Optional[str],
        batch_id: Optional[str],
    ) -> str:
        if not file_url and not file_path:
            raise InvalidRequestError("Either file_url or file_path is required.")
        if file_url and file_path:
            raise InvalidRequestError("file_url and file_path are mutually exclusive.")
        payload = options.to_payload() if options else self._default_payload(model)
        if file_url:
            return self._http.submit_url(
                model.value,
                file_url,
                payload,
                page_ranges=page_ranges,
                batch_id=batch_id,
            )
        return self._http.submit_file(
            model.value,
            file_path,
            payload,
            page_ranges=page_ranges,
            batch_id=batch_id,
        )

    def _default_payload(self, model: Model) -> dict:
        if is_ocr_model(model):
            return OCROptions().to_payload()
        return DocParsingOptions().to_payload()

    def _job_id_for_task(self, job: Union[Job, str], expected_task: str) -> str:
        if isinstance(job, str):
            return job
        if job.task != expected_task:
            raise InvalidRequestError(
                f"Job {job.job_id} is for task {job.task}, not {expected_task}."
            )
        job_is_ocr_model = is_ocr_model(job.model)
        if expected_task == "ocr" and not job_is_ocr_model:
            raise InvalidRequestError(
                f"Job {job.job_id} model {job.model} is not valid for OCR."
            )
        if expected_task == "document_parsing" and not is_document_parsing_model(
            job.model
        ):
            raise InvalidRequestError(
                f"Job {job.job_id} model {job.model} is not valid for document parsing."
            )
        return job.job_id

    def _download_resource(self, url: str, destination: Path, overwrite: bool) -> Path:
        parent = destination.parent
        if not parent.exists():
            raise FileNotFoundError(str(parent))
        if destination.exists() and not overwrite:
            raise InvalidRequestError(
                f"Destination already exists: {destination}. Pass overwrite=True to replace it."
            )
        try:
            response = requests.get(url, timeout=self._http._timeout)
        except requests.Timeout as e:
            raise RequestTimeoutError(f"Request timed out: {e}") from e
        except requests.ConnectionError as e:
            raise NetworkError(f"Connection failed: {e}") from e
        except requests.RequestException as e:
            raise NetworkError(f"Resource download failed: {e}") from e

        status_code = getattr(response, "status_code", 200)
        if not isinstance(status_code, int):
            status_code = 200
        if not 200 <= status_code < 300:
            raise APIError(status_code, _extract_api_message(response))
        destination.write_bytes(response.content)
        return destination

    def _iter_result_resources(
        self, result: Union[OCRResult, DocParsingResult]
    ) -> list[tuple[str, str]]:
        resources = []
        if isinstance(result, OCRResult):
            for index, page in enumerate(result.pages, start=1):
                if page.ocr_image_url:
                    resources.append(
                        (
                            self._resource_filename(
                                preferred_name=None,
                                url=page.ocr_image_url,
                                fallback_name=f"ocr_page_{index}",
                            ),
                            page.ocr_image_url,
                        )
                    )
            return resources

        if isinstance(result, DocParsingResult):
            for page in result.pages:
                for name, url in page.markdown_images.items():
                    resources.append(
                        (
                            self._resource_filename(
                                preferred_name=name,
                                url=url,
                                fallback_name=None,
                            ),
                            url,
                        )
                    )
                for name, url in page.output_images.items():
                    resources.append(
                        (
                            self._resource_filename(
                                preferred_name=name,
                                url=url,
                                fallback_name=None,
                            ),
                            url,
                        )
                    )
            return resources

        raise InvalidRequestError(
            "resource must be a URL, OCRResult, or DocParsingResult."
        )

    def _ocr_model(self, model: Union[Model, str]) -> Model:
        try:
            resolved = model if isinstance(model, Model) else Model(model)
        except ValueError as e:
            raise InvalidRequestError(f"Unsupported model: {model}") from e
        if not is_ocr_model(resolved):
            raise InvalidRequestError(
                f"{resolved.value} is not an OCR model and cannot be used for OCR."
            )
        return resolved

    def _document_model(self, model: Union[Model, str]) -> Model:
        try:
            resolved = model if isinstance(model, Model) else Model(model)
        except ValueError as e:
            raise InvalidRequestError(f"Unsupported model: {model}") from e
        if is_ocr_model(resolved):
            raise InvalidRequestError(
                f"{resolved.value} is an OCR model and cannot be used for document parsing."
            )
        if not is_document_parsing_model(resolved):
            raise InvalidRequestError(
                f"Unsupported document parsing model: {resolved.value}"
            )
        return resolved

    def _resource_filename(
        self,
        preferred_name: Optional[str],
        url: str,
        fallback_name: Optional[str],
    ) -> str:
        # Duplicate resource names intentionally flow through overwrite handling:
        # without overwrite=True the second write raises instead of replacing.
        candidates = [
            preferred_name,
            Path(urlparse(url).path).name,
            fallback_name,
        ]
        for candidate in candidates:
            if candidate:
                return self._sanitize_resource_filename(candidate)
        raise InvalidRequestError("Resource filename cannot be empty.")

    def _sanitize_resource_filename(self, filename: str) -> str:
        if not filename:
            raise InvalidRequestError("Resource filename cannot be empty.")
        path = Path(filename)
        if (
            path.is_absolute()
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or ".." in path.parts
        ):
            raise InvalidRequestError(f"Unsafe resource filename: {filename}")
        return filename
