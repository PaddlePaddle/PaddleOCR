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
from typing import Optional, Union

from ._http import DEFAULT_BASE_URL, HTTPClient
from ._poller import Poller, parse_doc_parsing_result, parse_ocr_result
from .errors import AuthError, InvalidRequestError
from .models import (
    DocParsingOptions,
    Model,
    OCROptions,
    PaddleOCRVLOptions,
    PPStructureV3Options,
    is_document_parsing_model,
    is_ocr_model,
    is_vl_model,
)
from .results import BatchStatus, DocParsingResult, Job, JobStatus, OCRResult


class PaddleOCRClient:
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
        model: Union[Model, str] = Model.PADDLE_OCR_VL_16,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[DocParsingOptions] = None,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> DocParsingResult:
        model = self._document_model(model)
        options = self._document_options(model, options)
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
        model: Union[Model, str] = Model.PADDLE_OCR_VL_16,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        options: Optional[DocParsingOptions] = None,
        page_ranges: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> Job:
        model = self._document_model(model)
        options = self._document_options(model, options)
        job_id = self._submit(
            model, file_url, file_path, options, page_ranges, batch_id
        )
        return Job(job_id=job_id, model=model.value, task="document_parsing")

    def wait_ocr_result(self, job: Union[Job, str]) -> OCRResult:
        job_id = self._job_id_for_task(job, "ocr")
        jsonl_data, _ = self._poller.poll_until_done(job_id)
        return parse_ocr_result(job_id, jsonl_data)

    def wait_document_parsing_result(self, job: Union[Job, str]) -> DocParsingResult:
        job_id = self._job_id_for_task(job, "document_parsing")
        jsonl_data, _ = self._poller.poll_until_done(job_id)
        return parse_doc_parsing_result(job_id, jsonl_data)

    def get_status(self, job_id: str) -> JobStatus:
        return self._poller.get_status(job_id)

    def get_batch_status(self, batch_id: str) -> BatchStatus:
        return self._poller.get_batch_status(batch_id)

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
        if model == Model.PP_OCRV5:
            return OCROptions().to_payload()
        return self._document_options(model, None).to_payload()

    def _ocr_model(self, model: Union[Model, str]) -> Model:
        resolved = self._model(model)
        if not is_ocr_model(resolved):
            raise InvalidRequestError(f"Unsupported OCR model: {model}")
        return resolved

    def _document_model(self, model: Union[Model, str]) -> Model:
        resolved = self._model(model)
        if not is_document_parsing_model(resolved):
            raise InvalidRequestError(f"Unsupported document parsing model: {model}")
        return resolved

    def _model(self, model: Union[Model, str]) -> Model:
        if isinstance(model, Model):
            return model
        try:
            return Model(model)
        except ValueError as e:
            raise InvalidRequestError(f"Unsupported model: {model}") from e

    def _document_options(
        self, model: Model, options: Optional[DocParsingOptions]
    ) -> DocParsingOptions:
        if options is not None:
            if model == Model.PP_STRUCTURE_V3 and not isinstance(
                options, PPStructureV3Options
            ):
                raise InvalidRequestError(
                    "PP-StructureV3 requires PPStructureV3Options."
                )
            if is_vl_model(model) and not isinstance(options, PaddleOCRVLOptions):
                raise InvalidRequestError(
                    "PaddleOCR-VL models require PaddleOCRVLOptions."
                )
            return options
        if model == Model.PP_STRUCTURE_V3:
            return PPStructureV3Options()
        return PaddleOCRVLOptions()

    def _job_id_for_task(self, job: Union[Job, str], task: str) -> str:
        if isinstance(job, str):
            return job
        if job.task != task:
            raise InvalidRequestError(
                f"Job task mismatch: expected {task}, got {job.task}."
            )
        if task == "ocr" and not is_ocr_model(job.model):
            raise InvalidRequestError(f"Job model is not an OCR model: {job.model}.")
        if task == "document_parsing" and not is_document_parsing_model(job.model):
            raise InvalidRequestError(
                f"Job model is not a document parsing model: {job.model}."
            )
        return job.job_id
