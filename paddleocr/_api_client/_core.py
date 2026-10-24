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

from typing import Optional, Union

from .errors import (
    APIError,
    AuthError,
    InvalidRequestError,
    RateLimitError,
    ResponseFormatError,
    ServiceUnavailableError,
)
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
from .results import BatchStatus, Job, JobStatus, Progress


def validate_input_source(file_url: Optional[str], file_path: Optional[str]) -> None:
    if not file_url and not file_path:
        raise InvalidRequestError("Either file_url or file_path is required.")
    if file_url and file_path:
        raise InvalidRequestError("file_url and file_path are mutually exclusive.")


def default_payload(model: Model) -> dict:
    if is_ocr_model(model):
        return OCROptions().to_payload()
    return resolve_document_options(model, None).to_payload()


def resolve_ocr_model(model: Union[Model, str]) -> Model:
    resolved = resolve_model(model)
    if not is_ocr_model(resolved):
        raise InvalidRequestError(f"Unsupported OCR model: {model}")
    return resolved


def resolve_document_model(model: Union[Model, str]) -> Model:
    resolved = resolve_model(model)
    if not is_document_parsing_model(resolved):
        raise InvalidRequestError(f"Unsupported document parsing model: {model}")
    return resolved


def resolve_model(model: Union[Model, str]) -> Model:
    if isinstance(model, Model):
        return model
    try:
        return Model(model)
    except ValueError as e:
        raise InvalidRequestError(f"Unsupported model: {model}") from e


def resolve_document_options(
    model: Model, options: Optional[DocParsingOptions]
) -> DocParsingOptions:
    if options is not None:
        if model == Model.PP_STRUCTURE_V3 and not isinstance(
            options, PPStructureV3Options
        ):
            raise InvalidRequestError("PP-StructureV3 requires PPStructureV3Options.")
        if is_vl_model(model) and not isinstance(options, PaddleOCRVLOptions):
            raise InvalidRequestError("PaddleOCR-VL models require PaddleOCRVLOptions.")
        return options
    if model == Model.PP_STRUCTURE_V3:
        return PPStructureV3Options()
    return PaddleOCRVLOptions()


def job_id_for_task(job: Union[Job, str], task: str) -> str:
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


def extract_api_message_from_payload(payload: dict) -> Optional[str]:
    for key in ("msg", "errorMsg", "message"):
        value = payload.get(key)
        if value:
            return str(value)
    data = payload.get("data")
    if isinstance(data, dict):
        value = data.get("errorMsg")
        if value:
            return str(value)
    return None


def validate_state(data: dict) -> str:
    state = data.get("state")
    if state not in {"pending", "running", "done", "failed"}:
        raise ResponseFormatError(f"Unknown or missing job state: {state}")
    return state


def job_status_from_data(job_id: str, data: dict) -> JobStatus:
    state = validate_state(data)
    progress = None
    ep = data.get("extractProgress")
    if ep:
        if not isinstance(ep, dict):
            raise ResponseFormatError("'extractProgress' must be an object.")
        progress = Progress(
            total_pages=ep.get("totalPages", 0),
            extracted_pages=ep.get("extractedPages", 0),
            start_time=ep.get("startTime"),
            end_time=ep.get("endTime"),
        )
    return JobStatus(
        job_id=job_id,
        state=state,
        progress=progress,
        result=data.get("resultUrl"),
        error_msg=data.get("errorMsg"),
    )


def raise_for_status(status_code: int, msg: str) -> None:
    if 200 <= status_code < 300:
        return
    if status_code in (401, 403):
        raise AuthError(f"Authentication failed: {msg}")
    if status_code == 400:
        raise InvalidRequestError(f"Bad request: {msg}")
    if status_code == 429:
        raise RateLimitError(f"Rate limit exceeded: {msg}")
    if status_code in (503, 504):
        raise ServiceUnavailableError(status_code, f"Service unavailable: {msg}")
    raise APIError(status_code, msg)


def unwrap_api_response(payload: dict, status_code: int) -> dict:
    if not isinstance(payload, dict):
        raise ResponseFormatError("Response body must be a JSON object.")
    code = payload.get("code", 0)
    if code not in (0, None):
        raise APIError(status_code, extract_api_message_from_payload(payload) or "")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ResponseFormatError("Response JSON must contain object field 'data'.")
    return data


def extract_job_id(data: dict) -> str:
    job_id = data.get("jobId")
    if not isinstance(job_id, str) or not job_id:
        raise ResponseFormatError(
            "Response data must contain non-empty string 'jobId'."
        )
    return job_id


def validate_result_json_url(data: dict) -> str:
    result_url = data.get("resultUrl")
    if not isinstance(result_url, dict):
        raise ResponseFormatError("Done job response must contain object 'resultUrl'.")
    json_url = result_url.get("jsonUrl")
    if not isinstance(json_url, str) or not json_url:
        raise ResponseFormatError(
            "Done job response resultUrl must contain non-empty string 'jsonUrl'."
        )
    return json_url


def parse_batch_status(batch_id: str, data: dict) -> BatchStatus:
    result = data.get("extractResult")
    if not isinstance(result, list):
        raise ResponseFormatError(
            "Batch response data must contain list 'extractResult'."
        )
    jobs = []
    for item in result:
        if not isinstance(item, dict):
            raise ResponseFormatError("Batch extractResult items must be objects.")
        job_id = item.get("jobId")
        if not isinstance(job_id, str) or not job_id:
            raise ResponseFormatError(
                "Batch extractResult items must contain non-empty string 'jobId'."
            )
        jobs.append(job_status_from_data(job_id, item))
    return BatchStatus(batch_id=batch_id, jobs=jobs)
