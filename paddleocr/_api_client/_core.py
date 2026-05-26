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

from .errors import InvalidRequestError
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
from .results import Job


def validate_input_source(file_url: Optional[str], file_path: Optional[str]) -> None:
    if not file_url and not file_path:
        raise InvalidRequestError("Either file_url or file_path is required.")
    if file_url and file_path:
        raise InvalidRequestError("file_url and file_path are mutually exclusive.")


def default_payload(model: Model) -> dict:
    if model == Model.PP_OCRV5:
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
