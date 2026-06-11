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

import pytest

from paddleocr._api_client._core import (
    default_payload,
    job_id_for_task,
    resolve_document_model,
    resolve_document_options,
    resolve_ocr_model,
    validate_input_source,
)
from paddleocr._api_client.errors import InvalidRequestError
from paddleocr._api_client.models import (
    Model,
    OCROptions,
    PaddleOCRVLOptions,
    PPStructureV3Options,
)
from paddleocr._utils.naming import snake_to_camel
from paddleocr._api_client._poller import parse_doc_parsing_result, parse_ocr_result
from paddleocr._api_client.results import Job


def test_ppstructurev3_options_to_payload_uses_official_camel_case_keys():
    assert snake_to_camel("top_p") == "topP"
    assert snake_to_camel("use_e2e_wired_table_rec_model") == "useE2eWiredTableRecModel"

    options = PPStructureV3Options(
        use_e2e_wired_table_rec_model=True,
        format_block_content=True,
        markdown_ignore_labels=["image"],
        text_det_limit_side_len=1280,
        extra_options={"futureOption": "enabled"},
    )
    assert options.to_payload() == {
        "useE2eWiredTableRecModel": True,
        "formatBlockContent": True,
        "markdownIgnoreLabels": ["image"],
        "textDetLimitSideLen": 1280,
        "futureOption": "enabled",
    }


def test_paddleocr_vl_options_include_service_parameters_and_extra_options():
    options = PaddleOCRVLOptions(
        use_ocr_for_image_block=True,
        format_block_content=True,
        markdown_ignore_labels=["image"],
        vlm_extra_args={"temperature": 0.1},
        return_markdown_images=False,
        output_formats=["docx"],
        extra_options={"futureOption": "enabled"},
    )

    assert options.to_payload() == {
        "useOcrForImageBlock": True,
        "formatBlockContent": True,
        "markdownIgnoreLabels": ["image"],
        "vlmExtraArgs": {"temperature": 0.1},
        "returnMarkdownImages": False,
        "outputFormats": ["docx"],
        "futureOption": "enabled",
    }


def test_result_parsers_preserve_raw_fields_and_data_info():
    ocr_line = {
        "result": {
            "dataInfo": {"numPages": 1},
            "ocrResults": [
                {
                    "prunedResult": {"rec_texts": ["hello"]},
                    "ocrImage": "ocr.png",
                    "docPreprocessingImage": "pre.png",
                    "inputImage": "input.png",
                }
            ],
        }
    }
    ocr_result = parse_ocr_result("job-ocr", [ocr_line])
    assert ocr_result.data_info == {"numPages": 1}
    assert ocr_result.pages[0].doc_preprocessing_image_url == "pre.png"
    assert ocr_result.pages[0].input_image_url == "input.png"
    assert ocr_result.pages[0].raw["ocrImage"] == "ocr.png"

    doc_page = {
        "prunedResult": {"blocks": [{"label": "text", "content": "hello"}]},
        "markdown": {
            "text": "hello",
            "images": {"figure.png": "figure-url"},
            "isStart": True,
        },
        "outputImages": {"page.png": "page-url"},
        "inputImage": "input.png",
        "exports": {"docx": "docx-url"},
    }
    doc_line = {
        "result": {
            "dataInfo": {"numPages": 1},
            "layoutParsingResults": [doc_page],
        }
    }
    doc_result = parse_doc_parsing_result("job-doc", [doc_line])
    assert doc_result.data_info == {"numPages": 1}
    assert doc_result.pages[0].pruned_result == doc_page["prunedResult"]
    assert doc_result.pages[0].markdown == doc_page["markdown"]
    assert doc_result.pages[0].exports == {"docx": "docx-url"}
    assert doc_result.pages[0].raw == doc_page


def test_core_resolves_models_and_default_payloads():
    assert resolve_ocr_model("PP-OCRv5") is Model.PP_OCRV5
    assert resolve_ocr_model("PP-OCRv5-latin") is Model.PP_OCRV5_LATIN
    assert resolve_ocr_model("PP-OCRv6") is Model.PP_OCRV6
    assert resolve_document_model("PaddleOCR-VL-1.6") is Model.PADDLE_OCR_VL_16

    assert default_payload(Model.PP_OCRV5) == OCROptions().to_payload()
    assert default_payload(Model.PP_OCRV5_LATIN) == OCROptions().to_payload()
    assert default_payload(Model.PP_OCRV6) == OCROptions().to_payload()
    assert default_payload(Model.PADDLE_OCR_VL_16) == PaddleOCRVLOptions().to_payload()

    with pytest.raises(InvalidRequestError):
        resolve_ocr_model("PP-StructureV3")

    with pytest.raises(InvalidRequestError):
        resolve_document_model("unknown-model")


def test_core_selects_document_options_by_model():
    assert isinstance(
        resolve_document_options(Model.PP_STRUCTURE_V3, None),
        PPStructureV3Options,
    )
    assert isinstance(
        resolve_document_options(Model.PADDLE_OCR_VL_16, None),
        PaddleOCRVLOptions,
    )

    with pytest.raises(InvalidRequestError):
        resolve_document_options(Model.PP_STRUCTURE_V3, PaddleOCRVLOptions())

    with pytest.raises(InvalidRequestError):
        resolve_document_options(Model.PADDLE_OCR_VL_16, PPStructureV3Options())


def test_core_validates_input_source_and_job_task():
    validate_input_source("https://example.test/file.pdf", None)
    validate_input_source(None, "/tmp/file.pdf")

    with pytest.raises(InvalidRequestError):
        validate_input_source(None, None)

    with pytest.raises(InvalidRequestError):
        validate_input_source("https://example.test/file.pdf", "/tmp/file.pdf")

    job = Job(job_id="job-1", model=Model.PP_OCRV5.value, task="ocr")
    assert job_id_for_task(job, "ocr") == "job-1"
    assert job_id_for_task("job-2", "ocr") == "job-2"

    with pytest.raises(InvalidRequestError):
        job_id_for_task(job, "document_parsing")
