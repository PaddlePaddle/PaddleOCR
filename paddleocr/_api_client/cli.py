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

import json
import sys

from .client import APIClient
from .models import DocParsingOptions, Model, OCROptions


def register_api_command(subparsers):
    """Register the 'api' subcommand into paddleocr CLI."""
    subparser = subparsers.add_parser(
        "api",
        help="Call PaddleOCR cloud API for OCR or document parsing",
    )
    subparser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["ocr", "doc_parsing"],
        help="Task type: ocr or doc_parsing",
    )
    subparser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[m.value for m in Model],
        help="Model name. Defaults to PP-OCRv5 for ocr task.",
    )
    subparser.add_argument(
        "--file_url",
        type=str,
        default=None,
        help="URL of the file to process",
    )
    subparser.add_argument(
        "--file_path",
        type=str,
        default=None,
        help="Local file path to process",
    )
    subparser.add_argument(
        "--token",
        type=str,
        default=None,
        help="API token (or set PADDLE_OCR_TOKEN env variable)",
    )
    subparser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (prints to stdout if omitted)",
    )
    subparser.add_argument(
        "--use_doc_orientation_classify",
        action="store_true",
        help="Enable document orientation classification",
    )
    subparser.add_argument(
        "--use_doc_unwarping",
        action="store_true",
        help="Enable document unwarping",
    )
    subparser.add_argument(
        "--use_textline_orientation",
        action="store_true",
        help="Enable textline orientation detection (OCR only)",
    )
    subparser.add_argument(
        "--use_chart_recognition",
        action="store_true",
        help="Enable chart recognition (doc_parsing only)",
    )
    subparser.set_defaults(executor=_execute_api)


def _execute_api(args):
    kwargs = {}
    if args.token:
        kwargs["token"] = args.token

    try:
        client = APIClient(**kwargs)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        model = _resolve_model(args.model) if args.model else None

        if args.model_type == "ocr":
            options = OCROptions(
                use_doc_orientation_classify=args.use_doc_orientation_classify,
                use_doc_unwarping=args.use_doc_unwarping,
                use_textline_orientation=args.use_textline_orientation,
            )
            result = client.ocr(
                file_url=args.file_url,
                file_path=args.file_path,
                options=options,
            )
            output = _ocr_result_to_dict(result)
        else:
            if model is None:
                model = Model.PP_STRUCTURE_V3
            options = DocParsingOptions(
                use_doc_orientation_classify=args.use_doc_orientation_classify,
                use_doc_unwarping=args.use_doc_unwarping,
                use_chart_recognition=args.use_chart_recognition,
            )
            result = client.doc_parsing(
                model=model,
                file_url=args.file_url,
                file_path=args.file_path,
                options=options,
            )
            output = _doc_parsing_result_to_dict(result)

        json_str = json.dumps(output, ensure_ascii=False, indent=2)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"Result saved to: {args.output}")
        else:
            print(json_str)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        client.close()


def _resolve_model(model_str: str) -> Model:
    try:
        return Model(model_str)
    except ValueError:
        print(
            f"Error: Unknown model '{model_str}'. "
            f"Choose from: {', '.join(m.value for m in Model)}",
            file=sys.stderr,
        )
        sys.exit(1)


def _ocr_result_to_dict(result) -> dict:
    return {
        "jobId": result.job_id,
        "pages": [
            {
                "prunedResult": page.pruned_result,
                "ocrImageUrl": page.ocr_image_url,
            }
            for page in result.pages
        ],
    }


def _doc_parsing_result_to_dict(result) -> dict:
    return {
        "jobId": result.job_id,
        "pages": [
            {
                "markdownText": page.markdown_text,
                "markdownImages": page.markdown_images,
                "outputImages": page.output_images,
            }
            for page in result.pages
        ],
    }
