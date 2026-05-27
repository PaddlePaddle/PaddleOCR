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

import argparse
import json
import sys

from .client import PaddleOCRClient
from .models import (
    Model,
    OCROptions,
    PaddleOCRVLOptions,
    PPStructureV3Options,
    is_ocr_model,
    is_vl_model,
)


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
        "--base_url",
        type=str,
        default=None,
        help="Base URL of the PaddleOCR API service",
    )
    subparser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Access token (or set PADDLEOCR_ACCESS_TOKEN env variable)",
    )
    subparser.add_argument(
        "--client_platform",
        type=str,
        default=None,
        help="Value for the Client-Platform request header",
    )
    subparser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (prints to stdout if omitted)",
    )
    subparser.add_argument(
        "--request_timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for one HTTP request",
    )
    subparser.add_argument(
        "--poll_timeout",
        type=float,
        default=600.0,
        help="Total timeout in seconds while waiting for the remote job",
    )
    subparser.add_argument(
        "--save_resources",
        type=str,
        default=None,
        help="Directory for saving resources referenced by the result",
    )
    subparser.add_argument(
        "--overwrite_resources",
        action="store_true",
        help="Overwrite existing files when saving resources",
    )
    subparser.add_argument(
        "--page_ranges",
        type=str,
        default=None,
        help='Page ranges to parse, for example "2,4-6"',
    )
    subparser.add_argument(
        "--batch_id",
        type=str,
        default=None,
        help="Optional batch identifier for querying related jobs",
    )
    # --- Preprocessing ---
    subparser.add_argument(
        "--use_doc_orientation_classify",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable document orientation classification",
    )
    subparser.add_argument(
        "--use_doc_unwarping",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable document unwarping",
    )
    # --- Text detection ---
    subparser.add_argument(
        "--use_textline_orientation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable textline orientation detection (OCR only)",
    )
    subparser.add_argument(
        "--text_det_limit_side_len",
        type=int,
        default=None,
        help="Image side length limit for text detection",
    )
    subparser.add_argument(
        "--text_det_limit_type",
        type=str,
        default=None,
        choices=["min", "max"],
        help="Side length limit type: min or max",
    )
    # --- Text recognition ---
    subparser.add_argument(
        "--text_rec_score_thresh",
        type=float,
        default=None,
        help="Score threshold for text recognition results",
    )
    # --- Layout and feature toggles (doc_parsing only) ---
    subparser.add_argument(
        "--use_layout_detection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable layout detection (doc_parsing only)",
    )
    subparser.add_argument(
        "--use_seal_recognition",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable seal recognition (doc_parsing only)",
    )
    subparser.add_argument(
        "--use_table_recognition",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable table recognition (PP-StructureV3 only)",
    )
    subparser.add_argument(
        "--use_formula_recognition",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable formula recognition (PP-StructureV3 only)",
    )
    subparser.add_argument(
        "--use_chart_recognition",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable chart recognition (doc_parsing only)",
    )
    # --- Output ---
    subparser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable result visualization images",
    )
    subparser.add_argument(
        "--prettify_markdown",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable markdown prettification (doc_parsing only)",
    )
    subparser.set_defaults(executor=_execute_api)


def _execute_api(args):
    kwargs = {}
    if args.token:
        kwargs["token"] = args.token
    if args.base_url:
        kwargs["base_url"] = args.base_url
    kwargs["request_timeout"] = args.request_timeout
    kwargs["poll_timeout"] = args.poll_timeout
    if args.client_platform:
        kwargs["client_platform"] = args.client_platform

    try:
        client = PaddleOCRClient(**kwargs)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        model = _resolve_model(args.model) if args.model else None

        if args.model_type == "ocr":
            if model is not None and not is_ocr_model(model):
                print(
                    f"Error: OCR task does not support {model.value}.",
                    file=sys.stderr,
                )
                sys.exit(2)
            options = OCROptions(
                use_doc_orientation_classify=args.use_doc_orientation_classify,
                use_doc_unwarping=args.use_doc_unwarping,
                use_textline_orientation=args.use_textline_orientation,
                text_det_limit_side_len=args.text_det_limit_side_len,
                text_det_limit_type=args.text_det_limit_type,
                text_rec_score_thresh=args.text_rec_score_thresh,
                visualize=args.visualize,
            )
            result = client.ocr(
                file_url=args.file_url,
                file_path=args.file_path,
                options=options,
                page_ranges=args.page_ranges,
                batch_id=args.batch_id,
                model=model or Model.PP_OCRV5,
            )
            output = _ocr_result_to_dict(result)
            save_resources = client.save_ocr_result_resources
        else:
            if model is None:
                model = Model.PADDLE_OCR_VL_16
            if is_vl_model(model):
                options = PaddleOCRVLOptions(
                    use_doc_orientation_classify=args.use_doc_orientation_classify,
                    use_doc_unwarping=args.use_doc_unwarping,
                    use_chart_recognition=args.use_chart_recognition,
                    use_seal_recognition=args.use_seal_recognition,
                    use_layout_detection=args.use_layout_detection,
                    prettify_markdown=args.prettify_markdown,
                    visualize=args.visualize,
                )
            else:
                options = PPStructureV3Options(
                    use_doc_orientation_classify=args.use_doc_orientation_classify,
                    use_doc_unwarping=args.use_doc_unwarping,
                    use_textline_orientation=args.use_textline_orientation,
                    use_chart_recognition=args.use_chart_recognition,
                    use_seal_recognition=args.use_seal_recognition,
                    use_table_recognition=args.use_table_recognition,
                    use_formula_recognition=args.use_formula_recognition,
                    use_layout_detection=args.use_layout_detection,
                    text_det_limit_side_len=args.text_det_limit_side_len,
                    text_det_limit_type=args.text_det_limit_type,
                    text_rec_score_thresh=args.text_rec_score_thresh,
                    prettify_markdown=args.prettify_markdown,
                    visualize=args.visualize,
                )
            result = client.parse_document(
                model=model,
                file_url=args.file_url,
                file_path=args.file_path,
                options=options,
                page_ranges=args.page_ranges,
                batch_id=args.batch_id,
            )
            output = _doc_parsing_result_to_dict(result)
            save_resources = client.save_document_parsing_result_resources

        json_str = json.dumps(output, ensure_ascii=False, indent=2)

        if args.save_resources:
            saved_paths = save_resources(
                result,
                args.save_resources,
                overwrite=args.overwrite_resources,
            )
            print(
                f"Resources saved to: {args.save_resources} ({len(saved_paths)} files)",
                file=sys.stderr,
            )

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
