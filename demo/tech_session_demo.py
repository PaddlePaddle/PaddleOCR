#!/usr/bin/env python3
"""Short PaddleOCR demo for a technical session.

This script runs OCR against a sample invoice-style image and prints the
recognized text. It also saves an annotated result image and JSON output to a
folder so the audience can see the end-to-end workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_IMAGE = (
    Path(__file__).resolve().parents[1]
    / "deploy"
    / "ios_demo"
    / "PaddleOCRDemo"
    / "Resources"
    / "SampleImages"
    / "general_ocr_002.jpg"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a short PaddleOCR demo on a sample document image."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Path to the input document image (default: shipped demo image).",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="OCR language, for example: en, ch, or a multilingual language code.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_output"),
        help="Directory for OCR JSON and visualized output files.",
    )
    parser.add_argument(
        "--no-draw",
        action="store_true",
        help="Skip saving the annotated image output.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=10,
        help="Maximum number of recognized lines to print from the result.",
    )
    return parser


def run_demo(args: argparse.Namespace) -> None:
    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise SystemExit(f"Input image not found: {image_path}")

    try:
        from paddleocr import PaddleOCR
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PaddleOCR dependencies are not installed. Run:\n"
            "  pip install -r requirements.txt\n"
            "or install the project in editable mode."
        ) from exc

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running demo on:", image_path)
    print("Language:", args.lang)
    print("Saving output to:", output_dir)

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=args.lang,
        engine="paddle",
    )

    result = ocr.predict(str(image_path))
    if not result:
        raise SystemExit("OCR returned no results for the input image.")

    print("\nRecognized text:")
    for index, res in enumerate(result, start=1):
        print(f"\nPage {index}:")
        res.print()
        if index >= args.max_lines:
            break

    if not args.no_draw:
        if hasattr(result[0], "save_to_img"):
            result[0].save_to_img(str(output_dir))
            print(f"\nSaved annotated image to: {output_dir}")
        else:
            print("\nAnnotated image export is not available in this PaddleOCR build.")

    if hasattr(result[0], "save_to_json"):
        result[0].save_to_json(str(output_dir))
        print(f"Saved JSON result to: {output_dir}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
