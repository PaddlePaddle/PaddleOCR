#!/usr/bin/env python3
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

"""Generate reference OCR output from PaddleX for validation against iOS pipeline.

Runs PaddleOCR (backed by PaddleX) on a set of test images and exports
per-image JSON files containing detection polygons, recognized text, and
confidence scores. These reference files are then compared against the
iOS pipeline output by validate.py.

Usage:
    python3 generate_reference.py [--images-dir DIR] [--output-dir DIR]

Requires: paddleocr >= 3.4.0, paddlex >= 3.4.0
"""

import argparse
import json
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def generate_reference(images_dir: str, output_dir: str) -> None:
    """Run PaddleX OCR on each image and save reference JSON.

    Args:
        images_dir: Path to directory containing test images.
        output_dir: Path to directory where reference JSON files will be written.
    """
    # Import inside function so the module can be imported without paddleocr
    # installed (useful for validate.py which shares this directory).
    from paddleocr import PaddleOCR

    # Initialize with PP-OCRv5 mobile models -- must match the models
    # bundled in the iOS demo exactly.
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_detection_model_dir="Models/det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        text_recognition_model_dir="Models/rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        f for f in images_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print(f"Error: No images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(image_files)} images...")

    for img_file in image_files:
        print(f"  {img_file.name}...", end=" ", flush=True)
        results = list(ocr.predict(str(img_file)))

        boxes = _extract_boxes(results)

        reference = {
            "image": img_file.name,
            "box_count": len(boxes),
            "boxes": boxes,
        }

        out_file = output_path / f"{img_file.stem}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(reference, f, indent=2, ensure_ascii=False)

        print(f"{len(boxes)} boxes")

    print(f"\nReference files written to {output_dir}")


def _extract_boxes(results: list) -> list:
    """Extract box data from PaddleOCR result objects.

    PaddleOCR 3.x predict() returns OCRResult objects with attributes:
        - dt_polys: list of polygons (numpy arrays or lists of [x, y] pairs)
        - rec_texts: list of recognized text strings
        - rec_scores: list of confidence scores

    Returns:
        List of dicts with "polygon", "text", "confidence" keys.
    """
    boxes = []
    for result in results:
        if hasattr(result, "rec_texts"):
            # PaddleOCR 3.x API (PaddleX-backed)
            for i in range(len(result.rec_texts)):
                poly = result.dt_polys[i]
                if hasattr(poly, "tolist"):
                    poly = poly.tolist()
                boxes.append(
                    {
                        "polygon": [[int(p[0]), int(p[1])] for p in poly],
                        "text": result.rec_texts[i],
                        "confidence": float(result.rec_scores[i]),
                    }
                )
        elif hasattr(result, "__getitem__"):
            # Legacy API fallback: list of dicts with dt_polys/rec_text/rec_score
            for item in result:
                if isinstance(item, dict):
                    poly = item.get("dt_polys", item.get("points", []))
                    if hasattr(poly, "tolist"):
                        poly = poly.tolist()
                    boxes.append(
                        {
                            "polygon": [[int(p[0]), int(p[1])] for p in poly],
                            "text": item.get("rec_text", ""),
                            "confidence": float(item.get("rec_score", 0.0)),
                        }
                    )
    return boxes


def main():
    parser = argparse.ArgumentParser(
        description="Generate PaddleX OCR reference output for iOS validation"
    )
    parser.add_argument(
        "--images-dir",
        default="test_images",
        help=("Directory containing test images " "(default: test_images)"),
    )
    parser.add_argument(
        "--output-dir",
        default="reference",
        help=("Directory for reference JSON output " "(default: reference)"),
    )
    args = parser.parse_args()
    generate_reference(args.images_dir, args.output_dir)


if __name__ == "__main__":
    main()
