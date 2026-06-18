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

"""Generate a JSON reference OCR result for comparison with the iOS demo.

Uses ``PaddleOCR`` with ``engine="onnxruntime"`` and no doc-orientation /
unwarping / textline-orientation modules.

Examples:
  python ocr_reference_run.py --image a.png --output ref.json \\
    --device cpu --align-ios-defaults

  # Explicit models path
  python ocr_reference_run.py --image a.png --output ref.json \\
    --ios-models-root /path/to/PaddleOCRDemo/Models --device cpu --align-ios-defaults
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _paddleocr_package_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_ios_models_root() -> Path:
    return Path(__file__).resolve().parent.parent / "PaddleOCRDemo" / "Models"


def _load_yaml_model_name(path: Path) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read inference.yml") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["Global"]["model_name"]


def _numpy_to_python(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_python(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _extract_items(result_obj: Any) -> List[Dict[str, Any]]:
    """Build a list of {polygon, text, score} from an OCR pipeline result."""
    if isinstance(result_obj, dict):
        res = result_obj
    elif hasattr(result_obj, "__getitem__"):
        res = dict(result_obj)
    else:
        raise TypeError(f"Unsupported result type: {type(result_obj)}")
    return _extract_items_from_res_dict(res)


def _extract_items_from_res_dict(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    texts = res.get("rec_texts") or []
    scores = res.get("rec_scores")
    polys = res.get("rec_polys") or res.get("dt_polys") or []
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    items: List[Dict[str, Any]] = []
    n = min(len(texts), len(polys))
    for i in range(n):
        poly = polys[i]
        if hasattr(poly, "tolist"):
            poly = poly.tolist()
        score = float(scores[i]) if scores is not None and i < len(scores) else None
        items.append(
            {
                "polygon": _numpy_to_python(poly),
                "text": str(texts[i]),
                "score": score,
            }
        )
    return items


def _build_ocr(
    *,
    device: str,
    text_detection_model_name: str | None,
    text_recognition_model_name: str | None,
    text_detection_model_dir: str | None,
    text_recognition_model_dir: str | None,
) -> Any:
    from paddleocr import PaddleOCR

    kwargs: Dict[str, Any] = dict(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=device,
        engine="onnxruntime",
    )
    if text_detection_model_name:
        kwargs["text_detection_model_name"] = text_detection_model_name
    if text_recognition_model_name:
        kwargs["text_recognition_model_name"] = text_recognition_model_name
    if text_detection_model_dir:
        kwargs["text_detection_model_dir"] = text_detection_model_dir
    if text_recognition_model_dir:
        kwargs["text_recognition_model_dir"] = text_recognition_model_dir
    return PaddleOCR(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OCR reference JSON for iOS demo comparison"
    )
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--device", default="cpu", help="Device, e.g. cpu or gpu:0")
    parser.add_argument(
        "--ios-models-root",
        type=Path,
        default=_default_ios_models_root(),
        help="Path to models directory",
    )
    parser.add_argument(
        "--align-ios-defaults",
        action="store_true",
        help="Match OCRParameterResolver app-tier defaults (Swift demo)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 2

    root = Path(args.ios_models_root).resolve()
    det_yml = root / "det" / "inference.yml"
    rec_yml = root / "rec" / "inference.yml"
    if not det_yml.is_file() or not rec_yml.is_file():
        print(f"error: expected {det_yml} and {rec_yml}", file=sys.stderr)
        return 2
    text_detection_model_name = _load_yaml_model_name(det_yml)
    text_recognition_model_name = _load_yaml_model_name(rec_yml)
    text_detection_model_dir = str(root / "det")
    text_recognition_model_dir = str(root / "rec")

    ocr = _build_ocr(
        device=args.device,
        text_detection_model_name=text_detection_model_name,
        text_recognition_model_name=text_recognition_model_name,
        text_detection_model_dir=text_detection_model_dir,
        text_recognition_model_dir=text_recognition_model_dir,
    )

    predict_kwargs: Dict[str, Any] = {}
    if args.align_ios_defaults:
        predict_kwargs.update(
            text_det_limit_side_len=64,
            text_det_limit_type="min",
            text_det_thresh=0.3,
            text_det_box_thresh=0.6,
            text_det_unclip_ratio=1.5,
            text_rec_score_thresh=0.0,
        )

    results = ocr.predict(str(image_path), **predict_kwargs)
    if not results:
        print("error: empty predict() result", file=sys.stderr)
        return 3

    items = _extract_items(results[0])
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "source": "paddleocr_reference",
        "engine": "onnxruntime",
        "image": str(image_path),
        "items": items,
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(items)} lines to {out_path}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(_paddleocr_package_root()))
    raise SystemExit(main())
