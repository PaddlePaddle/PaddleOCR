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

"""Build float32 NCHW ``.npy`` tensors for static ONNX quantize (``--calib-data-dir``).

Uses :func:`paddlex.create_predictor` with ``engine="onnxruntime"`` and the same
preprocess as inference (PaddleX ``TextDetRunnerPredictor`` / ``TextRecRunnerPredictor``;
see ``paddlex/inference/models/text_detection|text_recognition/predictor.py``).

Requires a working **PaddleX** install (e.g. ``pip install -e /path/to/PaddleX`` or
``PYTHONPATH`` to the repo root), plus PyYAML, OpenCV, NumPy, and onnxruntime.

Examples (from the iOS demo project root):

  # Detection model: one .npy per image, shapes follow DetResize (e.g. long side 960)
  python3 Scripts/build_onnx_calib_npy.py --task det \\
    --model-dir PaddleOCRDemo/Models/det --image-dir /path/to/images --output-dir /tmp/calib_det

  # Recognition model: one .npy per image (entire page resized/padded; use text-line crops for stricter rec calib)
  python3 Scripts/build_onnx_calib_npy.py --task rec \\
    --model-dir PaddleOCRDemo/Models/rec --image-dir /path/to/images --output-dir /tmp/calib_rec
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _read_model_name(model_dir: Path) -> str:
    try:
        import yaml
    except ImportError as e:
        _die(f"PyYAML is required: {e}")
    p = model_dir / "inference.yml"
    if not p.is_file():
        _die(f"missing {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    name = (data or {}).get("Global", {}).get("model_name")
    if not name:
        _die("Global.model_name not found in inference.yml")
    return str(name)


def _collect_images(image_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    out: list[Path] = []
    for p in sorted(image_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    if not out:
        _die(f"no images found under {image_dir}")
    return out


def _safe_stem(p: Path, idx: int) -> str:
    s = p.stem
    s = re.sub(r"[^\w\-.]+", "_", s)[:80]
    return f"{idx:04d}_{s}" if s else f"{idx:04d}"


def _tensor_det(pred, image_path: str) -> np.ndarray:
    """Mirror TextDetRunnerPredictor.process() preprocess, return (1,C,H,W) float32."""
    batches = list(pred.batch_sampler([image_path]))
    b = batches[0]
    batch_raw_imgs = pred.pre_tfs["Read"](imgs=b.instances)
    batch_imgs, _ = pred.pre_tfs["Resize"](
        imgs=batch_raw_imgs,
        limit_side_len=pred.limit_side_len,
        limit_type=pred.limit_type,
        max_side_limit=pred.max_side_limit,
    )
    batch_imgs = pred.pre_tfs["Normalize"](imgs=batch_imgs)
    batch_imgs = pred.pre_tfs["ToCHW"](imgs=batch_imgs)
    x = pred.pre_tfs["ToBatch"](imgs=batch_imgs)
    if not x:
        _die("ToBatch returned empty (det)")
    return np.ascontiguousarray(x[0], dtype=np.float32)


def _tensor_rec(pred, image_path: str) -> np.ndarray:
    from paddlex.inference.models.text_recognition.processors import (
        validate_text_rec_image_array,
    )

    batches = list(pred.batch_sampler([image_path]))
    b = batches[0]
    batch_raw_imgs = pred.pre_tfs["Read"](imgs=b.instances)
    for i, img in enumerate(batch_raw_imgs):
        validate_text_rec_image_array(img, index=i)
    batch_imgs = pred.pre_tfs["ReisizeNorm"](imgs=batch_raw_imgs)
    x = pred.pre_tfs["ToBatch"](imgs=batch_imgs)
    if not x:
        _die("ToBatch returned empty (rec)")
    return np.ascontiguousarray(x[0], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build float32 NCHW .npy files for static ONNX quantize (calib-data-dir), "
        "using PaddleX ONNX preprocess."
    )
    ap.add_argument(
        "--image-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory of input images (default: current working directory).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for .npy files (created if missing).",
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Model directory with inference.yml and inference.onnx (det or rec).",
    )
    ap.add_argument(
        "--task",
        choices=("det", "rec"),
        required=True,
        help="Model role: det = text detection, rec = text recognition.",
    )
    ap.add_argument("--device", default="cpu", help="Device string, e.g. cpu or gpu:0.")
    args = ap.parse_args()

    if not args.image_dir.is_dir():
        _die(f"not a directory: {args.image_dir}")
    if not (args.model_dir / "inference.onnx").is_file():
        _die(f"missing {args.model_dir / 'inference.onnx'}")

    model_name = _read_model_name(args.model_dir)
    if args.task == "det" and "_rec" in model_name and "_det" not in model_name:
        print(
            "warning: --task det but model_name looks like recognition; check --model-dir",
            file=sys.stderr,
        )
    if args.task == "rec" and "_det" in model_name and "_rec" not in model_name:
        print(
            "warning: --task rec but model_name looks like detection; check --model-dir",
            file=sys.stderr,
        )

    try:
        from paddlex import create_predictor
    except ImportError as e:
        _die(
            f"import paddlex failed ({e}). Install PaddleX, e.g. "
            f"pip install -e <PaddleX_repo>  or set PYTHONPATH to the repo root that contains `paddlex`."
        )

    os.environ.setdefault("PADDLE_IS_COMPILED_WITH_CUDA", "0")
    try:
        pred = create_predictor(
            model_name,
            model_dir=str(args.model_dir.resolve()),
            device=args.device,
            engine="onnxruntime",
            batch_size=1,
        )
    except Exception as e:
        _die(f"create_predictor failed: {e}")

    if args.task == "det" and "Resize" not in pred.pre_tfs:
        _die("det predictor has no 'Resize' op; is this a detection model?")
    if args.task == "rec" and "ReisizeNorm" not in pred.pre_tfs:
        _die("rec predictor has no 'ReisizeNorm' op; is this a recognition model?")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    images = _collect_images(args.image_dir)
    for idx, path in enumerate(images):
        pstr = str(path.resolve())
        try:
            if args.task == "det":
                arr = _tensor_det(pred, pstr)
            else:
                arr = _tensor_rec(pred, pstr)
        except Exception as e:
            _die(f"preprocess failed for {path}: {e}")
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        name = _safe_stem(path, idx) + ".npy"
        out_path = out_dir / name
        np.save(str(out_path), arr)
        print(f"wrote {out_path}  shape={arr.shape}  dtype={arr.dtype}")

    print(f"done. {len(images)} files -> {out_dir}")


if __name__ == "__main__":
    main()
