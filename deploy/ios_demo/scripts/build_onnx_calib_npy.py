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

"""Build float32 NCHW ``.npy`` tensors for static ONNX quantization.

Note:
    Uses undocumented PaddleX predictor internals (e.g. ``pre_tfs``). May break when
    upgrading PaddleX.

Examples:

  # Detection model
  python3 scripts/build_onnx_calib_npy.py --task det \\
    --model-dir PaddleOCRDemo/Models/det --image-dir /path/to/images --output-dir calib_det

  # Recognition model (one tensor per full image)
  python3 scripts/build_onnx_calib_npy.py --task rec \\
    --model-dir PaddleOCRDemo/Models/rec --image-dir /path/to/images --output-dir calib_rec

  # Recognition: full-page images — run det to crop line boxes, one tensor per crop
  python3 scripts/build_onnx_calib_npy.py --task rec \\
    --model-dir PaddleOCRDemo/Models/rec --det-model-dir PaddleOCRDemo/Models/det \\
    --image-dir /path/to/pages --output-dir calib_rec_crops
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from paddlex import create_predictor

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from utils import die


def _read_model_name(model_dir: Path) -> str:
    try:
        import yaml
    except ImportError as e:
        die(f"PyYAML is required: {e}")
    p = model_dir / "inference.yml"
    if not p.is_file():
        die(f"missing {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    name = (data or {}).get("Global", {}).get("model_name")
    if not name:
        die("Global.model_name not found in inference.yml")
    return str(name)


def _collect_images(image_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    out: list[Path] = []
    for p in sorted(image_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    if not out:
        die(f"no images found under {image_dir}")
    return out


def _safe_stem(p: Path, idx: int) -> str:
    s = p.stem
    s = re.sub(r"[^\w\-.]+", "_", s)[:80]
    return f"{idx:04d}_{s}" if s else f"{idx:04d}"


def _iter_text_boxes(dt_polys: Any) -> Iterator[np.ndarray]:
    """Turn ``dt_polys`` from a TextDetResult into a sequence of (N,2) point arrays."""
    if dt_polys is None:
        return
    if isinstance(dt_polys, np.ndarray):
        if dt_polys.size == 0:
            return
        if dt_polys.ndim == 2 and dt_polys.shape[0] >= 3:
            yield np.array(dt_polys, dtype=np.float32, copy=True)
        elif dt_polys.ndim == 3:
            for i in range(dt_polys.shape[0]):
                box = np.array(dt_polys[i], dtype=np.float32, copy=True)
                if box.shape[0] >= 3:
                    yield box
        return
    for box in dt_polys:
        b = np.array(box, dtype=np.float32, copy=True)
        if b.size >= 6 and b.ndim == 2 and b.shape[0] >= 3:
            yield b


def _minarea_rect_rotate_crop(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Min-area rect crop aligned with PP-OCR (see PaddleX CropByPolys, quad).
    *img* is RGB (same as TextDet Read output).
    """
    tmp_box = np.array(points, dtype=np.float32, copy=True)
    bounding_box = cv2.minAreaRect(tmp_box.astype(np.int32))
    pts = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    if pts[1][1] > pts[0][1]:
        index_a, index_d = 0, 1
    else:
        index_a, index_d = 1, 0
    if pts[3][1] > pts[2][1]:
        index_b, index_c = 2, 3
    else:
        index_b, index_c = 3, 2
    box = [pts[index_a], pts[index_b], pts[index_c], pts[index_d]]
    pts4 = [np.array(p, dtype=np.float32) for p in box]
    assert len(pts4) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(pts4[0] - pts4[1]),
            np.linalg.norm(pts4[2] - pts4[3]),
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(pts4[0] - pts4[3]),
            np.linalg.norm(pts4[1] - pts4[2]),
        )
    )
    if img_crop_width < 1 or img_crop_height < 1:
        raise ValueError("degenerate text crop (zero width/height)")
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    pstack = np.stack(pts4, axis=0)
    m = cv2.getPerspectiveTransform(pstack, pts_std)
    dst = cv2.warpPerspective(
        img,
        m,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dh, dw = dst.shape[0:2]
    if dh * 1.0 / max(dw, 1) >= 1.5:
        dst = np.rot90(dst)
    return dst


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
        die("ToBatch returned empty (det)")
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
        die("ToBatch returned empty (rec)")
    return np.ascontiguousarray(x[0], dtype=np.float32)


def _tensor_rec_from_rgb_ndarray(rec_pred, img_rgb: np.ndarray) -> np.ndarray:
    """Same preprocessing as *path* input, for an RGB array (e.g. det crop on full page)."""
    from paddlex.inference.models.text_recognition.processors import (
        validate_text_rec_image_array,
    )

    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    batch_raw_imgs = rec_pred.pre_tfs["Read"](imgs=[bgr])
    for i, img in enumerate(batch_raw_imgs):
        validate_text_rec_image_array(img, index=i)
    batch_imgs = rec_pred.pre_tfs["ReisizeNorm"](imgs=batch_raw_imgs)
    x = rec_pred.pre_tfs["ToBatch"](imgs=batch_imgs)
    if not x:
        die("ToBatch returned empty (rec, ndarray)")
    return np.ascontiguousarray(x[0], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build float32 NCHW .npy files for static ONNX quantize."
    )
    ap.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory of input images.",
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
        help="Model directory.",
    )
    ap.add_argument(
        "--task",
        choices=("det", "rec"),
        required=True,
        help="Model role: det = text detection, rec = text recognition.",
    )
    ap.add_argument(
        "--det-model-dir",
        type=Path,
        default=None,
        help=(
            "For --task rec only: text-detection model directory. "
            "When set, each full image is run through det; each text box is cropped and one "
            "rec calibration tensor is written per crop (instead of one tensor per file)."
        ),
    )
    ap.add_argument(
        "--max-crops-per-image",
        type=int,
        default=0,
        help="With --det-model-dir, cap how many box crops to use per input image (0 = no cap).",
    )
    ap.add_argument("--device", default="cpu", help="Device string, e.g. cpu or gpu:0.")
    args = ap.parse_args()

    if not args.image_dir.is_dir():
        die(f"not a directory: {args.image_dir}")
    if not (args.model_dir / "inference.onnx").is_file():
        die(f"missing {args.model_dir / 'inference.onnx'}")

    if args.det_model_dir is not None:
        if args.task != "rec":
            die("--det-model-dir is only valid with --task rec")
        if not (args.det_model_dir / "inference.onnx").is_file():
            die(f"missing {args.det_model_dir / 'inference.onnx'}")

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
        pred = create_predictor(
            model_name,
            model_dir=str(args.model_dir.resolve()),
            device=args.device,
            engine="onnxruntime",
            batch_size=1,
        )
    except Exception as e:
        die(f"create_predictor failed: {e}")

    det_pred = None
    det_name = None
    if args.det_model_dir is not None:
        det_name = _read_model_name(args.det_model_dir)
        if "_rec" in det_name and "_det" not in det_name:
            print(
                "warning: --det-model-dir model_name looks like recognition; check path",
                file=sys.stderr,
            )
        try:
            det_pred = create_predictor(
                det_name,
                model_dir=str(args.det_model_dir.resolve()),
                device=args.device,
                engine="onnxruntime",
                batch_size=1,
            )
        except Exception as e:
            die(f"create_predictor (det) failed: {e}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    images = _collect_images(args.image_dir)
    written = 0
    for idx, path in enumerate(images):
        pstr = str(path.resolve())
        if args.task == "det":
            try:
                arr = _tensor_det(pred, pstr)
            except Exception as e:
                die(f"preprocess failed for {path}: {e}")
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            name = _safe_stem(path, idx) + ".npy"
            out_path = out_dir / name
            np.save(str(out_path), arr)
            print(f"wrote {out_path}  shape={arr.shape}  dtype={arr.dtype}")
            written += 1
            continue

        # rec
        if det_pred is None:
            try:
                arr = _tensor_rec(pred, pstr)
            except Exception as e:
                die(f"preprocess failed for {path}: {e}")
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            out_path = out_dir / (_safe_stem(path, idx) + ".npy")
            np.save(str(out_path), arr)
            print(f"wrote {out_path}  shape={arr.shape}  dtype={arr.dtype}")
            written += 1
            continue

        # rec + det: crops from full page
        try:
            det_res = next(det_pred(pstr))  # type: ignore[union-attr]
        except Exception as e:
            die(f"detection failed for {path}: {e}")
        polys = list(_iter_text_boxes(det_res["dt_polys"]))
        if not polys:
            print(
                f"warning: no text boxes for {path}; skipping (no .npy written)",
                file=sys.stderr,
            )
            continue
        if args.max_crops_per_image > 0:
            polys = polys[: args.max_crops_per_image]
        full_rgb = det_res["input_img"]
        if isinstance(full_rgb, (list, tuple)) and len(full_rgb) == 1:
            full_rgb = full_rgb[0]
        for box_i, quad in enumerate(polys):
            try:
                line_rgb = _minarea_rect_rotate_crop(full_rgb, quad)
            except Exception as e:
                print(
                    f"warning: skip box {box_i} in {path}: {e}",
                    file=sys.stderr,
                )
                continue
            try:
                arr = _tensor_rec_from_rgb_ndarray(pred, line_rgb)
            except Exception as e:
                die(f"rec preprocess failed for {path} box {box_i}: {e}")
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            out_path = out_dir / (f"{_safe_stem(path, idx)}_box{box_i:03d}.npy")
            np.save(str(out_path), arr)
            print(f"wrote {out_path}  shape={arr.shape}  dtype={arr.dtype}")
            written += 1

    print(f"done. {written} .npy file(s) -> {out_dir}")


if __name__ == "__main__":
    main()
