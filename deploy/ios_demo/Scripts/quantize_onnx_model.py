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

"""Quantize a bundled PaddleOCR ONNX model.

Reads ``inference.onnx`` from ``--input-model-dir``, writes quantized ``inference.onnx`` to
``--output-model-dir``. When input and output directories differ, copies ``inference.yml`` from
source to destination after a successful quantize. In-place mode (same directory) only
replaces ``inference.onnx``.

Examples:
  # Dynamic (weights int8); no calibration data
  python quantize_onnx_model.py --input-model-dir ./PaddleOCRDemo/Models/det \\
    --output-model-dir ./out/det_q --mode dynamic

  # Static; one .npy file per calibration sample (float32, model input shape)
  python quantize_onnx_model.py --input-model-dir ./Models/rec \\
    --output-model-dir ./Models/rec_int8 --mode static \\
    --calib-data-dir ./my_calib_npy
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _user_input_names(model_path: Path) -> list[str]:
    import onnx

    m = onnx.load(str(model_path), load_external_data=True)
    init = {t.name for t in m.graph.initializer}
    return [i.name for i in m.graph.input if i.name not in init]


def _build_npy_dir_reader(model_path: Path, data_dir: Path) -> Any:
    from onnxruntime.quantization import CalibrationDataReader

    class NpyDirectoryDataReader(CalibrationDataReader):
        def __init__(self) -> None:
            names = _user_input_names(model_path)
            if len(names) != 1:
                _die(
                    "this tool supports models with exactly one graph input for static "
                    f"quantization; found {len(names)}: {names}"
                )
            self._input_name = names[0]
            files = sorted(data_dir.glob("*.npy"))
            if not files:
                _die(f"no .npy files under {data_dir}")
            self._it = iter(files)

        def get_next(self) -> Any:
            import numpy as np

            try:
                path = next(self._it)
            except StopIteration:
                return None
            arr = np.load(str(path))
            if not isinstance(arr, np.ndarray):
                _die(f"expected ndarray in {path}, got {type(arr)}")
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            return {self._input_name: arr}

    return NpyDirectoryDataReader()


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def _quantize_dynamic(src_onnx: Path, dst_onnx: Path, per_channel: bool) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(src_onnx),
        model_output=str(dst_onnx),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
    )


def _quantize_static(
    src_onnx: Path,
    dst_onnx: Path,
    calib_dir: Path,
    per_channel: bool,
    calibrate_method_name: str,
) -> None:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    reader = _build_npy_dir_reader(src_onnx, calib_dir)
    try:
        method = getattr(CalibrationMethod, calibrate_method_name)
    except AttributeError:
        _die(
            f"unknown calibration method {calibrate_method_name!r}; "
            f"valid names: {', '.join(CalibrationMethod.__members__)}"
        )
    quantize_static(
        model_input=str(src_onnx),
        model_output=str(dst_onnx),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=method,
    )


def _verify_onnx_file(path: Path) -> None:
    """Validate the output with the ONNX checker (avoids ORT IR / build skew in the host venv)."""
    import onnx

    m = onnx.load(str(path), load_external_data=True)
    try:
        onnx.checker.check_model(m, full_check=True)
    except Exception as e:
        _die(f"output model failed ONNX checker validation: {e}")


def _atomic_replace(src: Path, dst: Path) -> None:
    os.replace(str(src), str(dst))


def main() -> None:
    p = argparse.ArgumentParser(description="Quantize PaddleOCR ONNX model.")
    p.add_argument(
        "--input-model-dir",
        required=True,
        type=Path,
        help="Input model directory",
    )
    p.add_argument(
        "--output-model-dir",
        required=True,
        type=Path,
        help="Output model directory",
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=("dynamic", "static"),
        help="dynamic: weight-only (quantize_dynamic). static: QDQ (quantize_static, needs --calib-data-dir).",
    )
    p.add_argument(
        "--calib-data-dir",
        type=Path,
        default=None,
        help="Directory of float32 .npy calibration samples (static mode only; one tensor per file).",
    )
    p.add_argument(
        "--per-channel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Per-channel weight quantization (default: true).",
    )
    p.add_argument(
        "--calibration-method",
        default="MinMax",
        help="ORT CalibrationMethod name (e.g. MinMax, Entropy, Percentile).",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX checker validation of the output model after quantization.",
    )
    args = p.parse_args()

    if args.mode == "static" and args.calib_data_dir is None:
        _die("static mode requires --calib-data-dir")
    if args.mode == "dynamic" and args.calib_data_dir is not None:
        print("warning: --calib-data-dir is ignored for dynamic mode", file=sys.stderr)

    input_model_dir: Path = args.input_model_dir
    output_model_dir: Path = args.output_model_dir
    if not input_model_dir.is_dir():
        _die(f"input directory does not exist: {input_model_dir}")

    src_onnx = input_model_dir / "inference.onnx"
    src_yml = input_model_dir / "inference.yml"
    if not src_onnx.is_file():
        _die(f"missing {src_onnx}")
    if not src_yml.is_file():
        _die(f"missing {src_yml} (expected alongside inference.onnx)")

    out_onnx = output_model_dir / "inference.onnx"
    in_place = _same_file(input_model_dir, output_model_dir)
    if not in_place:
        output_model_dir.mkdir(parents=True, exist_ok=True)

    if in_place:
        fd, tmp_name = tempfile.mkstemp(
            prefix="inference.onnx.",
            suffix=".tmp",
            dir=str(input_model_dir),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            if args.mode == "dynamic":
                _quantize_dynamic(src_onnx, tmp_path, per_channel=args.per_channel)
            else:
                assert args.calib_data_dir is not None
                _quantize_static(
                    src_onnx,
                    tmp_path,
                    args.calib_data_dir,
                    per_channel=args.per_channel,
                    calibrate_method_name=args.calibration_method,
                )
            _atomic_replace(tmp_path, out_onnx)
        finally:
            if tmp_path.is_file() and not _same_file(tmp_path, out_onnx):
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
    else:
        try:
            if args.mode == "dynamic":
                _quantize_dynamic(src_onnx, out_onnx, per_channel=args.per_channel)
            else:
                assert args.calib_data_dir is not None
                _quantize_static(
                    src_onnx,
                    out_onnx,
                    args.calib_data_dir,
                    per_channel=args.per_channel,
                    calibrate_method_name=args.calibration_method,
                )
            shutil.copy2(src_yml, output_model_dir / "inference.yml")
        except Exception:
            if out_onnx.is_file():
                try:
                    out_onnx.unlink()
                except OSError:
                    pass
            raise

    if not args.no_verify:
        _verify_onnx_file(out_onnx)

    extra = f" and copied inference.yml" if not in_place else ""
    print(f"Wrote {out_onnx}{extra}")


if __name__ == "__main__":
    main()
