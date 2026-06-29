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

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from utils import build_npy_dir_reader, die


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def _run_ort_quant_pre_process(src_onnx: Path, work_dir: Path) -> Path:
    """Run ORT *quant_pre_process*; write a new ONNX with ``onnx.quant.pre_process`` metadata.

    Tries, in order: default → ``skip_symbolic_shape`` (some det/NMS graphs fail symbolic infer)
    → also ``skip_optimization`` (last resort). Removes any failed output before retrying.
    """
    try:
        from onnxruntime.quantization import quant_pre_process
    except ImportError as e:
        die(
            f"onnxruntime.quantization.quant_pre_process is required for --ort-preprocess: {e}"
        )

    fd, raw = tempfile.mkstemp(
        suffix=".pre.onnx",
        prefix="inference.ort_",
        dir=str(work_dir),
    )
    os.close(fd)
    out = Path(raw)

    attempts: list[tuple[str, dict[str, bool]]] = [
        (
            "(full symbolic + ORT optimize)",
            {"skip_symbolic_shape": False, "skip_optimization": False},
        ),
        (
            "(skip symbolic shape, keep ORT optimize)",
            {"skip_symbolic_shape": True, "skip_optimization": False},
        ),
        (
            "(skip symbolic shape, skip ORT graph optimization)",
            {"skip_symbolic_shape": True, "skip_optimization": True},
        ),
    ]
    last_err: Exception | None = None
    for label, kwargs in attempts:
        out.unlink(missing_ok=True)
        try:
            quant_pre_process(
                input_model=str(src_onnx),
                output_model_path=str(out),
                **kwargs,
            )
        except Exception as e:
            last_err = e
            continue
        if label != attempts[0][0]:
            print(
                f"warning: ORT quant_pre_process succeeded with {label}.",
                file=sys.stderr,
            )
        return out

    out.unlink(missing_ok=True)
    die(
        f"ORT quant_pre_process failed after {len(attempts)} attempt(s). Last error: {last_err!r}. "
        "You can try again with --no-ort-preprocess to quantize the original model only."
    )


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

    reader = build_npy_dir_reader(src_onnx, calib_dir)
    try:
        method = getattr(CalibrationMethod, calibrate_method_name)
    except AttributeError:
        die(
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


def _default_domain_onnx_opset(m: Any) -> int:
    """Max declared opset for default / ``ai.onnx`` imports."""
    v = 0
    for oi in m.opset_import:
        dom = oi.domain or ""
        if dom in ("", "ai.onnx"):
            v = max(v, int(oi.version))
    return v


def _try_onnx_opset_via_version_converter(path: Path, target_opset: int) -> str:
    import onnx
    from onnx import version_converter

    m = onnx.load(str(path), load_external_data=True)
    cur = _default_domain_onnx_opset(m)
    if cur >= target_opset:
        return "skipped_already_ge_target"
    try:
        m2 = version_converter.convert_version(m, target_opset)
    except Exception as e:
        print(
            f"warning: onnx.version_converter.convert_version(..., {target_opset}) failed: {e!r}. "
            "Full checker may still fail. Try a newer `onnx`, or use --no-verify if ORT loads the model.",
            file=sys.stderr,
        )
        return "convert_failed"
    try:
        onnx.save(m2, str(path))
    except Exception as e:
        die(f"failed to write ONNX after version conversion: {e}")
    return "converted"


def _verify_onnx_file(path: Path) -> None:
    """Validate the output with the ONNX checker (avoids ORT IR / build skew in the host venv)."""
    import onnx

    m = onnx.load(str(path), load_external_data=True)
    try:
        onnx.checker.check_model(m, full_check=True)
    except Exception as e:
        die(f"output model failed ONNX checker validation: {e}")


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
    p.add_argument(
        "--ort-preprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before quantize_static / quantize_dynamic, run ORT quant_pre_process (shape infer + "
            "optional graph optimization) and attach onnx.quant metadata."
        ),
    )
    p.add_argument(
        "--onnx-opset-convert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After ORT writes the output, run onnx.version_converter when the graph's declared "
            "default-domain opset is below --onnx-target-opset. Use --no-onnx-opset-convert to keep "
            "raw ORT output."
        ),
    )
    p.add_argument(
        "--onnx-target-opset",
        type=int,
        default=13,
        metavar="N",
        help=(
            "Target ONNX opset for --onnx-opset-convert (default: 13). Ignored when conversion is off."
        ),
    )
    args = p.parse_args()

    if args.mode == "static" and args.calib_data_dir is None:
        die("static mode requires --calib-data-dir")
    if args.mode == "dynamic" and args.calib_data_dir is not None:
        print("warning: --calib-data-dir is ignored for dynamic mode", file=sys.stderr)
    if args.onnx_target_opset < 1:
        die("--onnx-target-opset must be >= 1")

    input_model_dir: Path = args.input_model_dir
    output_model_dir: Path = args.output_model_dir
    if not input_model_dir.is_dir():
        die(f"input directory does not exist: {input_model_dir}")

    src_onnx = input_model_dir / "inference.onnx"
    src_yml = input_model_dir / "inference.yml"
    if not src_onnx.is_file():
        die(f"missing {src_onnx}")
    if not src_yml.is_file():
        die(f"missing {src_yml} (expected alongside inference.onnx)")

    out_onnx = output_model_dir / "inference.onnx"
    in_place = _same_file(input_model_dir, output_model_dir)
    if not in_place:
        output_model_dir.mkdir(parents=True, exist_ok=True)

    pre_onnx: Path | None = None
    if args.ort_preprocess:
        pre_onnx = _run_ort_quant_pre_process(src_onnx, input_model_dir)
    quant_src = pre_onnx if pre_onnx is not None else src_onnx

    try:
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
                    _quantize_dynamic(quant_src, tmp_path, per_channel=args.per_channel)
                else:
                    assert args.calib_data_dir is not None
                    _quantize_static(
                        quant_src,
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
                    _quantize_dynamic(quant_src, out_onnx, per_channel=args.per_channel)
                else:
                    assert args.calib_data_dir is not None
                    _quantize_static(
                        quant_src,
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
    finally:
        if pre_onnx is not None and pre_onnx.is_file():
            try:
                pre_onnx.unlink()
            except OSError:
                pass

    if args.onnx_opset_convert:
        st = _try_onnx_opset_via_version_converter(out_onnx, args.onnx_target_opset)
        if st == "converted":
            print(
                f"note: applied onnx.version_converter to opset {args.onnx_target_opset}.",
                file=sys.stderr,
            )
    if not args.no_verify:
        _verify_onnx_file(out_onnx)

    extra = f" and copied inference.yml" if not in_place else ""
    print(f"Wrote {out_onnx}{extra}")


if __name__ == "__main__":
    main()
