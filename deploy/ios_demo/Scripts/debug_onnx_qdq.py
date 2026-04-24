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

"""QDQ quantization debug: align with ONNX Runtime ``qdq_loss_debug`` (official example flow).

This mirrors the *Debugging* section of:
https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md

Workflow:
1. Augment float32 and QDQ models to expose intermediate tensors
   (``modify_model_output_intermediate_tensors``).
2. Run both with the same calibration ``.npy`` inputs
   (``collect_activations``), matching ``quantize_onnx_model`` / ``build_onnx_calib_npy`` data.
3. ``create_activation_matching`` + SQNR; ``create_weight_matching`` + per-weight SQNR.
4. Print a report and optionally write JSON (scalar metrics only).

*float* should be the **same** float graph you quantized (ideally the ORT
``quant_pre_process`` output if you used ``--ort-preprocess`` in ``quantize_onnx_model.py``),
or the graph is unlikely to line-tensor match the QDQ model.

Example:
  python3 scripts/debug_onnx_qdq.py \\
    --float-model /path/to/float_infer.onnx \\
    --qdq-model /path/to/inference_int8.onnx \\
    --calib-data-dir /path/to/calib_npy \\
    --max-samples 3 \\
    --json-report qdq_debug.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from utils import build_npy_dir_reader, die

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_op_types(s: str | None) -> list[str] | None:
    if not s or not s.strip():
        return None
    return [t.strip() for t in s.split(",") if t.strip()]


def _safe_compute_weight_error(
    wmatch: dict[str, dict[str, Any]],
    err_func: Any,
) -> dict[str, float]:
    """Like ORT ``compute_weight_error``, but coerce scalars to 0-d arrays.

    ``create_weight_matching`` can set ``dequantized`` to a bare ``numpy.float32``; ORT
    ``compute_signal_to_quantization_noice_ratio`` then does ``len(ylist)`` on that scalar
    and raises ``TypeError``.
    """
    import numpy as np

    out: dict[str, float] = {}
    for name, m in wmatch.items():
        try:
            xf = np.asarray(m["float"])
            yd = np.asarray(m["dequantized"])
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("weight SQNR skip %r: %s", name, e)
            continue
        if xf.size == 0 and yd.size == 0:
            continue
        if xf.size != yd.size:
            logger.warning(
                "weight SQNR skip %r: size mismatch %s vs %s",
                name,
                getattr(xf, "shape", xf),
                getattr(yd, "shape", yd),
            )
            continue
        try:
            out[name] = float(err_func(xf, yd))
        except (TypeError, ValueError, RuntimeError) as e:
            logger.warning("weight SQNR skip %r: %s", name, e)
    return out


def _safe_compute_activation_error(
    match: dict[str, dict[str, Any]],
    err_func: Any,
) -> dict[str, dict[str, float]]:
    """Like ORT ``compute_activation_error``, but skip missing *float* entries (avoids KeyError)."""
    result: dict[str, dict[str, float]] = {}
    for name, m in match.items():
        e: dict[str, float] = {
            "qdq_err": float(err_func(m["pre_qdq"], m["post_qdq"])),
        }
        if m.get("float"):
            e["xmodel_err"] = float(err_func(m["float"], m["post_qdq"]))
        result[name] = e
    return result


def _sqnr_key(entry: dict[str, float], prefer_xmodel: bool) -> float:
    if prefer_xmodel and "xmodel_err" in entry:
        return entry["xmodel_err"]
    return entry.get("qdq_err", float("nan"))


def _run(
    float_model: Path,
    qdq_model: Path,
    calib: Path,
    max_samples: int,
    op_types: list[str] | None,
    json_report: Path | None,
    top_n: int,
    skip_weights: bool,
    skip_activations: bool,
) -> None:
    from onnxruntime.quantization.qdq_loss_debug import (
        collect_activations,
        compute_signal_to_quantization_noice_ratio,
        create_activation_matching,
        create_weight_matching,
        modify_model_output_intermediate_tensors,
    )

    if not float_model.is_file():
        die(f"not a file: {float_model}")
    if not qdq_model.is_file():
        die(f"not a file: {qdq_model}")
    if not skip_activations:
        if not calib.is_dir():
            die(f"not a directory: {calib}")

    out_weights: dict[str, float] = {}
    out_act: dict[str, dict[str, float]] = {}

    if not skip_weights:
        logger.info("weight matching (no inference run)…")
        try:
            wmatch = create_weight_matching(str(float_model), str(qdq_model))
        except Exception as e:
            die(f"create_weight_matching failed: {e}")
        out_weights = _safe_compute_weight_error(
            wmatch, compute_signal_to_quantization_noice_ratio
        )
        print("\n--- Weight SQNR (dB, higher is better) ---\n")
        for name in sorted(out_weights, key=lambda x: out_weights[x]):
            print(f"  {name}: {out_weights[name]:.4f}")

    if not skip_activations:
        with tempfile.TemporaryDirectory(prefix="ort.qdq.debug.") as td:
            tmp = Path(td)
            aug_f = tmp / "augmented_float.onnx"
            aug_q = tmp / "augmented_qdq.onnx"
            op_arg = op_types if op_types is not None else []
            logger.info("augmenting float model → %s", aug_f)
            try:
                modify_model_output_intermediate_tensors(
                    str(float_model), str(aug_f), op_types_for_saving=op_arg
                )
            except Exception as e:
                die(f"modify_model_output_intermediate_tensors (float) failed: {e}")
            logger.info("augmenting QDQ model → %s", aug_q)
            try:
                modify_model_output_intermediate_tensors(
                    str(qdq_model), str(aug_q), op_types_for_saving=op_arg
                )
            except Exception as e:
                die(f"modify_model_output_intermediate_tensors (qdq) failed: {e}")

            r_float = build_npy_dir_reader(float_model, calib, max_samples=max_samples)
            r_qdq = build_npy_dir_reader(qdq_model, calib, max_samples=max_samples)
            try:
                logger.info("collect_activations (float)…")
                act_f = collect_activations(str(aug_f), r_float)
                logger.info("collect_activations (QDQ)…")
                act_q = collect_activations(str(aug_q), r_qdq)
            except Exception as e:
                die(f"collect_activations failed: {e}")

        match = create_activation_matching(act_q, act_f)
        if not match:
            print(
                "\nwarning: empty activation match — is the quantized model QDQ (not QOperator)?",
                file=sys.stderr,
            )
        out_act = _safe_compute_activation_error(
            match, compute_signal_to_quantization_noice_ratio
        )
        has_xm = any("xmodel_err" in v for v in out_act.values())
        scored = sorted(
            out_act.items(),
            key=lambda kv: _sqnr_key(kv[1], prefer_xmodel=has_xm),
        )[: top_n if top_n > 0 else len(out_act)]
        which = "xmodel_err (float vs post_qdq)" if has_xm else "qdq_err (q dq pair)"
        print(
            f"\n--- Top activation SQNR (dB, {which}; lower = worse) — showing "
            f"{len(scored)} of {len(out_act)} —\n"
        )
        for name, m in scored:
            if has_xm and "xmodel_err" in m:
                print(
                    f"  {name}: pre/post_qdq {m['qdq_err']:.4f}  "
                    f"float vs post_qdq {m['xmodel_err']:.4f}"
                )
            else:
                print(f"  {name}: pre/post_qdq {m['qdq_err']:.4f}")

    if json_report is not None:
        payload: dict[str, Any] = {
            "weight_sqnr_db": out_weights,
            "activation_sqnr_db": out_act,
        }
        json_report = json_report.resolve()
        json_report.parent.mkdir(parents=True, exist_ok=True)
        with json_report.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nwrote {json_report}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ORT-style QDQ debug: float vs QDQ activations/weights (same .npy calib as quantize).",
    )
    ap.add_argument(
        "--float-model",
        type=Path,
        required=True,
        help="Float32 ONNX (same topology as the graph you quantized; use ORT preprocessed model if you used it before quantize).",
    )
    ap.add_argument(
        "--qdq-model",
        type=Path,
        required=True,
        help="Statically quantized (QDQ) ONNX model, e.g. from quantize_onnx_model.py.",
    )
    ap.add_argument(
        "--calib-data-dir",
        type=Path,
        default=None,
        help="Directory of float32 .npy used for calibration (required unless --skip-activations).",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Max number of .npy files to use (default 5; full runs can be slow and memory-heavy).",
    )
    ap.add_argument(
        "--op-types",
        type=str,
        default="",
        help="Comma-separated op types to save intermediates for (e.g. Conv,Add). Empty = all float tensors (very slow and large).",
    )
    ap.add_argument(
        "--json-report",
        type=Path,
        default=None,
        help="Write metrics-only JSON to this path.",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=40,
        help="How many activations to print (sorted worst-first by SQNR). 0 = print all.",
    )
    ap.add_argument(
        "--skip-weights",
        action="store_true",
        help="Only run activation path (faster if you only care about activations).",
    )
    ap.add_argument(
        "--skip-activations",
        action="store_true",
        help="Only run weight matching (no collect_activations; ignores calib).",
    )
    args = ap.parse_args()

    if args.skip_weights and args.skip_activations:
        die("cannot set both --skip-weights and --skip-activations")
    if not args.skip_activations:
        if args.calib_data_dir is None:
            die("activation path requires --calib-data-dir")
        if args.max_samples < 1:
            die("--max-samples must be >= 1 for activation path")

    _run(
        float_model=args.float_model,
        qdq_model=args.qdq_model,
        calib=args.calib_data_dir or Path(),
        max_samples=args.max_samples,
        op_types=_parse_op_types(args.op_types),
        json_report=args.json_report,
        top_n=args.top,
        skip_weights=args.skip_weights,
        skip_activations=args.skip_activations,
    )


if __name__ == "__main__":
    main()
