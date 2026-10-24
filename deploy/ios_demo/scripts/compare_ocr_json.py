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

"""Compare two OCR JSON files (reference vs device) using polygon IoU + CER.

Default thresholds (see argparse defaults) target a stricter CI gate than a loose
0.5 IoU baseline: tighter box alignment, ~95% char-level headroom on mean CER,
and limited reference-side misses.

Example:
  python compare_ocr_json.py ref.json ios.json \\
    --iou-threshold 0.65 --cer-threshold 0.05 --max-unmatched-ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _cer(ref: str, hyp: str) -> float:
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    return _levenshtein(ref, hyp) / max(len(ref), 1)


def _polygon_iou(
    poly_a: Sequence[Sequence[float]], poly_b: Sequence[Sequence[float]]
) -> float:
    try:
        from shapely.geometry import Polygon
    except ImportError as exc:
        raise RuntimeError(
            "compare_ocr_json.py requires `pip install shapely`"
        ) from exc

    def _to_poly(p: Sequence[Sequence[float]]) -> Polygon:
        pts = [(float(x), float(y)) for x, y in p]
        if len(pts) < 3:
            return Polygon()
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        return Polygon(pts)

    p1 = _to_poly(poly_a)
    p2 = _to_poly(poly_b)
    if p1.is_empty or p2.is_empty or not p1.is_valid or not p2.is_valid:
        return 0.0
    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    if union <= 0:
        return 0.0
    return float(inter / union)


def _load_items(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{path}: missing 'items' array")
    return items


def _greedy_match(
    ref_items: List[Dict[str, Any]],
    hyp_items: List[Dict[str, Any]],
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Return (pairs as ref_idx, hyp_idx, iou), unmatched_ref, unmatched_hyp."""
    candidates: List[Tuple[float, int, int]] = []
    for i, ri in enumerate(ref_items):
        ra = ri.get("polygon")
        if not isinstance(ra, list):
            continue
        for j, hj in enumerate(hyp_items):
            ha = hj.get("polygon")
            if not isinstance(ha, list):
                continue
            iou = _polygon_iou(ra, ha)
            candidates.append((iou, i, j))

    candidates.sort(key=lambda t: t[0], reverse=True)
    used_r = set()
    used_h = set()
    pairs: List[Tuple[int, int, float]] = []
    for iou, i, j in candidates:
        if iou < iou_threshold:
            break
        if i in used_r or j in used_h:
            continue
        used_r.add(i)
        used_h.add(j)
        pairs.append((i, j, iou))

    unmatched_r = [i for i in range(len(ref_items)) if i not in used_r]
    unmatched_h = [j for j in range(len(hyp_items)) if j not in used_h]
    return pairs, unmatched_r, unmatched_h


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare OCR JSON outputs")
    parser.add_argument(
        "reference", type=Path, help="Reference JSON (e.g. paddleocr_reference)"
    )
    parser.add_argument("hypothesis", type=Path, help="Device / second JSON")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.65,
        help="Min IoU to pair quadrilateral boxes (stricter than common 0.5 VOC-style cutoff)",
    )
    parser.add_argument(
        "--cer-threshold",
        type=float,
        default=0.05,
        help="Fail if mean CER on matched pairs exceeds this (e.g. 0.05 ≈ 95%% char accuracy headroom)",
    )
    parser.add_argument(
        "--max-unmatched-ratio",
        type=float,
        default=0.1,
        help="Fail if unmatched ref lines / len(ref) exceeds this",
    )
    parser.add_argument(
        "--json-summary-out",
        type=Path,
        default=None,
        help="Write the same JSON as stdout to this path (PASS or FAIL) for generate_benchmark_report.py",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    ref_items = _load_items(args.reference)
    hyp_items = _load_items(args.hypothesis)

    pairs, unmatched_r, unmatched_h = _greedy_match(
        ref_items, hyp_items, args.iou_threshold
    )

    cers: List[float] = []
    for ri, hj, _ in pairs:
        rt = str(ref_items[ri].get("text", ""))
        ht = str(hyp_items[hj].get("text", ""))
        cers.append(_cer(rt, ht))

    mean_cer = sum(cers) / len(cers) if cers else 0.0
    nref = max(len(ref_items), 1)
    unmatched_ratio = len(unmatched_r) / nref

    report = {
        "matched_pairs": len(pairs),
        "reference_count": len(ref_items),
        "hypothesis_count": len(hyp_items),
        "unmatched_reference": len(unmatched_r),
        "unmatched_hypothesis": len(unmatched_h),
        "unmatched_reference_ratio": unmatched_ratio,
        "mean_cer_matched": mean_cer,
        "iou_threshold": args.iou_threshold,
        "cer_threshold": args.cer_threshold,
        "max_unmatched_ratio": args.max_unmatched_ratio,
    }
    failed = False
    if mean_cer > args.cer_threshold:
        print(f"FAIL: mean CER {mean_cer:.4f} > {args.cer_threshold}", file=sys.stderr)
        failed = True
    if unmatched_ratio > args.max_unmatched_ratio:
        print(
            f"FAIL: unmatched ref ratio {unmatched_ratio:.4f} > {args.max_unmatched_ratio}",
            file=sys.stderr,
        )
        failed = True

    report["pass"] = not failed
    out_txt = json.dumps(report, indent=2, ensure_ascii=False)
    print(out_txt)

    if args.json_summary_out is not None:
        args.json_summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_summary_out.write_text(out_txt + "\n", encoding="utf-8")

    if not failed:
        print("PASS", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
