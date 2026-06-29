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

"""Export XCTest performance metrics from an xcresult bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _die(message: str, code: int = 1) -> int:
    sys.stderr.write(f"[extract_xctest_metrics] {message}\n")
    return code


def _metric_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def export_metrics(result: Path) -> Dict[str, Any]:
    proc = subprocess.run(
        [
            "xcrun",
            "xcresulttool",
            "get",
            "test-results",
            "metrics",
            "--path",
            str(result),
            "--compact",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "xcresulttool metrics export failed")
    raw = json.loads(proc.stdout or "[]")
    tests = raw if isinstance(raw, list) else []
    memory_metrics: List[Dict[str, Any]] = []
    for test in tests:
        if not isinstance(test, dict):
            continue
        for run in test.get("testRuns") or []:
            if not isinstance(run, dict):
                continue
            device = run.get("device") if isinstance(run.get("device"), dict) else {}
            config = (
                run.get("testPlanConfiguration")
                if isinstance(run.get("testPlanConfiguration"), dict)
                else {}
            )
            for metric in run.get("metrics") or []:
                if not isinstance(metric, dict):
                    continue
                display_name = str(metric.get("displayName") or "")
                identifier = str(metric.get("identifier") or "")
                unit = str(metric.get("unitOfMeasurement") or "")
                haystack = f"{display_name} {identifier} {unit}".lower()
                if "memory" not in haystack:
                    continue
                values = [
                    float(v)
                    for v in metric.get("measurements") or []
                    if isinstance(v, (int, float))
                ]
                memory_metrics.append(
                    {
                        "testIdentifier": test.get("testIdentifier"),
                        "testIdentifierURL": test.get("testIdentifierURL"),
                        "device": device,
                        "configuration": config,
                        "displayName": display_name,
                        "identifier": identifier or None,
                        "unitOfMeasurement": unit,
                        "measurements": values,
                        "summary": _metric_summary(values),
                    }
                )
    return {
        "schemaVersion": 1,
        "source": "xcresulttool get test-results metrics",
        "metrics": memory_metrics,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        doc = export_metrics(args.result)
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        return _die(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
