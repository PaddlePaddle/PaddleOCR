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

"""Merge optional validation JSONs into a local Markdown report.

Inputs:
    --compare-summary        JSON from compare_ocr_json.py
    --on-device-performance-json   JSON attachment "on-device-performance.json"
    --run-status             JSON from run_validation.sh

All inputs are optional; missing sections render placeholders.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _run_text(cmd: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_short() -> Optional[str]:
    return _run_text(["git", "rev-parse", "--short", "HEAD"])


def _xcode_version() -> Optional[str]:
    return _run_text(["xcodebuild", "-version"])


def _sw_vers() -> Optional[str]:
    return _run_text(["sw_vers", "-productVersion"])


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "— (path missing)"
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n / (1024 * 1024):.2f} MiB"


def _fmt_duration_ms(raw: Any) -> str:
    if raw is None:
        return "—"
    try:
        ms = float(raw)
    except (TypeError, ValueError):
        return "—"
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:.2f} s"


# ---------- run-status ----------


def _run_status_section(status: Dict[str, Any]) -> str:
    lines: List[str] = ["## Run status", ""]
    lines.append(f"- **Exit code:** `{status.get('exitCode', '—')}`")
    image = status.get("image") or {}
    if image:
        lines.append(
            f"- **Image:** `{image.get('name', '—')}` ({image.get('source', '—')})"
        )
    dest = status.get("destination")
    if dest:
        lines.append(f"- **Destination:** `{dest}`")
    started = status.get("runStartedAt")
    finished = status.get("runFinishedAt")
    if started and finished:
        lines.append(f"- **Run window:** {started} → {finished}")
    lines.append("")
    lines.append("| Step | Status | Duration | Notes |")
    lines.append("|------|--------|----------|-------|")
    for step in status.get("steps") or []:
        name = step.get("name", "?")
        st = (step.get("status") or "?").upper()
        dur = _fmt_duration_ms(step.get("durationMs"))
        note = step.get("reason") or ""
        if step.get("logPath"):
            note = (note + " · " if note else "") + f"see `{step['logPath']}`"
        if step.get("exitCode") not in (None, 0):
            note = (note + " · " if note else "") + f"exit {step['exitCode']}"
        lines.append(f"| {name} | {st} | {dur} | {note} |")
    return "\n".join(lines)


def _warning_banner(overall: str) -> str:
    return (
        f"> ⚠ This run ended with status **{overall}**; the numbers below may be partial or stale. "
        "Do not use them as a baseline.\n"
    )


# ---------- timing + memory (on-device performance) ----------


def _timing_markdown(data: Dict[str, Any]) -> str:
    lines = ["| Metric | mean | stdev | p90 |", "|--------|------|-------|-----|"]
    rows = [
        ("totalTimeMs", "total (ms)"),
        ("detectionTimeMs", "detection total (ms)"),
        ("detectionPreprocessTimeMs", "detection preprocess (ms)"),
        ("detectionInferenceTimeMs", "detection inference (ms)"),
        ("detectionPostprocessTimeMs", "detection postprocess (ms)"),
        ("recognitionTimeMs", "recognition total (ms)"),
        ("recognitionPreprocessTimeMs", "recognition preprocess (ms)"),
        ("recognitionInferenceTimeMs", "recognition inference (ms)"),
        ("recognitionPostprocessTimeMs", "recognition postprocess (ms)"),
        ("pipelineOverheadTimeMs", "pipeline overhead (ms)"),
    ]
    for key, label in rows:
        block = data.get(key)
        if not isinstance(block, dict):
            continue
        lines.append(
            "| {label} | {mean:.4f} | {stdev:.4f} | {p90:.4f} |".format(
                label=label,
                mean=float(block.get("mean", 0)),
                stdev=float(block.get("stdev", 0)),
                p90=float(block.get("p90", 0)),
            )
        )
    w = data.get("warmupIterations")
    m = data.get("measuredIterations")
    if w is not None or m is not None:
        lines.append("")
        lines.append(f"Warmup iterations: `{w}` · Measured iterations: `{m}`")
    return "\n".join(lines)


def _memory_markdown(data: Dict[str, Any]) -> str:
    if (
        data.get("memoryFootprintAfterLoadBytes") is None
        and data.get("memoryInferencePeakBytes") is None
    ):
        return ""
    rows = [
        "| Metric | Value |",
        "|--------|-------|",
    ]
    before = data.get("memoryFootprintBeforeLoadBytes")
    if before is not None:
        rows.append(f"| Footprint before load | {_fmt_bytes(int(before))} |")
    after = data.get("memoryFootprintAfterLoadBytes")
    if after is not None:
        rows.append(f"| Footprint after model load | {_fmt_bytes(int(after))} |")
    peak = data.get("memoryInferencePeakBytes")
    if peak is not None:
        rows.append(
            f"| Inference peak (max sample before/after each measured run) | {_fmt_bytes(int(peak))} |"
        )
    mean = data.get("memoryInferenceMeanBytes")
    if mean is not None:
        rows.append(
            f"| Inference mean (mean footprint after each measured run) | {_fmt_bytes(int(mean))} |"
        )
    n = data.get("memoryInferenceSampleCount")
    if n is not None:
        rows.append(f"| Inference memory samples | `{n}` |")
    ts = data.get("thermalState")
    if ts is not None:
        rows.append(f"| Thermal state | `{ts}` |")
    rows.append("")
    rows.append(
        "*Memory uses `task_vm_info.phys_footprint` (see Xcode Memory gauge). "
        "Peak is sampled around each inference, not a continuous trace; use Instruments for mid-call spikes.*"
    )
    return "\n".join(rows)


def _on_device_section(data: Dict[str, Any]) -> str:
    parts: List[str] = []
    ver = data.get("schemaVersion")
    if ver is not None:
        parts.append(f"Schema version: `{ver}`")
        parts.append("")
    parts.append(
        "*Latency and memory are both **runtime performance** attributes of the same on-device run.*"
    )
    parts.append("")
    parts.append("### Latency")
    parts.append("")
    parts.append(_timing_markdown(data))
    mem = _memory_markdown(data)
    if mem:
        parts.append("")
        parts.append("### Memory (resource footprint)")
        parts.append("")
        parts.append(mem)
    return "\n".join(parts)


# ---------- accuracy ----------


def _compare_section(data: Dict[str, Any]) -> str:
    passed = data.get("pass")
    status = "**PASS**" if passed is True else ("**FAIL**" if passed is False else "—")
    return "\n".join(
        [
            f"Status: {status}",
            "",
            "```json",
            json.dumps(data, indent=2, ensure_ascii=False),
            "```",
        ]
    )


# ---------- driver ----------


def _placeholder_from_status(status: Dict[str, Any], for_step: str) -> str:
    for step in status.get("steps") or []:
        if step.get("status") in ("fail", "skipped"):
            reason = step.get("reason") or ""
            if step.get("name") == for_step or reason:
                return (
                    f"*Unavailable — step `{step.get('name', '?')}` ended in "
                    f"`{step.get('status')}`"
                    + (f"; reason: {reason}" if reason else "")
                    + ".*"
                )
    return "*Unavailable — no artifact and no run-status reason.*"


def generate_report(
    *,
    out_path: Path,
    on_device_performance_json: Optional[Path],
    compare_summary: Optional[Path],
    run_status: Optional[Path],
) -> None:
    status: Dict[str, Any] = {}
    if run_status and run_status.is_file():
        status = _load_json(run_status)

    overall = (status.get("overall") or "").upper() or None

    compare_data = (
        _load_json(compare_summary)
        if compare_summary and compare_summary.is_file()
        else None
    )
    perf_data = (
        _load_json(on_device_performance_json)
        if on_device_performance_json and on_device_performance_json.is_file()
        else None
    )

    head_lines: List[str] = ["# iOS demo — validation report", ""]
    if overall:
        head_lines.append(f"**Overall:** {overall}")
        head_lines.append("")

    parts: List[str] = ["\n".join(head_lines)]

    if status:
        parts.append(_run_status_section(status))
        parts.append("")

    banner = _warning_banner(overall) if overall and overall != "PASS" else ""

    parts.append("## Accuracy validation")
    parts.append("")
    if banner:
        parts.append(banner)
    if compare_data is not None:
        parts.append(_compare_section(compare_data))
    else:
        parts.append(
            _placeholder_from_status(status, "compare")
            if status
            else "*No compare summary provided.*"
        )
    parts.append("")

    parts.append("## On-device runtime performance")
    parts.append("")
    if banner:
        parts.append(banner)
    if perf_data is not None:
        parts.append(_on_device_section(perf_data))
    else:
        parts.append(
            _placeholder_from_status(status, "extract-attachments")
            if status
            else "*No performance JSON provided.*"
        )
    parts.append("")

    parts.append("## Metadata")
    parts.append("")
    parts.append(
        f"- **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    if v := _sw_vers():
        parts.append(f"- **macOS:** {v}")
    if v := _xcode_version():
        parts.append("- **Xcode:**\n```\n" + v + "\n```")
    if g := _git_short():
        parts.append(f"- **Git (short):** `{g}`")
    parts.append("")
    parts.append("*Generated by `Scripts/generate_validation_report.py`.*")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=root / "out" / "validation-report.md"
    )
    parser.add_argument("--on-device-performance-json", type=Path, default=None)
    parser.add_argument("--compare-summary", type=Path, default=None)
    parser.add_argument("--run-status", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    generate_report(
        out_path=args.output.resolve(),
        on_device_performance_json=args.on_device_performance_json,
        compare_summary=args.compare_summary,
        run_status=args.run_status,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
