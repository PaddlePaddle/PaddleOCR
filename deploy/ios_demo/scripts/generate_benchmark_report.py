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

"""Merge optional benchmark JSONs into a local Markdown report.

Inputs:
    --accuracy-summary       Optional JSON from compare_ocr_json.py
    --on-device-performance-json   JSON attachment "on-device-performance.json"
    --run-status             JSON from run_benchmark.sh
    --xctest-metrics-json    JSON from extract_xctest_metrics.py

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


def _fmt_ms(raw: Any) -> str:
    try:
        return f"{float(raw):.4f} ms"
    except (TypeError, ValueError):
        return "—"


def _fmt_shape(raw: Any) -> str:
    if isinstance(raw, list) and raw:
        return "`[" + ", ".join(str(x) for x in raw) + "]`"
    return "—"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _benchmark_value(
    status: Dict[str, Any],
    perf_data: Optional[Dict[str, Any]],
    key: str,
) -> Any:
    benchmark = (
        status.get("benchmark") if isinstance(status.get("benchmark"), dict) else {}
    )
    if key in benchmark:
        return benchmark.get(key)
    if isinstance(perf_data, dict):
        return perf_data.get(key)
    return None


def _benchmark_warnings(
    status: Dict[str, Any],
    perf_data: Optional[Dict[str, Any]],
) -> List[str]:
    warnings: List[str] = []
    build = _benchmark_value(status, perf_data, "buildConfiguration")
    is_sim = _benchmark_value(status, perf_data, "isSimulator")
    profiling = _benchmark_value(status, perf_data, "ortProfilingEnabled")
    if build and str(build) != "Release":
        warnings.append(f"build configuration is `{build}`, not `Release`")
    if _truthy(is_sim):
        warnings.append("destination is simulator")
    if _truthy(profiling):
        warnings.append("ORT profiling is enabled")
    return warnings


def _benchmark_metadata_markdown(
    status: Dict[str, Any],
    perf_data: Optional[Dict[str, Any]],
) -> str:
    rows: List[str] = ["| Metric | Value |", "|--------|-------|"]
    ep = _benchmark_value(status, perf_data, "ortExecutionProvider")
    if ep is not None:
        rows.append(f"| ORT execution provider | `{ep}` |")
    for key, label in [
        ("buildConfiguration", "Build configuration"),
        ("deviceModel", "Device model"),
        ("osVersion", "OS version"),
        ("isSimulator", "Simulator"),
        ("modelPreset", "Model preset"),
        ("ortProfilingEnabled", "ORT profiling"),
    ]:
        value = _benchmark_value(status, perf_data, key)
        if value is None:
            continue
        rows.append(f"| {label} | `{value}` |")
    warnings = _benchmark_warnings(status, perf_data)
    if warnings:
        rows.append("")
        rows.append(
            "> Not publishable: "
            + "; ".join(warnings)
            + ". Only Release runs on a real device with ORT profiling disabled should be used as publishable latency numbers."
        )
    else:
        rows.append("")
        rows.append(
            "> Publishable baseline: Release run on a real device with ORT profiling disabled."
        )
    return "\n".join(rows)


def _model_artifacts_markdown(status: Dict[str, Any]) -> str:
    benchmark = (
        status.get("benchmark") if isinstance(status.get("benchmark"), dict) else {}
    )
    model_files = (
        benchmark.get("modelFiles")
        if isinstance(benchmark.get("modelFiles"), dict)
        else {}
    )
    det = (
        model_files.get("detection")
        if isinstance(model_files.get("detection"), dict)
        else {}
    )
    rec = (
        model_files.get("recognition")
        if isinstance(model_files.get("recognition"), dict)
        else {}
    )
    rows = ["| Artifact | Format | Size |", "|----------|--------|------|"]
    rows.append(
        f"| Detection model | `{det.get('format') or '—'}` | {_fmt_bytes(det.get('sizeBytes'))} |"
    )
    rows.append(
        f"| Recognition model | `{rec.get('format') or '—'}` | {_fmt_bytes(rec.get('sizeBytes'))} |"
    )
    rows.append(
        f"| Total model weights | — | {_fmt_bytes(model_files.get('totalSizeBytes'))} |"
    )
    rows.append(
        f"| App executable | — | {_fmt_bytes(benchmark.get('appBinarySizeBytes'))} |"
    )
    return "\n".join(rows)


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
    benchmark = (
        status.get("benchmark") if isinstance(status.get("benchmark"), dict) else {}
    )
    if benchmark:
        if build := benchmark.get("buildConfiguration"):
            lines.append(f"- **Build configuration:** `{build}`")
        ep = benchmark.get("ortExecutionProvider")
        if ep:
            lines.append(f"- **ORT execution provider:** `{ep}`")
        if scope := benchmark.get("testingScope"):
            lines.append(f"- **Testing scope:** `{scope}`")
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


def _stale_data_banner(overall: str) -> str:
    return (
        f"> This run ended with status **{overall}**; the numbers below may be partial or stale. "
        "Do not use them as a baseline.\n"
    )


def _performance_section_banner(overall: Optional[str]) -> str:
    if overall and overall != "COMPLETED":
        return _stale_data_banner(overall)
    return ""


# ---------- timing + memory (on-device performance) ----------


def _timing_markdown(data: Dict[str, Any]) -> str:
    lines = ["| Metric | mean | stdev | p90 |", "|--------|------|-------|-----|"]
    for key, label in [
        ("totalTimeMs", "Full OCR latency (ms)"),
        ("detectionTimeMs", "Detection stage latency (ms)"),
        ("detectionPreprocessTimeMs", "Detection preprocess latency (ms)"),
        ("detectionInferenceTimeMs", "Detection inference segment latency (ms)"),
        ("detectionPostprocessTimeMs", "Detection postprocess latency (ms)"),
        ("recognitionTimeMs", "Recognition stage latency (ms)"),
        ("recognitionPreprocessTimeMs", "Recognition preprocess latency (ms)"),
        ("recognitionInferenceTimeMs", "Recognition inference segment latency (ms)"),
        ("recognitionPostprocessTimeMs", "Recognition postprocess latency (ms)"),
    ]:
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
    ovh = data.get("pipelineOverheadTimeMs")
    if isinstance(ovh, dict):
        lines.append(
            "| Pipeline overhead latency (ms) | {mean:.4f} | {stdev:.4f} | {p90:.4f} |".format(
                mean=float(ovh.get("mean", 0)),
                stdev=float(ovh.get("stdev", 0)),
                p90=float(ovh.get("p90", 0)),
            )
        )
    w = data.get("warmupIterations")
    m = data.get("measuredIterations")
    if w is not None or m is not None:
        lines.append("")
        lines.append(f"Warmup iterations: `{w}` · Measured iterations: `{m}`")
    lines.append("")
    lines.append(
        "*Inference segment latency includes the tensor input/output handling required by the ORT wrapper, "
        "`session.run`, output reads, and validation. It is not pure kernel latency or isolated `session.run` latency.*"
    )

    per_line_extra = _recognition_per_line_markdown(data)
    if per_line_extra:
        lines.append("")
        lines.append(per_line_extra)
    return "\n".join(lines)


def _input_profile_markdown(data: Dict[str, Any]) -> str:
    dist = (
        data.get("inputShapeDistribution")
        if isinstance(data.get("inputShapeDistribution"), dict)
        else {}
    )
    rows = [
        "| Model | Input tensor shape | Count |",
        "|-------|--------------------|-------|",
    ]
    for model_key, label in [
        ("detection", "Detection"),
        ("recognition", "Recognition"),
    ]:
        samples = dist.get(model_key)
        if not isinstance(samples, list) or not samples:
            rows.append(f"| {label} | — | `0` |")
            continue
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            rows.append(
                f"| {label} | {_fmt_shape(sample.get('shape'))} | `{sample.get('count', '—')}` |"
            )
    rows.append("")
    rows.append("| Metric | Value |")
    rows.append("|--------|-------|")
    rows.append(
        f"| First measured run line count | `{data.get('firstMeasuredLineCount', '—')}` |"
    )
    rows.append(f"| Cold model load time | {_fmt_ms(data.get('coldLoadTimeMs'))} |")
    rows.append("")
    rows.append(
        "*Input tensor shape counts are counted per model invocation in the measured loop.*"
    )
    return "\n".join(rows)


def _recognition_per_line_markdown(data: Dict[str, Any]) -> str:
    """Renders pooled `recognitionPerLine` samples."""
    pl = data.get("recognitionPerLine")
    if not isinstance(pl, dict):
        return ""
    out: List[str] = []
    pooled = pl.get("pooled")
    any_pooled = False
    pooled_lines: List[str] = [
        "| Pooled (all lines × measured runs) | count | mean | stdev | p90 |",
        "|------------------------------------|------|------|-------|-----|",
    ]
    for key, label in [
        ("preprocessMs", "Per-line recognition preprocess pooled (ms)"),
        ("inferenceMs", "Per-line recognition inference segment pooled (ms)"),
        ("postprocessMs", "Per-line recognition postprocess pooled (ms)"),
        ("totalMs", "Per-line recognition total pooled (ms)"),
    ]:
        if not isinstance(pooled, dict):
            break
        b = pooled.get(key)
        if not isinstance(b, dict):
            continue
        c = pooled.get("count") or b.get("count")
        if c is None or c == 0:
            continue
        any_pooled = True
        pooled_lines.append(
            "| {label} | {c} | {me:.4f} | {sd:.4f} | {p90:.4f} |".format(
                label=label,
                c=int(c),
                me=float(b.get("mean", 0)),
                sd=float(b.get("stdev", 0)),
                p90=float(b.get("p90", 0)),
            )
        )
    if any_pooled:
        out.append("#### Per-line Recognition Latency (ms)")
        out.append("")
        out.append(
            "*Pooled* merges every detected line from every measured full-image run."
        )
        out.append("")
        out.extend(pooled_lines)

    return "\n".join(out) if out else ""


def _fmt_metric_number(value: Any, unit: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if unit.lower() == "kb":
        return f"{number / 1024:.2f} MB"
    if "byte" in unit.lower():
        return _fmt_bytes(int(number))
    if abs(number) >= 1000:
        text = f"{number:.0f}"
    else:
        text = f"{number:.4f}".rstrip("0").rstrip(".")
    return f"{text} {unit}".rstrip()


def _xctest_memory_rows(xctest_metrics: Optional[Dict[str, Any]]) -> List[str]:
    rows: List[str] = []
    metrics = (
        xctest_metrics.get("metrics") if isinstance(xctest_metrics, dict) else None
    )
    if not isinstance(metrics, list) or not metrics:
        rows.append("| XCTest memory metric | `not available` |")
        return rows
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        summary = (
            metric.get("summary") if isinstance(metric.get("summary"), dict) else {}
        )
        unit = str(metric.get("unitOfMeasurement") or "")
        raw_label = str(
            metric.get("displayName") or metric.get("identifier") or "memory"
        )
        label = {
            "Memory Physical": "physical memory delta",
            "Memory Peak Physical": "physical memory peak",
        }.get(raw_label, raw_label)
        test_id = metric.get("testIdentifier")
        prefix = f"XCTest {label}"
        if test_id:
            prefix += f" ({test_id})"
        rows.append(f"| {prefix} count | `{int(summary.get('count', 0))}` |")
        rows.append(
            f"| {prefix} min | {_fmt_metric_number(summary.get('min'), unit)} |"
        )
        rows.append(
            f"| {prefix} max | {_fmt_metric_number(summary.get('max'), unit)} |"
        )
        rows.append(
            f"| {prefix} mean | {_fmt_metric_number(summary.get('mean'), unit)} |"
        )
    return rows


def _memory_markdown(
    data: Dict[str, Any],
    xctest_metrics: Optional[Dict[str, Any]],
) -> str:
    if data.get("memoryFootprintAfterLoadBytes") is None and xctest_metrics is None:
        return ""
    rows = [
        "| Metric | Value |",
        "|--------|-------|",
    ]
    rows.extend(_xctest_memory_rows(xctest_metrics))
    before = data.get("memoryFootprintBeforeLoadBytes")
    if before is not None:
        rows.append(f"| Footprint before load | {_fmt_bytes(int(before))} |")
    after = data.get("memoryFootprintAfterLoadBytes")
    if after is not None:
        rows.append(f"| Footprint after model load | {_fmt_bytes(int(after))} |")
    ts = data.get("thermalState")
    if ts is not None:
        rows.append(f"| Thermal state | `{ts}` |")
    rows.append("")
    rows.append(
        "`physical memory delta` is a per-iteration net change, so negative values mean memory was released during that measured block. "
        "`task_vm_info.phys_footprint` rows are diagnostic snapshots before and after model loading, not continuous peak measurements.*"
    )
    return "\n".join(rows)


def _on_device_section(
    data: Dict[str, Any],
    xctest_metrics: Optional[Dict[str, Any]],
    status: Optional[Dict[str, Any]] = None,
) -> str:
    parts: List[str] = []
    ver = data.get("schemaVersion")
    if ver is not None:
        parts.append(f"Schema version: `{ver}`")
        parts.append("")
    parts.append("### Benchmark metadata")
    parts.append("")
    parts.append(_benchmark_metadata_markdown(status or {}, data))
    parts.append("")
    parts.append(
        "*Latency and memory are both runtime performance attributes, but they are sampled by separate benchmark paths.*"
    )
    parts.append("")
    parts.append("### Model and app artifacts")
    parts.append("")
    parts.append(_model_artifacts_markdown(status or {}))
    parts.append("")
    parts.append("### Input profile")
    parts.append("")
    parts.append(_input_profile_markdown(data))
    parts.append("")
    parts.append("### Latency")
    parts.append("")
    parts.append(_timing_markdown(data))
    mem = _memory_markdown(data, xctest_metrics)
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


def _accuracy_section(
    summary_data: Optional[Dict[str, Any]],
    run_status: Dict[str, Any],
) -> str:
    accuracy = (
        run_status.get("accuracy")
        if isinstance(run_status.get("accuracy"), dict)
        else {}
    )
    enabled = bool(accuracy.get("enabled")) or summary_data is not None
    if not enabled:
        return ""

    parts: List[str] = ["## Accuracy check", ""]
    status = str(accuracy.get("status") or "").upper()
    reason = accuracy.get("reason")
    if status:
        parts.append(f"Precheck status: **{status}**")
        if reason:
            parts.append("")
            parts.append(f"> {reason}")
        parts.append("")
    parts.append("")
    if summary_data is not None:
        parts.append(_compare_section(summary_data))
    elif enabled:
        parts.append("*No accuracy summary JSON was produced.*")
    return "\n".join(parts)


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
    accuracy_summary: Optional[Path],
    run_status: Optional[Path],
    xctest_metrics_json: Optional[Path],
) -> None:
    status: Dict[str, Any] = {}
    if run_status and run_status.is_file():
        status = _load_json(run_status)

    overall = (status.get("overall") or "").upper() or None

    accuracy_data = (
        _load_json(accuracy_summary)
        if accuracy_summary and accuracy_summary.is_file()
        else None
    )
    perf_data = (
        _load_json(on_device_performance_json)
        if on_device_performance_json and on_device_performance_json.is_file()
        else None
    )
    xctest_metrics_data = (
        _load_json(xctest_metrics_json)
        if xctest_metrics_json and xctest_metrics_json.is_file()
        else None
    )

    head_lines: List[str] = ["# iOS demo — benchmark report", ""]
    if overall:
        head_lines.append(f"**Overall:** {overall}")
        head_lines.append("")

    parts: List[str] = ["\n".join(head_lines)]

    if status:
        parts.append(_run_status_section(status))
        parts.append("")

    perf_banner = _performance_section_banner(overall)

    accuracy_section = _accuracy_section(accuracy_data, status)
    if accuracy_section:
        parts.append(accuracy_section)
        parts.append("")

    parts.append("## On-device runtime performance")
    parts.append("")
    if perf_banner:
        parts.append(perf_banner)
    if perf_data is not None:
        parts.append(_on_device_section(perf_data, xctest_metrics_data, status))
    else:
        parts.append(
            _placeholder_from_status(status, "extract-artifacts")
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
    parts.append("*Generated by `scripts/generate_benchmark_report.py`.*")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=root / "out" / "benchmark-report.md"
    )
    parser.add_argument("--on-device-performance-json", type=Path, default=None)
    parser.add_argument("--accuracy-summary", type=Path, default=None)
    parser.add_argument("--run-status", type=Path, default=None)
    parser.add_argument("--xctest-metrics-json", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    generate_report(
        out_path=args.output.resolve(),
        on_device_performance_json=args.on_device_performance_json,
        accuracy_summary=args.accuracy_summary,
        run_status=args.run_status,
        xctest_metrics_json=args.xctest_metrics_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
