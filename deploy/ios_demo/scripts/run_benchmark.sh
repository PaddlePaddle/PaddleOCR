#!/usr/bin/env bash
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
#
# End-to-end benchmark runner for the iOS demo.
#
# Pipeline: preflight → resolve-image → resolve-destination →
#           optional accuracy-precheck → benchmark xcodebuild-test → extract-artifacts → report
#
# Requires: bash 3.2+, xcodebuild, xcrun, python3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IOS_DEMO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Fixtures/: images shipped for benchmark runs; `local-*` entries come from --image; auto-pick ignores `local-*`.
FIXTURES_DIR="${IOS_DEMO_ROOT}/PaddleOCRDemoTests/Fixtures"
DEFAULT_SIMULATOR="iPhone 16"
BUILD_CONFIGURATION="Release"
# Comma-separated xcodebuild -only-testing targets.
TESTING_SCOPE="${PADDLEOCR_BENCHMARK_ONLY_TESTING_SCOPE:-PaddleOCRDemoTests/OCRBenchmarkTests/testOnDeviceLatencyBenchmark,PaddleOCRDemoTests/OCRBenchmarkTests/testOnDeviceMemoryBenchmark}"
XCODE_TEST_SINGLE_RUN_FLAGS=(
  -parallel-testing-enabled NO
  -collect-test-diagnostics never
)

UDID=""
SIMULATOR=""
IMAGE=""
FIXTURE_ARG=""
WARMUP_CLI=""
MEASURED_CLI=""
ORT_EP_CLI=""
THREADS_CLI=""
XNNPACK_THREADS_CLI=""
REC_BATCH_SIZE_CLI=""
COREML_OPTIONS_CLI=""
ORT_PROFILING_CLI=0
ACCURACY_CHECK=0
ACCURACY_STOP_ON_FAILURE=0
ACCURACY_REFERENCE_JSON=""
OUT_DIR="${IOS_DEMO_ROOT}/out"
CLEAN=0

usage() {
  cat <<EOF
Usage: ./scripts/run_benchmark.sh [OPTIONS]

Options:
  --udid <id>           Real-device UDID (preferred real-device path)
  --simulator <name>    Simulator name (default: ${DEFAULT_SIMULATOR})
  --image <path>        Ad-hoc benchmark image; copied as Fixtures/local-<basename>.
                        Without --image/--fixture, see image selection below.
  --fixture <name>      Use a file already under PaddleOCRDemoTests/Fixtures/ (stem or
                        file name, e.g. ios_ocr_benchmark_reference or .jpg). Useful when
                        several fixtures exist.
  --warmup <n>          Benchmark warmup iterations (non-negative int).
  --measured-iterations <n>  Timed benchmark iterations.
  --ort-execution-provider <NAME>  Preferred ONNX Runtime EP preset for tests:
                        CPU, XNNPACK, or CORE_ML.
  --threads <n>         Session-level intra-op thread count (positive int).
  --xnnpack-threads <n> XNNPACK EP thread count (positive int).
  --rec-batch-size <n>  Recognition batch size override (positive int).
  --coreml-options <flags>  Comma-separated CoreML EP flags: enableOnSubgraphs,
                        createMLProgram, staticInputShapes, cpuOnly, cpuAndGPU, aneOnly.
  --ort-profiling       Enable ONNX Runtime session profiling.
  --accuracy-check      Run an accuracy precheck before benchmark tests.
  --stop-on-accuracy-failure
                        With --accuracy-check, treat accuracy FAIL as blocking.
  --accuracy-reference-json <path>
                        Existing OCR reference JSON for --accuracy-check. If omitted,
                        the script generates accuracy-reference.json under --output-dir.
  --output-dir <dir>       Output directory (default: out/)
  --clean               Delete Fixtures/local-* and prior non-log artifacts under <output-dir>
                        before running.
  -h, --help            Show help

Environment (optional; CLI wins when both are set):
  PADDLEOCR_BENCHMARK_IMAGE_NAME   Select which bundled fixture to use (same as --fixture)
  PADDLEOCR_BENCHMARK_WARMUP_ITERATIONS
  PADDLEOCR_BENCHMARK_MEASURED_ITERATIONS
  PADDLEOCR_BENCHMARK_ORT_EXECUTION_PROVIDER  CPU, XNNPACK, or CORE_ML
  PADDLEOCR_BENCHMARK_INTRA_OP_THREADS    Session-level thread count
  PADDLEOCR_BENCHMARK_XNNPACK_THREADS     XNNPACK EP thread count
  PADDLEOCR_BENCHMARK_REC_BATCH_SIZE      Recognition batch size
  PADDLEOCR_BENCHMARK_COREML_OPTIONS      Comma-separated CoreML EP flags
  PADDLEOCR_BENCHMARK_ORT_PROFILING   Set to 1/true/yes/on to emit ORT JSON profiles
  PADDLEOCR_BENCHMARK_ONLY_TESTING_SCOPE   Optional comma-separated xcodebuild -only-testing scopes.

Image selection when --image is not used:
  1) --fixture <name>, if given, else PADDLEOCR_BENCHMARK_IMAGE_NAME, if set
  2) otherwise exactly one non-local-* file in Fixtures/
EOF
}

die() {
  printf "[run_benchmark] ERROR: %s\n" "$1" >&2
  printf "  Where: %s\n" "$2" >&2
  printf "  Next:  %s\n" "$3" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --udid) UDID="$2"; shift 2 ;;
    --simulator) SIMULATOR="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --fixture) FIXTURE_ARG="$2"; shift 2 ;;
    --warmup) WARMUP_CLI="$2"; shift 2 ;;
    --measured-iterations) MEASURED_CLI="$2"; shift 2 ;;
    --ort-execution-provider) ORT_EP_CLI="$2"; shift 2 ;;
    --threads) THREADS_CLI="$2"; shift 2 ;;
    --xnnpack-threads) XNNPACK_THREADS_CLI="$2"; shift 2 ;;
    --rec-batch-size) REC_BATCH_SIZE_CLI="$2"; shift 2 ;;
    --coreml-options) COREML_OPTIONS_CLI="$2"; shift 2 ;;
    --ort-profiling) ORT_PROFILING_CLI=1; shift ;;
    --accuracy-check) ACCURACY_CHECK=1; shift ;;
    --stop-on-accuracy-failure) ACCURACY_STOP_ON_FAILURE=1; shift ;;
    --accuracy-reference-json) ACCURACY_REFERENCE_JSON="$2"; shift 2 ;;
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" "argument parsing" "See --help." ;;
  esac
done

if [[ "${ACCURACY_STOP_ON_FAILURE}" -eq 1 ]]; then
  [[ "${ACCURACY_CHECK}" -eq 1 ]] \
    || die "--stop-on-accuracy-failure requires --accuracy-check" "argument parsing" "Add --accuracy-check, or remove --stop-on-accuracy-failure."
fi

if [[ -n "${ACCURACY_REFERENCE_JSON}" ]]; then
  [[ "${ACCURACY_CHECK}" -eq 1 ]] \
    || die "--accuracy-reference-json requires --accuracy-check" "argument parsing" "Add --accuracy-check, or remove --accuracy-reference-json."
  [[ -f "${ACCURACY_REFERENCE_JSON}" ]] \
    || die "Accuracy reference JSON not found: ${ACCURACY_REFERENCE_JSON}" "argument parsing" "Pass an existing OCR JSON reference file."
fi

# Fixture label: --fixture overrides PADDLEOCR_BENCHMARK_IMAGE_NAME.
EXPLICIT_FIXTURE="${FIXTURE_ARG:-${PADDLEOCR_BENCHMARK_IMAGE_NAME:-}}"

# Warmup / measured: CLI over env; forward to xcodebuild only when non-empty.
WARMUP_MERGED="${WARMUP_CLI:-${PADDLEOCR_BENCHMARK_WARMUP_ITERATIONS:-}}"
MEASURED_MERGED="${MEASURED_CLI:-${PADDLEOCR_BENCHMARK_MEASURED_ITERATIONS:-}}"
if [[ -n "${WARMUP_MERGED}" ]]; then
  [[ "${WARMUP_MERGED}" =~ ^[0-9]+$ ]] \
    || die "Invalid --warmup / PADDLEOCR_BENCHMARK_WARMUP_ITERATIONS: ${WARMUP_MERGED}" "argument parsing" "Use a non-negative integer."
fi
if [[ -n "${MEASURED_MERGED}" ]]; then
  [[ "${MEASURED_MERGED}" =~ ^[0-9]+$ ]] \
    || die "Invalid --measured-iterations / PADDLEOCR_BENCHMARK_MEASURED_ITERATIONS: ${MEASURED_MERGED}" "argument parsing" "Use a non-negative integer."
fi

# ORT execution provider preset: CPU, XNNPACK, or CORE_ML; forward when set (flag or env).
ORT_EP_MERGED="${ORT_EP_CLI:-${PADDLEOCR_BENCHMARK_ORT_EXECUTION_PROVIDER:-}}"
ORT_EP_CANON=""
if [[ -n "${ORT_EP_MERGED}" ]]; then
  _ort_ep_lc="$(printf '%s' "${ORT_EP_MERGED}" | tr '[:upper:]' '[:lower:]')"
  case "${_ort_ep_lc}" in
    core_ml) ORT_EP_CANON="CORE_ML" ;;
    xnnpack) ORT_EP_CANON="XNNPACK" ;;
    cpu) ORT_EP_CANON="CPU" ;;
    *)
      die "Invalid --ort-execution-provider / PADDLEOCR_BENCHMARK_ORT_EXECUTION_PROVIDER: ${ORT_EP_MERGED}" "argument parsing" "Use CPU, XNNPACK, or CORE_ML."
      ;;
  esac
fi

# Tuning parameters.
THREADS_MERGED="${THREADS_CLI:-${PADDLEOCR_BENCHMARK_INTRA_OP_THREADS:-}}"
XNNPACK_THREADS_MERGED="${XNNPACK_THREADS_CLI:-${PADDLEOCR_BENCHMARK_XNNPACK_THREADS:-}}"
REC_BATCH_SIZE_MERGED="${REC_BATCH_SIZE_CLI:-${PADDLEOCR_BENCHMARK_REC_BATCH_SIZE:-}}"
COREML_OPTIONS_MERGED="${COREML_OPTIONS_CLI:-${PADDLEOCR_BENCHMARK_COREML_OPTIONS:-}}"
if [[ -n "${THREADS_MERGED}" ]]; then
  [[ "${THREADS_MERGED}" =~ ^[0-9]+$ && "${THREADS_MERGED}" -gt 0 ]] \
    || die "Invalid --threads: ${THREADS_MERGED}" "argument parsing" "Use a positive integer."
fi
if [[ -n "${XNNPACK_THREADS_MERGED}" ]]; then
  [[ "${XNNPACK_THREADS_MERGED}" =~ ^[0-9]+$ && "${XNNPACK_THREADS_MERGED}" -gt 0 ]] \
    || die "Invalid --xnnpack-threads: ${XNNPACK_THREADS_MERGED}" "argument parsing" "Use a positive integer."
fi
if [[ -n "${REC_BATCH_SIZE_MERGED}" ]]; then
  [[ "${REC_BATCH_SIZE_MERGED}" =~ ^[0-9]+$ && "${REC_BATCH_SIZE_MERGED}" -gt 0 ]] \
    || die "Invalid --rec-batch-size: ${REC_BATCH_SIZE_MERGED}" "argument parsing" "Use a positive integer."
fi

if [[ "${ORT_PROFILING_CLI}" -eq 1 ]]; then
  ORT_PROFILING_MERGED="1"
else
  ORT_PROFILING_MERGED="${PADDLEOCR_BENCHMARK_ORT_PROFILING:-}"
fi

LOGS_DIR="${OUT_DIR}/logs"
STATUS_PATH="${OUT_DIR}/run-status.json"
REPORT_PATH="${OUT_DIR}/benchmark-report.md"
XCTEST_METRICS_PATH="${OUT_DIR}/xctest-memory-metrics.json"
ACCURACY_RESULT_PATH="${OUT_DIR}/accuracy-result.xcresult"
LATENCY_RESULT_PATH="${OUT_DIR}/latency-result.xcresult"
MEMORY_RESULT_PATH="${OUT_DIR}/memory-result.xcresult"
ACCURACY_REFERENCE_OUT="${OUT_DIR}/accuracy-reference.json"
ACCURACY_IOS_EXPORT_PATH="${OUT_DIR}/ios-ocr-export.json"
ACCURACY_SUMMARY_PATH="${OUT_DIR}/accuracy-summary.json"

mkdir -p "${OUT_DIR}" "${LOGS_DIR}"

if [[ "${CLEAN}" -eq 1 ]]; then
  rm -f "${FIXTURES_DIR}"/local-*
  find "${OUT_DIR}" -mindepth 1 -not -path "${LOGS_DIR}" -not -path "${LOGS_DIR}/*" -exec rm -rf {} + 2>/dev/null || true
  mkdir -p "${LOGS_DIR}"
fi

# ---------- Step state (parallel indexed arrays) ----------

STEP_NAMES=(preflight resolve-image resolve-destination accuracy-precheck xcodebuild-test extract-artifacts report)
STEP_STATUS=()
STEP_DURATION_MS=()
STEP_EXIT=()
STEP_REASON=()
STEP_LOG=()

for i in "${!STEP_NAMES[@]}"; do
  STEP_STATUS[$i]=pending
  STEP_DURATION_MS[$i]=""
  STEP_EXIT[$i]=""
  STEP_REASON[$i]=""
  STEP_LOG[$i]=""
done

step_index() {
  local want=$1 i
  for i in "${!STEP_NAMES[@]}"; do
    [[ "${STEP_NAMES[$i]}" == "$want" ]] && { echo "$i"; return 0; }
  done
  return 1
}

step_status_set() { local i; i=$(step_index "$1") || return 1; STEP_STATUS[$i]="$2"; }
step_status_get() { local i; i=$(step_index "$1") || return 1; echo "${STEP_STATUS[$i]}"; }
step_duration_set() { local i; i=$(step_index "$1") || return 1; STEP_DURATION_MS[$i]="$2"; }
step_exit_set() { local i; i=$(step_index "$1") || return 1; STEP_EXIT[$i]="$2"; }
step_reason_set() { local i; i=$(step_index "$1") || return 1; STEP_REASON[$i]="$2"; }
step_log_set() { local i; i=$(step_index "$1") || return 1; STEP_LOG[$i]="$2"; }

HALTED=0
HALT_REASON=""
CURRENT_STEP_NAME=""
CURRENT_STEP_START_MS=""
ACTIVE_CHILD_PID=""
ACTIVE_CHILD_LABEL=""
IMAGE_NAME=""
IMAGE_SRC=""
IMAGE_SOURCE=""
DEST=""
IS_SIMULATOR=1
RUN_STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ACCURACY_STATUS="SKIPPED"
ACCURACY_REFERENCE_USED=""
ACCURACY_SUMMARY_USED=""
ACCURACY_REASON=""
MODEL_PRESET="unknown"
DET_MODEL_PATH=""
DET_MODEL_FORMAT=""
DET_MODEL_SIZE_BYTES=""
REC_MODEL_PATH=""
REC_MODEL_FORMAT=""
REC_MODEL_SIZE_BYTES=""
TOTAL_MODEL_SIZE_BYTES=""
APP_BINARY_SIZE_BYTES=""

now_ms() {
  python3 -c 'import time; print(int(time.time()*1000))'
}

file_size_bytes() {
  [[ -f "$1" ]] || { echo ""; return 0; }
  stat -f%z "$1" 2>/dev/null || echo ""
}

first_model_file() {
  local dir="$1" f matches
  for f in "${dir}/inference.ort" "${dir}/inference.with_runtime_opt.ort"; do
    [[ -f "$f" ]] && { echo "$f"; return 0; }
  done
  matches=("${dir}"/inference*.ort)
  if [[ -f "${matches[0]:-}" ]]; then
    printf "%s\n" "${matches[@]}" | sort | head -n 1
    return 0
  fi
  [[ -f "${dir}/inference.onnx" ]] && { echo "${dir}/inference.onnx"; return 0; }
  echo ""
}

yml_model_name() {
  local path="$1"
  python3 - "$path" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("")
    raise SystemExit(0)
for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    m = re.match(r"\s*model_name:\s*['\"]?([^'\"]+?)['\"]?\s*$", line)
    if m:
        print(m.group(1).strip())
        break
else:
    print("")
PY
}

infer_model_preset() {
  local det_name="$1" rec_name="$2" joined
  joined="$(printf "%s %s" "$det_name" "$rec_name" | tr '[:upper:]' '[:lower:]')"
  case "$joined" in
    *small*) echo "PP-OCRv6_small" ;;
    *tiny*) echo "PP-OCRv6_tiny" ;;
    *mobile*) echo "PP-OCRv5_mobile" ;;
    *) echo "unknown" ;;
  esac
}

collect_model_metadata() {
  local det_name rec_name
  DET_MODEL_PATH="$(first_model_file "${IOS_DEMO_ROOT}/PaddleOCRDemo/Models/det")"
  REC_MODEL_PATH="$(first_model_file "${IOS_DEMO_ROOT}/PaddleOCRDemo/Models/rec")"
  DET_MODEL_FORMAT="${DET_MODEL_PATH##*.}"
  REC_MODEL_FORMAT="${REC_MODEL_PATH##*.}"
  DET_MODEL_SIZE_BYTES="$(file_size_bytes "$DET_MODEL_PATH")"
  REC_MODEL_SIZE_BYTES="$(file_size_bytes "$REC_MODEL_PATH")"
  if [[ -n "$DET_MODEL_SIZE_BYTES" && -n "$REC_MODEL_SIZE_BYTES" ]]; then
    TOTAL_MODEL_SIZE_BYTES=$((DET_MODEL_SIZE_BYTES + REC_MODEL_SIZE_BYTES))
  fi
  det_name="$(yml_model_name "${IOS_DEMO_ROOT}/PaddleOCRDemo/Models/det/inference.yml")"
  rec_name="$(yml_model_name "${IOS_DEMO_ROOT}/PaddleOCRDemo/Models/rec/inference.yml")"
  MODEL_PRESET="$(infer_model_preset "$det_name" "$rec_name")"
}

collect_app_binary_size() {
  local settings target_build_dir executable_path app_binary
  settings="$(xcodebuild -showBuildSettings -workspace "${IOS_DEMO_ROOT}/PaddleOCRDemo.xcworkspace" -scheme PaddleOCRDemo -configuration "${BUILD_CONFIGURATION}" -destination "${DEST}" 2>/dev/null || true)"
  target_build_dir="$(printf "%s\n" "$settings" | awk -F'= ' '/ TARGET_BUILD_DIR = / {print $2; exit}')"
  executable_path="$(printf "%s\n" "$settings" | awk -F'= ' '/ EXECUTABLE_PATH = / {print $2; exit}')"
  if [[ -n "$target_build_dir" && -n "$executable_path" ]]; then
    app_binary="${target_build_dir}/${executable_path}"
    APP_BINARY_SIZE_BYTES="$(file_size_bytes "$app_binary")"
  fi
}

record_step_ok() {
  local name="$1" duration="$2"
  step_status_set "$name" ok
  step_duration_set "$name" "$duration"
  step_exit_set "$name" 0
  printf "  -> OK (%sms)\n" "$duration"
}

record_step_fail() {
  local name="$1" duration="$2" rc="$3" reason="$4"
  step_status_set "$name" fail
  step_duration_set "$name" "$duration"
  step_exit_set "$name" "$rc"
  step_reason_set "$name" "$reason"
  step_log_set "$name" "logs/${name}.log"
  HALTED=1
  HALT_REASON="$reason"
  printf "  -> ERROR (%sms, exit %s): %s\n" "$duration" "$rc" "$reason" >&2
}

terminate_process_tree() {
  local pid="$1" sig="$2" child
  while IFS= read -r child; do
    [[ -n "$child" ]] && terminate_process_tree "$child" "$sig"
  done < <(pgrep -P "$pid" 2>/dev/null || true)
  kill "-${sig}" "$pid" 2>/dev/null || true
}

handle_signal() {
  local sig="$1" rc=130 reason end duration
  [[ "$sig" == "TERM" ]] && rc=143
  trap - INT TERM
  reason="interrupted by SIG${sig}"
  printf "\n[run_benchmark] %s\n" "$reason" >&2
  if [[ -n "${ACTIVE_CHILD_PID}" ]] && kill -0 "${ACTIVE_CHILD_PID}" 2>/dev/null; then
    printf "[run_benchmark] stopping %s (pid %s)\n" "${ACTIVE_CHILD_LABEL:-active child}" "${ACTIVE_CHILD_PID}" >&2
    terminate_process_tree "${ACTIVE_CHILD_PID}" "$sig"
    sleep 1
    if kill -0 "${ACTIVE_CHILD_PID}" 2>/dev/null; then
      terminate_process_tree "${ACTIVE_CHILD_PID}" TERM
    fi
  fi
  if [[ -n "${CURRENT_STEP_NAME}" ]]; then
    end="$(now_ms)"; duration=$((end - CURRENT_STEP_START_MS))
    record_step_fail "${CURRENT_STEP_NAME}" "$duration" "$rc" "$reason"
    mark_remaining_skipped "$reason"
  else
    HALTED=1
    HALT_REASON="$reason"
  fi
  exit "$rc"
}

mark_remaining_skipped() {
  local reason="$1" skipped=0 i st
  for i in "${!STEP_NAMES[@]}"; do
    st="${STEP_STATUS[$i]}"
    if [[ "$st" == "pending" || "$st" == "running" ]]; then
      if [[ "${STEP_NAMES[$i]}" != "report" ]]; then
        STEP_STATUS[$i]="skipped"
        STEP_REASON[$i]="$reason"
        skipped=1
      fi
    fi
  done
  [[ $skipped -eq 1 ]] && echo "  (remaining steps marked skipped)" >&2 || true
}

write_status_json() {
  local overall="$1" exit_code="$2"
  local finished
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 - "$STATUS_PATH" "$overall" "$exit_code" "$RUN_STARTED" "$finished" \
    "$IMAGE_NAME" "$IMAGE_SOURCE" "$DEST" "$BUILD_CONFIGURATION" "$TESTING_SCOPE" \
    "$IS_SIMULATOR" "${ORT_EP_CANON:-CPU}" "${WARMUP_MERGED:-}" "${MEASURED_MERGED:-}" \
    "$MODEL_PRESET" "$DET_MODEL_FORMAT" "$DET_MODEL_SIZE_BYTES" "$REC_MODEL_FORMAT" "$REC_MODEL_SIZE_BYTES" "$TOTAL_MODEL_SIZE_BYTES" "$APP_BINARY_SIZE_BYTES" \
    "${ORT_PROFILING_MERGED:-}" "$ACCURACY_CHECK" "$ACCURACY_STATUS" \
    "$ACCURACY_STOP_ON_FAILURE" "$ACCURACY_REFERENCE_USED" "$ACCURACY_IOS_EXPORT_PATH" "$ACCURACY_SUMMARY_USED" "$ACCURACY_REASON" \
    "${#STEP_NAMES[@]}" \
    "${STEP_NAMES[@]}" \
    "${STEP_STATUS[@]}" \
    "${STEP_DURATION_MS[@]}" \
    "${STEP_EXIT[@]}" \
    "${STEP_REASON[@]}" \
    "${STEP_LOG[@]}" <<'PY'
import json
import sys

args = sys.argv[1:]
(
    path,
    overall,
    exit_code,
    started,
    finished,
    image_name,
    image_source,
    destination,
    build_configuration,
    testing_scope,
    is_simulator,
    ort_ep_preset,
    warmup_iterations,
    measured_iterations,
    model_preset,
    det_model_format,
    det_model_size_bytes,
    rec_model_format,
    rec_model_size_bytes,
    total_model_size_bytes,
    app_binary_size_bytes,
    ort_profiling,
    accuracy_enabled,
    accuracy_status,
    accuracy_stop_on_failure,
    accuracy_reference,
    accuracy_ios_export,
    accuracy_summary,
    accuracy_reason,
) = args[:29]
n = int(args[29])
base = 30
names  = args[base + 0 * n : base + 1 * n]
status = args[base + 1 * n : base + 2 * n]
dur    = args[base + 2 * n : base + 3 * n]
exitc  = args[base + 3 * n : base + 4 * n]
reason = args[base + 4 * n : base + 5 * n]
logp   = args[base + 5 * n : base + 6 * n]
steps = []
for i in range(n):
    row = {"name": names[i], "status": status[i] or "pending"}
    if dur[i]:
        row["durationMs"] = int(dur[i])
    if exitc[i]:
        row["exitCode"] = int(exitc[i])
    if reason[i]:
        row["reason"] = reason[i]
    if logp[i]:
        row["logPath"] = logp[i]
    steps.append(row)
doc = {
    "schemaVersion": 1,
    "overall": overall,
    "exitCode": int(exit_code),
    "image": {"name": image_name, "source": image_source},
    "destination": destination,
    "benchmark": {
        "buildConfiguration": build_configuration,
        "testingScope": testing_scope,
        "isSimulator": is_simulator == "1",
        "ortExecutionProvider": ort_ep_preset,
        "warmupIterations": int(warmup_iterations) if warmup_iterations else None,
        "measuredIterations": int(measured_iterations) if measured_iterations else None,
        "modelPreset": model_preset or "unknown",
        "modelFiles": {
            "detection": {
                "format": det_model_format or None,
                "sizeBytes": int(det_model_size_bytes) if det_model_size_bytes else None,
            },
            "recognition": {
                "format": rec_model_format or None,
                "sizeBytes": int(rec_model_size_bytes) if rec_model_size_bytes else None,
            },
            "totalSizeBytes": int(total_model_size_bytes) if total_model_size_bytes else None,
        },
        "appBinarySizeBytes": int(app_binary_size_bytes) if app_binary_size_bytes else None,
        "ortProfilingEnabled": ort_profiling.lower() in ("1", "true", "yes", "on"),
    },
    "accuracy": {
        "enabled": accuracy_enabled == "1",
        "status": accuracy_status,
        "stopOnFailure": accuracy_stop_on_failure == "1",
        "referenceJson": accuracy_reference or None,
        "iosExportJson": accuracy_ios_export if accuracy_enabled == "1" else None,
        "summaryJson": accuracy_summary or None,
        "reason": accuracy_reason or None,
    },
    "runStartedAt": started,
    "runFinishedAt": finished,
    "steps": steps,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(doc, f, indent=2)
PY
}

finish() {
  local exit_code=$?
  local overall
  if [[ "${HALTED}" -eq 1 ]]; then
    overall="ERROR"
  else
    overall="COMPLETED"
    exit_code=0
  fi
  printf "\n===== SUMMARY =====\n"
  printf "Overall:   %s\n" "${overall}"
  printf "Exit code: %s\n" "${exit_code}"
  if [[ "${overall}" == "COMPLETED" ]]; then
    printf "Status:    %s\n" "${STATUS_PATH}"
    printf "Report:    %s\n" "${REPORT_PATH}"
  else
    printf "ERROR: See full logs in %s\n" "${LOGS_DIR}"
  fi
  exit "${exit_code}"
}
trap finish EXIT
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

run_xcodebuild() {
  local rc
  ACTIVE_CHILD_LABEL="$*"
  "$@" &
  ACTIVE_CHILD_PID=$!
  set +e
  wait "${ACTIVE_CHILD_PID}"
  rc=$?
  set -e
  ACTIVE_CHILD_PID=""
  ACTIVE_CHILD_LABEL=""
  return "$rc"
}

run_step() {
  local name="$1"; shift
  local log="${LOGS_DIR}/${name}.log"
  local start end duration rc tail
  printf "\n===== [%s] step %s: %s =====\n" "$(date +%H:%M:%S)" "$name" "$*"
  step_status_set "$name" running
  start="$(now_ms)"
  CURRENT_STEP_NAME="$name"
  CURRENT_STEP_START_MS="$start"
  if "$@" >"$log" 2>&1 </dev/null; then
    CURRENT_STEP_NAME=""
    CURRENT_STEP_START_MS=""
    end="$(now_ms)"; duration=$((end - start))
    record_step_ok "$name" "$duration"
    return 0
  else
    rc=$?
    CURRENT_STEP_NAME=""
    CURRENT_STEP_START_MS=""
    end="$(now_ms)"; duration=$((end - start))
    tail="$(tail -n 5 "$log" | head -n 1 || true)"
    record_step_fail "$name" "$duration" "$rc" "${tail:-step failed}"
    return "$rc"
  fi
}

generate_report_impl() {
  python3 "${SCRIPT_DIR}/generate_benchmark_report.py" \
    --accuracy-summary "${ACCURACY_SUMMARY_PATH}" \
    --on-device-performance-json "${OUT_DIR}/on-device-performance.json" \
    --run-status "${STATUS_PATH}" \
    --xctest-metrics-json "${XCTEST_METRICS_PATH}" \
    --output "${REPORT_PATH}"
}

# ---------- Step: preflight ----------
preflight_impl() {
  [[ -f "${IOS_DEMO_ROOT}/PaddleOCRDemo/Models/det/inference.yml" ]] \
    || { echo "Models/det/inference.yml missing; run ./scripts/fetch_ios_demo_models.sh" >&2; return 1; }
  [[ -f "${IOS_DEMO_ROOT}/PaddleOCRDemo/Models/rec/inference.yml" ]] \
    || { echo "Models/rec/inference.yml missing; run ./scripts/fetch_ios_demo_models.sh" >&2; return 1; }
  command -v xcodebuild >/dev/null || { echo "xcodebuild not in PATH" >&2; return 1; }
  command -v xcrun >/dev/null || { echo "xcrun not in PATH" >&2; return 1; }
  command -v python3 >/dev/null || { echo "python3 not in PATH" >&2; return 1; }
  [[ -d "${FIXTURES_DIR}" ]] || { echo "Fixtures/ directory missing" >&2; return 1; }
  collect_model_metadata
}
run_step preflight preflight_impl || { mark_remaining_skipped "preflight failed: ${HALT_REASON}"; exit 1; }

# ---------- Step: resolve-image ----------
resolve_image_impl() {
  if [[ -n "$IMAGE" ]]; then
    [[ -f "$IMAGE" ]] || { echo "--image file not found: $IMAGE" >&2; return 1; }
    local base sanitized
    base="$(basename "$IMAGE")"
    sanitized="$(echo "$base" | tr -c 'A-Za-z0-9._-' '_')"
    IMAGE_NAME="local-${sanitized}"
    IMAGE_SRC="$IMAGE"
    IMAGE_SOURCE="override"
    cp -f "$IMAGE" "${FIXTURES_DIR}/${IMAGE_NAME}"
    echo "Copied override image to ${FIXTURES_DIR}/${IMAGE_NAME}"
    return 0
  fi
  if [[ -n "${EXPLICIT_FIXTURE}" ]]; then
    local name cand
    name="${EXPLICIT_FIXTURE}"
    cand=""
    if [[ -f "${FIXTURES_DIR}/${name}" ]]; then
      cand="${FIXTURES_DIR}/${name}"
    else
      # stem without extension: try common image extensions
      if [[ "$name" != *.* ]]; then
        for ext in jpg jpeg png heic webp; do
          if [[ -f "${FIXTURES_DIR}/${name}.${ext}" ]]; then
            cand="${FIXTURES_DIR}/${name}.${ext}"
            break
          fi
        done
      fi
    fi
    [[ -n "$cand" ]] || { echo "Fixture not found for --fixture / PADDLEOCR_BENCHMARK_IMAGE_NAME: ${name}" >&2; return 1; }
    IMAGE_NAME="$(basename "$cand")"
    IMAGE_SRC="$cand"
    IMAGE_SOURCE="fixture"
    echo "Using fixture (explicit): ${IMAGE_NAME}"
    return 0
  fi
  local found=()
  while IFS= read -r -d '' f; do
    local b; b="$(basename "$f")"
    case "$b" in
      local-*|.gitignore|.*) continue ;;
      *) found+=("$b") ;;
    esac
  done < <(find "${FIXTURES_DIR}" -mindepth 1 -maxdepth 1 -type f -print0)
  if [[ ${#found[@]} -eq 0 ]]; then
    echo "No fixture in ${FIXTURES_DIR}/ and no --image provided." >&2
    echo "Put exactly one fixture image in PaddleOCRDemoTests/Fixtures/, or pass --image <path>." >&2
    return 1
  fi
  if [[ ${#found[@]} -gt 1 ]]; then
    echo "Multiple fixtures in ${FIXTURES_DIR}/: ${found[*]}" >&2
    echo "Pass --image <path> to select one explicitly." >&2
    return 1
  fi
  IMAGE_NAME="${found[0]}"
  IMAGE_SRC="${FIXTURES_DIR}/${IMAGE_NAME}"
  IMAGE_SOURCE="fixture"
  echo "Using fixture: ${IMAGE_NAME}"
}
run_step resolve-image resolve_image_impl || { mark_remaining_skipped "resolve-image failed: ${HALT_REASON}"; exit 1; }

# ---------- Step: resolve-destination ----------
resolve_destination_impl() {
  if [[ -n "${UDID}" ]]; then
    DEST="platform=iOS,id=${UDID}"
    IS_SIMULATOR=0
  elif [[ -n "${SIMULATOR}" ]]; then
    DEST="platform=iOS Simulator,name=${SIMULATOR}"
    IS_SIMULATOR=1
  else
    DEST="platform=iOS Simulator,name=${DEFAULT_SIMULATOR}"
    IS_SIMULATOR=1
  fi
  echo "Destination: ${DEST}"
}
run_step resolve-destination resolve_destination_impl || { mark_remaining_skipped "resolve-destination failed: ${HALT_REASON}"; exit 1; }

# ---------- Step: accuracy-precheck (skipped unless --accuracy-check is set) ----------
accuracy_precheck_impl() {
  ACCURACY_STATUS="ERROR"
  ACCURACY_REASON=""
  ACCURACY_SUMMARY_USED="${ACCURACY_SUMMARY_PATH}"
  rm -rf "${ACCURACY_RESULT_PATH}"
  rm -f "${ACCURACY_IOS_EXPORT_PATH}" "${ACCURACY_SUMMARY_PATH}"

  if [[ -n "${ACCURACY_REFERENCE_JSON}" ]]; then
    ACCURACY_REFERENCE_USED="${ACCURACY_REFERENCE_JSON}"
  else
    ACCURACY_REFERENCE_USED="${ACCURACY_REFERENCE_OUT}"
    if ! python3 "${SCRIPT_DIR}/ocr_reference_run.py" \
      --image "${IMAGE_SRC}" \
      --output "${ACCURACY_REFERENCE_OUT}" \
      --device cpu \
      --align-ios-defaults; then
      ACCURACY_STATUS="ERROR"
      ACCURACY_REASON="reference generation failed"
      echo "Accuracy precheck: ${ACCURACY_STATUS} (${ACCURACY_REASON})" >&2
      return 1
    fi
  fi

  local runenv=(
    env "TEST_RUNNER_PADDLEOCR_BENCHMARK_IMAGE_NAME=${IMAGE_NAME}"
    "TEST_RUNNER_PADDLEOCR_BENCHMARK_BUILD_CONFIGURATION=${BUILD_CONFIGURATION}"
  )
  if [[ -n "${ORT_EP_CANON}" ]]; then
    runenv+=("TEST_RUNNER_PADDLEOCR_BENCHMARK_ORT_EXECUTION_PROVIDER=${ORT_EP_CANON}")
  fi

  if ! run_xcodebuild "${runenv[@]}" xcodebuild test \
    -workspace "${IOS_DEMO_ROOT}/PaddleOCRDemo.xcworkspace" \
    -scheme PaddleOCRDemo \
    -configuration "${BUILD_CONFIGURATION}" \
    -destination "${DEST}" \
    -resultBundlePath "${ACCURACY_RESULT_PATH}" \
    "${XCODE_TEST_SINGLE_RUN_FLAGS[@]}" \
    -only-testing:PaddleOCRDemoTests/OCRBenchmarkTests/testOCRExportJSONSchema; then
    ACCURACY_STATUS="ERROR"
    ACCURACY_REASON="accuracy XCTest export failed"
    echo "Accuracy precheck: ${ACCURACY_STATUS} (${ACCURACY_REASON})" >&2
    return 1
  fi

  if ! python3 "${SCRIPT_DIR}/extract_xcresult_attachments.py" \
    --result "${ACCURACY_RESULT_PATH}" \
    --output-dir "${OUT_DIR}" \
    --name ios-ocr-export.json; then
    ACCURACY_STATUS="ERROR"
    ACCURACY_REASON="accuracy attachment extraction failed"
    echo "Accuracy precheck: ${ACCURACY_STATUS} (${ACCURACY_REASON})" >&2
    return 1
  fi

  set +e
  python3 "${SCRIPT_DIR}/compare_ocr_json.py" \
    "${ACCURACY_REFERENCE_USED}" \
    "${ACCURACY_IOS_EXPORT_PATH}" \
    --json-summary-out "${ACCURACY_SUMMARY_PATH}"
  local compare_rc=$?
  set -e
  if [[ "${compare_rc}" -eq 0 ]]; then
    ACCURACY_STATUS="PASS"
  elif [[ -f "${ACCURACY_SUMMARY_PATH}" ]]; then
    ACCURACY_STATUS="FAIL"
    ACCURACY_REASON="accuracy thresholds were not met"
  else
    ACCURACY_STATUS="ERROR"
    ACCURACY_REASON="accuracy comparison failed"
  fi
  echo "Accuracy precheck: ${ACCURACY_STATUS}${ACCURACY_REASON:+ (${ACCURACY_REASON})}"
  [[ "${ACCURACY_STATUS}" != "ERROR" ]]
}

if [[ "${ACCURACY_CHECK}" -eq 1 ]]; then
  if ! run_step accuracy-precheck accuracy_precheck_impl; then
    HALT_REASON="accuracy precheck error: ${ACCURACY_REASON:-${HALT_REASON}}"
    step_reason_set accuracy-precheck "${HALT_REASON}"
    mark_remaining_skipped "${HALT_REASON}"
    write_status_json "ERROR" 1
    generate_report_impl || true
    exit 1
  fi
  if [[ "${ACCURACY_STATUS}" == "FAIL" ]]; then
    printf "Accuracy precheck: %s (%s)\n" "${ACCURACY_STATUS}" "${ACCURACY_REASON:-accuracy thresholds were not met}"
  fi
  if [[ "${ACCURACY_STOP_ON_FAILURE}" -eq 1 && "${ACCURACY_STATUS}" == "FAIL" ]]; then
    HALTED=1
    HALT_REASON="accuracy precheck did not pass: ${ACCURACY_STATUS}"
    step_status_set accuracy-precheck fail
    step_exit_set accuracy-precheck 1
    step_reason_set accuracy-precheck "${HALT_REASON}"
    mark_remaining_skipped "${HALT_REASON}"
    write_status_json "ERROR" 1
    generate_report_impl || true
    exit 1
  fi
else
  step_status_set accuracy-precheck skipped
  step_reason_set accuracy-precheck "accuracy precheck not requested"
  ACCURACY_STATUS="SKIPPED"
fi

# ---------- Step: xcodebuild-test ----------
xcodebuild_test_impl() {
  rm -rf "${LATENCY_RESULT_PATH}"
  rm -rf "${MEMORY_RESULT_PATH}"
  rm -f "${XCTEST_METRICS_PATH}"
  echo "xcodebuild: configuration=${BUILD_CONFIGURATION} benchmark image name=${IMAGE_NAME} warmup=${WARMUP_MERGED:-} measured=${MEASURED_MERGED:-} ort_ep=${ORT_EP_CANON:-default} ort_profiling=${ORT_PROFILING_MERGED:+on} scope=${TESTING_SCOPE}"
  local runenv=(
    env "TEST_RUNNER_PADDLEOCR_BENCHMARK_IMAGE_NAME=${IMAGE_NAME}"
    "TEST_RUNNER_PADDLEOCR_BENCHMARK_BUILD_CONFIGURATION=${BUILD_CONFIGURATION}"
  )
  [[ -n "${WARMUP_MERGED}" ]] && runenv+=("TEST_RUNNER_PADDLEOCR_BENCHMARK_WARMUP_ITERATIONS=${WARMUP_MERGED}")
  [[ -n "${MEASURED_MERGED}" ]] && runenv+=("TEST_RUNNER_PADDLEOCR_BENCHMARK_MEASURED_ITERATIONS=${MEASURED_MERGED}")
  if [[ -n "${ORT_EP_CANON}" ]]; then
    runenv+=("TEST_RUNNER_PADDLEOCR_BENCHMARK_ORT_EXECUTION_PROVIDER=${ORT_EP_CANON}")
  fi
  [[ -n "${ORT_PROFILING_MERGED:-}" ]] && runenv+=("TEST_RUNNER_PADDLEOCR_BENCHMARK_ORT_PROFILING=${ORT_PROFILING_MERGED}")
  [[ -n "${THREADS_MERGED}" ]] && runenv+=("TEST_RUNNER_PADDLEOCR_BENCHMARK_INTRA_OP_THREADS=${THREADS_MERGED}")
  [[ -n "${XNNPACK_THREADS_MERGED}" ]] && runenv+=("TEST_RUNNER_PADDLEOCR_BENCHMARK_XNNPACK_THREADS=${XNNPACK_THREADS_MERGED}")
  [[ -n "${REC_BATCH_SIZE_MERGED}" ]] && runenv+=("TEST_RUNNER_PADDLEOCR_BENCHMARK_REC_BATCH_SIZE=${REC_BATCH_SIZE_MERGED}")
  [[ -n "${COREML_OPTIONS_MERGED}" ]] && runenv+=("TEST_RUNNER_PADDLEOCR_BENCHMARK_COREML_OPTIONS=${COREML_OPTIONS_MERGED}")
  local scopes=() scope result_path
  IFS=',' read -r -a scopes <<< "${TESTING_SCOPE}"
  for scope in "${scopes[@]}"; do
    scope="$(printf '%s' "$scope" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -n "$scope" ]] || continue
    result_path="${LATENCY_RESULT_PATH}"
    if [[ "${scope}" == *"/testOnDeviceMemoryBenchmark" ]]; then
      result_path="${MEMORY_RESULT_PATH}"
    fi
    rm -rf "${result_path}"
    echo "xcodebuild: running scope=${scope} result=${result_path}"
    if ! run_xcodebuild "${runenv[@]}" xcodebuild test \
      -workspace "${IOS_DEMO_ROOT}/PaddleOCRDemo.xcworkspace" \
      -scheme PaddleOCRDemo \
      -configuration "${BUILD_CONFIGURATION}" \
      -destination "${DEST}" \
      -resultBundlePath "${result_path}" \
      "${XCODE_TEST_SINGLE_RUN_FLAGS[@]}" \
      -only-testing:"${scope}"; then
      return 1
    fi
  done
  collect_app_binary_size
}
run_step xcodebuild-test xcodebuild_test_impl || { mark_remaining_skipped "xcodebuild-test failed: ${HALT_REASON}"; exit 1; }

# ---------- Step: extract-artifacts ----------
extract_artifacts_impl() {
  local metrics_result="${MEMORY_RESULT_PATH}"
  [[ -d "${metrics_result}" ]] || metrics_result="${LATENCY_RESULT_PATH}"
  python3 "${SCRIPT_DIR}/extract_xcresult_attachments.py" \
    --result "${LATENCY_RESULT_PATH}" \
    --output-dir "${OUT_DIR}" \
    --name on-device-performance.json \
    --optional-name ios-ocr-export.json \
    --optional-name ort_profile_detection \
    --optional-name ort_profile_recognition
  python3 "${SCRIPT_DIR}/extract_xctest_metrics.py" \
    --result "${metrics_result}" \
    --output "${XCTEST_METRICS_PATH}"
}
run_step extract-artifacts extract_artifacts_impl || { mark_remaining_skipped "extract-artifacts failed: ${HALT_REASON}"; exit 1; }

# ---------- Step: report ----------
# First pass captures report step timing; write_status_json persists final step state;
# second pass refreshes the Markdown so **Overall** and the step table match this run.
report_impl() {
  generate_report_impl
}
report_refresh() {
  generate_report_impl
}
run_step report report_impl || { write_status_json "ERROR" 1; exit 1; }
write_status_json "COMPLETED" 0
report_refresh || true

exit 0
