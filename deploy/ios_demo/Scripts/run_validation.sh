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
# End-to-end validation runner for the iOS demo (accuracy gate + perf capture).
#
# Pipeline: preflight → resolve-image → ref-gen → resolve-destination →
#           xcodebuild-test → extract-attachments → compare → report
#
# Exit code reflects the accuracy comparison step only. Performance output is for the report, not pass/fail.
#
# Requires: bash 3.2+, xcodebuild, xcrun, python3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IOS_DEMO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Fixtures/: images shipped for validation; `local-*` entries come from --image; auto-pick ignores `local-*`.
FIXTURES_DIR="${IOS_DEMO_ROOT}/PaddleOCRDemoTests/Fixtures"
DEFAULT_SIMULATOR="iPhone 16"
# xcodebuild -only-testing target (override via ONLY_TESTING_SCOPE without editing this file).
ONLY_TESTING_SCOPE="${ONLY_TESTING_SCOPE:-PaddleOCRDemoTests/OCRValidationTests}"

UDID=""
SIMULATOR=""
IMAGE=""
FIXTURE_ARG=""
WARMUP_CLI=""
MEASURED_CLI=""
INFERENCE_CLI=""
SKIP_REF_GEN=0
OUT_DIR="${IOS_DEMO_ROOT}/out"
CLEAN=0

usage() {
  cat <<EOF
Usage: ./Scripts/run_validation.sh [OPTIONS]

Options:
  --udid <id>           Real-device UDID (preferred real-device path)
  --simulator <name>    Simulator name (default: ${DEFAULT_SIMULATOR})
  --image <path>        Ad-hoc validation image; copied as Fixtures/local-<basename>.
                        Without --image/--fixture, see image selection below.
  --fixture <name>      Use a file already under PaddleOCRDemoTests/Fixtures/ (stem or
                        file name, e.g. ios_ocr_validation_reference or .jpg). Useful when
                        several fixtures exist.
  --warmup <n>          Benchmark warmup iterations (non-negative int).
  --measured-iterations <n>  Timed benchmark iterations.
  --inference-backend <NAME>  ONNX Runtime EP for tests: CORE_ML or XNNPACK.
  --skip-ref-gen        Reuse existing <out-dir>/ref.json
  --out-dir <dir>       Output directory (default: out/)
  --clean               Delete Fixtures/local-* and <out-dir>/* before running
  -h, --help            Show help

Environment (optional; CLI wins when both are set):
  PADDLEOCR_VALIDATION_IMAGE_NAME   Select which bundled fixture to use (same as --fixture)
  PADDLEOCR_VALIDATION_WARMUP_ITERATIONS
  PADDLEOCR_VALIDATION_MEASURED_ITERATIONS
  PADDLEOCR_VALIDATION_INFERENCE_BACKEND   CORE_ML or XNNPACK
  ONLY_TESTING_SCOPE                Optional; passed to xcodebuild -only-testing (narrow test subset)

Image selection when --image is not used:
  1) --fixture <name>, if given, else PADDLEOCR_VALIDATION_IMAGE_NAME, if set
  2) otherwise exactly one non-local-* file in Fixtures/
EOF
}

die() {
  printf "[run_validation] FAIL: %s\n" "$1" >&2
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
    --inference-backend) INFERENCE_CLI="$2"; shift 2 ;;
    --skip-ref-gen) SKIP_REF_GEN=1; shift ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" "argument parsing" "See --help." ;;
  esac
done

# Fixture label: --fixture overrides PADDLEOCR_VALIDATION_IMAGE_NAME.
EXPLICIT_FIXTURE="${FIXTURE_ARG:-${PADDLEOCR_VALIDATION_IMAGE_NAME:-}}"

# Warmup / measured: CLI over env; forward to xcodebuild only when non-empty.
WARMUP_MERGED="${WARMUP_CLI:-${PADDLEOCR_VALIDATION_WARMUP_ITERATIONS:-}}"
MEASURED_MERGED="${MEASURED_CLI:-${PADDLEOCR_VALIDATION_MEASURED_ITERATIONS:-}}"
if [[ -n "${WARMUP_MERGED}" ]]; then
  [[ "${WARMUP_MERGED}" =~ ^[0-9]+$ ]] \
    || die "Invalid --warmup / PADDLEOCR_VALIDATION_WARMUP_ITERATIONS: ${WARMUP_MERGED}" "argument parsing" "Use a non-negative integer."
fi
if [[ -n "${MEASURED_MERGED}" ]]; then
  [[ "${MEASURED_MERGED}" =~ ^[0-9]+$ ]] \
    || die "Invalid --measured-iterations / PADDLEOCR_VALIDATION_MEASURED_ITERATIONS: ${MEASURED_MERGED}" "argument parsing" "Use a non-negative integer."
fi

# Inference EP: CORE_ML or XNNPACK; forward when set (flag or env).
INFERENCE_MERGED="${INFERENCE_CLI:-${PADDLEOCR_VALIDATION_INFERENCE_BACKEND:-}}"
INFERENCE_CANON=""
if [[ -n "${INFERENCE_MERGED}" ]]; then
  case "${INFERENCE_MERGED}" in
    coreMLOnly) INFERENCE_CANON="CORE_ML" ;;
    xnnpackOnly) INFERENCE_CANON="XNNPACK" ;;
    *)
      _inf_lc="$(printf '%s' "${INFERENCE_MERGED}" | tr '[:upper:]' '[:lower:]')"
      case "${_inf_lc}" in
        core_ml) INFERENCE_CANON="CORE_ML" ;;
        xnnpack) INFERENCE_CANON="XNNPACK" ;;
        *)
          die "Invalid --inference-backend / PADDLEOCR_VALIDATION_INFERENCE_BACKEND: ${INFERENCE_MERGED}" "argument parsing" "Use CORE_ML or XNNPACK (Swift: coreMLOnly / xnnpackOnly)."
          ;;
      esac
      ;;
  esac
fi

LOGS_DIR="${OUT_DIR}/logs"
STATUS_PATH="${OUT_DIR}/run-status.json"
REPORT_PATH="${OUT_DIR}/validation-report.md"

mkdir -p "${OUT_DIR}" "${LOGS_DIR}"

if [[ "${CLEAN}" -eq 1 ]]; then
  rm -f "${FIXTURES_DIR}"/local-*
  find "${OUT_DIR}" -mindepth 1 -not -path "${LOGS_DIR}" -not -path "${LOGS_DIR}/*" -exec rm -rf {} + 2>/dev/null || true
  mkdir -p "${LOGS_DIR}"
fi

# ---------- Step state (parallel indexed arrays, bash 3.2 compatible) ----------

STEP_NAMES=(preflight resolve-image ref-gen resolve-destination xcodebuild-test extract-attachments compare report)
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
IMAGE_NAME=""
IMAGE_SRC=""
IMAGE_SOURCE=""
DEST=""
COMPARE_EXIT=0
RUN_STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

now_ms() {
  python3 -c 'import time; print(int(time.time()*1000))'
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
  printf "  -> FAIL (%sms, exit %s): %s\n" "$duration" "$rc" "$reason" >&2
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

# Pass step state as flat positional args (parallel arrays) to Python — avoids
# shell assoc arrays (bash 4+) and env var name sanitization.
write_status_json() {
  local overall="$1" exit_code="$2"
  local finished
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 - "$STATUS_PATH" "$overall" "$exit_code" "$RUN_STARTED" "$finished" \
    "$IMAGE_NAME" "$IMAGE_SOURCE" "$DEST" "${#STEP_NAMES[@]}" \
    "${STEP_NAMES[@]}" \
    "${STEP_STATUS[@]}" \
    "${STEP_DURATION_MS[@]}" \
    "${STEP_EXIT[@]}" \
    "${STEP_REASON[@]}" \
    "${STEP_LOG[@]}" <<'PY'
import json
import sys

args = sys.argv[1:]
path, overall, exit_code, started, finished, image_name, image_source, destination = args[:8]
n = int(args[8])
base = 9
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
  elif [[ "${COMPARE_EXIT}" -ne 0 ]]; then
    overall="FAIL"
    exit_code="${COMPARE_EXIT}"
  else
    overall="PASS"
    exit_code=0
  fi
  printf "\n===== SUMMARY =====\n"
  printf "Overall:   %s\n" "${overall}"
  printf "Exit code: %s\n" "${exit_code}"
  if [[ "${overall}" == "PASS" || "${overall}" == "FAIL" ]]; then
    printf "Status:    %s\n" "${STATUS_PATH}"
    printf "Report:    %s\n" "${REPORT_PATH}"
    printf "Compare:   %s\n" "${OUT_DIR}/compare-summary.json"
  else
    printf "ERROR: See full logs in %s\n" "${LOGS_DIR}"
  fi
  exit "${exit_code}"
}
trap finish EXIT

run_step() {
  local name="$1"; shift
  local log="${LOGS_DIR}/${name}.log"
  local start end duration rc tail
  printf "\n===== [%s] step %s: %s =====\n" "$(date +%H:%M:%S)" "$name" "$*"
  step_status_set "$name" running
  start="$(now_ms)"
  if "$@" >"$log" 2>&1; then
    end="$(now_ms)"; duration=$((end - start))
    record_step_ok "$name" "$duration"
    return 0
  else
    rc=$?
    end="$(now_ms)"; duration=$((end - start))
    tail="$(tail -n 5 "$log" | head -n 1 || true)"
    record_step_fail "$name" "$duration" "$rc" "${tail:-step failed}"
    return "$rc"
  fi
}

# ---------- Step: preflight ----------
preflight_impl() {
  [[ -f "${IOS_DEMO_ROOT}/PaddleOCRDemo/Models/det/inference.yml" ]] \
    || { echo "Models/det/inference.yml missing; run ./Scripts/fetch_ios_demo_models.sh" >&2; return 1; }
  [[ -f "${IOS_DEMO_ROOT}/PaddleOCRDemo/Models/rec/inference.yml" ]] \
    || { echo "Models/rec/inference.yml missing; run ./Scripts/fetch_ios_demo_models.sh" >&2; return 1; }
  command -v xcodebuild >/dev/null || { echo "xcodebuild not in PATH" >&2; return 1; }
  command -v xcrun >/dev/null || { echo "xcrun not in PATH" >&2; return 1; }
  command -v python3 >/dev/null || { echo "python3 not in PATH" >&2; return 1; }
  [[ -d "${FIXTURES_DIR}" ]] || { echo "Fixtures/ directory missing" >&2; return 1; }
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
    [[ -n "$cand" ]] || { echo "Fixture not found for --fixture / PADDLEOCR_VALIDATION_IMAGE_NAME: ${name}" >&2; return 1; }
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

# ---------- Step: ref-gen ----------
if [[ "${SKIP_REF_GEN}" -eq 1 ]]; then
  step_status_set ref-gen skipped
  step_reason_set ref-gen "--skip-ref-gen"
  echo "Skipping ref-gen (--skip-ref-gen)"
else
  ref_gen_impl() {
    python3 "${SCRIPT_DIR}/ocr_reference_run.py" \
      --image "${IMAGE_SRC}" \
      --output "${OUT_DIR}/ref.json" \
      --device cpu \
      --align-ios-defaults
  }
  run_step ref-gen ref_gen_impl || { mark_remaining_skipped "ref-gen failed: ${HALT_REASON}"; exit 1; }
fi

# ---------- Step: resolve-destination ----------
resolve_destination_impl() {
  if [[ -n "${UDID}" ]]; then
    DEST="platform=iOS,id=${UDID}"
  elif [[ -n "${SIMULATOR}" ]]; then
    DEST="platform=iOS Simulator,name=${SIMULATOR}"
  else
    DEST="platform=iOS Simulator,name=${DEFAULT_SIMULATOR}"
  fi
  echo "Destination: ${DEST}"
}
run_step resolve-destination resolve_destination_impl || { mark_remaining_skipped "resolve-destination failed: ${HALT_REASON}"; exit 1; }

# ---------- Step: xcodebuild-test ----------
xcodebuild_test_impl() {
  rm -rf "${OUT_DIR}/result.xcresult"
  echo "xcodebuild: validation image name=${IMAGE_NAME} warmup=${WARMUP_MERGED:-} measured=${MEASURED_MERGED:-} inference=${INFERENCE_CANON:-default}"
  local runenv=(
    env "TEST_RUNNER_PADDLEOCR_VALIDATION_IMAGE_NAME=${IMAGE_NAME}"
  )
  [[ -n "${WARMUP_MERGED}" ]] && runenv+=("TEST_RUNNER_PADDLEOCR_VALIDATION_WARMUP_ITERATIONS=${WARMUP_MERGED}")
  [[ -n "${MEASURED_MERGED}" ]] && runenv+=("TEST_RUNNER_PADDLEOCR_VALIDATION_MEASURED_ITERATIONS=${MEASURED_MERGED}")
  [[ -n "${INFERENCE_CANON}" ]] && runenv+=("TEST_RUNNER_PADDLEOCR_VALIDATION_INFERENCE_BACKEND=${INFERENCE_CANON}")
  "${runenv[@]}" xcodebuild test \
    -workspace "${IOS_DEMO_ROOT}/PaddleOCRDemo.xcworkspace" \
    -scheme PaddleOCRDemo \
    -destination "${DEST}" \
    -resultBundlePath "${OUT_DIR}/result.xcresult" \
    -only-testing:"${ONLY_TESTING_SCOPE}"
}
run_step xcodebuild-test xcodebuild_test_impl || { mark_remaining_skipped "xcodebuild-test failed: ${HALT_REASON}"; exit 1; }

# ---------- Step: extract-attachments ----------
extract_attachments_impl() {
  python3 "${SCRIPT_DIR}/extract_xcresult_attachments.py" \
    --result "${OUT_DIR}/result.xcresult" \
    --out-dir "${OUT_DIR}" \
    --name ios-ocr-export.json \
    --name on-device-performance.json
}
run_step extract-attachments extract_attachments_impl || { mark_remaining_skipped "extract-attachments failed: ${HALT_REASON}"; exit 1; }

# ---------- Step: compare (non-halting) ----------
compare_impl() {
  python3 "${SCRIPT_DIR}/compare_ocr_json.py" \
    "${OUT_DIR}/ref.json" \
    "${OUT_DIR}/ios-ocr-export.json" \
    --iou-threshold 0.65 \
    --cer-threshold 0.05 \
    --max-unmatched-ratio 0.1 \
    --json-summary-out "${OUT_DIR}/compare-summary.json"
}
step_status_set compare running
_start="$(now_ms)"
if (compare_impl) >"${LOGS_DIR}/compare.log" 2>&1; then
  _end="$(now_ms)"; _dur=$((_end - _start))
  step_status_set compare ok
  step_duration_set compare "$_dur"
  step_exit_set compare 0
  echo "  -> OK (${_dur}ms)"
else
  COMPARE_EXIT=$?
  _end="$(now_ms)"; _dur=$((_end - _start))
  step_status_set compare fail
  step_duration_set compare "$_dur"
  step_exit_set compare "$COMPARE_EXIT"
  step_reason_set compare "accuracy thresholds not met (see logs/compare.log)"
  step_log_set compare "logs/compare.log"
  echo "  -> Compare exited ${COMPARE_EXIT} (FAIL verdict)"
fi

# ---------- Step: report (after compare; PASS and FAIL both get artifacts) ----------
# First pass captures report step timing; write_status_json persists final step state;
# second pass refreshes the Markdown so **Overall** and the step table match this run.
report_impl() {
  python3 "${SCRIPT_DIR}/generate_validation_report.py" \
    --compare-summary "${OUT_DIR}/compare-summary.json" \
    --on-device-performance-json "${OUT_DIR}/on-device-performance.json" \
    --run-status "${STATUS_PATH}" \
    --output "${REPORT_PATH}"
}
report_refresh() {
  python3 "${SCRIPT_DIR}/generate_validation_report.py" \
    --compare-summary "${OUT_DIR}/compare-summary.json" \
    --on-device-performance-json "${OUT_DIR}/on-device-performance.json" \
    --run-status "${STATUS_PATH}" \
    --output "${REPORT_PATH}"
}
if [[ "${COMPARE_EXIT}" -eq 0 ]]; then
  run_step report report_impl || true
  write_status_json "PASS" 0
else
  run_step report report_impl || true
  write_status_json "FAIL" "${COMPARE_EXIT}"
fi
report_refresh || true

exit "${COMPARE_EXIT}"
