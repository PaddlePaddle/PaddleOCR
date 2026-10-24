---
comments: true
---

# iOS Deployment

PaddleOCR ships an iOS OCR sample app under [`deploy/ios_demo`](https://github.com/PaddlePaddle/PaddleOCR/tree/{{PADDLEOCR_GITHUB_REF}}/deploy/ios_demo). The **SwiftUI** demo runs exported **ONNX** models on device using the [ONNX Runtime Objective-C API](https://onnxruntime.ai/docs/tutorials/mobile/) for detection and recognition.

This guide covers environment requirements, model asset setup, building and running in Xcode, and optional ORT format conversion and benchmarking.

## Project layout

App sources, bundled resources, and third-party **source** vendored for this demo live under **`PaddleOCRDemo/`**. Unit tests are in **`PaddleOCRDemoTests/`** next to the Xcode project. The project root also contains `Podfile`, **`scripts/`**, `README.md`, and `NOTICE`.

## Prerequisites

- **macOS** with **Xcode 16.0+** (deployment target **iOS 16+**)
- **CocoaPods** (`gem install cocoapods` or Homebrew)
- `curl`, `tar`

## Supported model presets

`scripts/fetch_ios_demo_models.sh` downloads ONNX bundles into **`PaddleOCRDemo/Models/`**. Supported presets:

- **`PP-OCRv6_small`** (default)
- **`PP-OCRv6_tiny`**
- **`PP-OCRv5_mobile`**

## Quick start

From the **project root** (`deploy/ios_demo`):

```bash
pod install
./scripts/fetch_ios_demo_models.sh
```

Optionally pass the **model preset** (bundle name such as `PP-OCRv6_small`) as a positional argument:

```bash
./scripts/fetch_ios_demo_models.sh PP-OCRv6_small
```

Intermediate `.tar` caches are stored under **`.fetch_ios_demo_models_work/`** at the project root.

After `pod install`, open the workspace:

```bash
open PaddleOCRDemo.xcworkspace
```

Build the **PaddleOCRDemo** scheme. Ensure **`PaddleOCRDemo/Models/`** and **`PaddleOCRDemo/Resources/SampleImages/`** are included in the app target via folder references / **Copy Bundle Resources**, and **`PaddleOCRDemoTests/Fixtures/`** in the test target (as in the checked-in project). The built-in picker sample is **`general_ocr_002.jpg`**.

### ONNX Runtime execution providers (EPs)

The **CPU** / **XNNPACK** / **Core ML** segmented control chooses which ONNX Runtime EP is preferred first when creating the session. ONNX Runtime may still execute some operators on the default **CPU** EP when another EP lacks an implementation. Selecting **CPU** does not register the XNNPACK or Core ML EPs.

## Convert to ORT model format (optional)

To produce [ORT format](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html) weights:

```bash
python3 -m pip install -r requirements-onnx-convert-ort.txt
./scripts/convert_onnx_to_ort.sh
```

By default, conversion writes `inference*.ort` next to each `inference.onnx` under `PaddleOCRDemo/Models/`. The demo loads `inference*.ort` when present, so you do not need to ship ONNX in the bundle. To keep only one weight file in the app, delete the `inference.onnx` files manually, or use **`--output-dir`** and replace files under `PaddleOCRDemo/Models/`. Use **`--input-dir`** to point at any ONNX tree on the host; it defaults to `PaddleOCRDemo/Models` when omitted.

By default the script uses **`--optimization-style Fixed`** (one `.ort` per ONNX file). The raw Python tool defaults to both Fixed and Runtime (`*.ort` and `*.with_runtime_opt.ort`). For Runtime-style ORT files (e.g. Core ML–oriented exports), pass **`--optimization-style Runtime`** (alias: `--optimization_style`). You can pass both: **`--optimization-style Fixed Runtime`**.

The converter emits **`required_operators*.config`** for minimal ORT builds; **by default the shell script deletes them** after a successful run. Pass **`--keep-operator-config`** to retain them. They are **not** read at inference time. The iOS demo loads only **`.ort`** (or `.onnx`) weights.

Example (Runtime conversion to a separate folder):

```bash
./scripts/convert_onnx_to_ort.sh --output-dir ./out/ort_bundles --optimization-style Runtime
```

## Benchmark

This demo provides a benchmark pipeline for measuring on-device OCR latency and memory.

### Prerequisites

Complete [Quick start](#quick-start) first. For benchmark runs on a **physical device** you also need:

- **Code signing (real device only):** `xcodebuild` requires a valid Apple Development certificate and provisioning profile. Open `PaddleOCRDemo.xcworkspace` in Xcode once, select your Development Team under *Signing & Capabilities*, and let Xcode manage provisioning; later `run_benchmark.sh` invocations reuse the cached profile. On first deploy, trust the developer certificate on the device under **Settings → General → VPN & Device Management** (one-time per device). Simulator runs do not require signing.
- Optional accuracy precheck: PaddleOCR (ONNX Runtime engine) on the host for reference generation, and `python3 -m pip install -r requirements-accuracy.txt`.

### Full pipeline

From the project root, use **`./scripts/run_benchmark.sh`**. The script resolves the input image, optionally runs an accuracy precheck, runs the XCTest benchmark on a simulator or device, extracts artifacts, and writes the Markdown report. It always invokes `xcodebuild test` with `-configuration Release`.

Prefer script flags. `--image` copies an arbitrary image into test fixtures for this run. `--fixture` selects an existing file under `PaddleOCRDemoTests/Fixtures/`. If neither is set, the script requires exactly one non-`local-*` image under `Fixtures/` (e.g. `ios_ocr_benchmark_reference.jpg`).

| Intent | Flags |
| --- | --- |
| Destination | `--udid <id>` for a real device, or `--simulator <name>` |
| Image from an arbitrary path | `--image <path>` |
| Image already under `Fixtures/` | `--fixture <name>` |
| Benchmark intensity | `--warmup <n>`, `--measured-iterations <n>` |
| Preferred ONNX Runtime EP | `--ort-execution-provider CPU`, `XNNPACK`, or `CORE_ML` |
| ONNX Runtime profiling JSON | `--ort-profiling` |
| Optional accuracy precheck | `--accuracy-check`, optionally `--accuracy-reference-json <path>` |
| Gate benchmark on accuracy `FAIL` | `--accuracy-check --stop-on-accuracy-failure` |
| Output directory | `--output-dir <dir>` (default: `out/`) |
| Clean previous artifacts | `--clean` |

```bash
./scripts/run_benchmark.sh --udid <device-udid> --warmup 5 --measured-iterations 30
./scripts/run_benchmark.sh --udid <udid> --image /path/to/photo.png --ort-execution-provider CPU
./scripts/run_benchmark.sh --fixture ios_ocr_benchmark_reference --warmup 2 --measured-iterations 20
./scripts/run_benchmark.sh
PADDLEOCR_BENCHMARK_MEASURED_ITERATIONS=30 ./scripts/run_benchmark.sh --warmup 0
PADDLEOCR_BENCHMARK_ORT_EXECUTION_PROVIDER=XNNPACK ./scripts/run_benchmark.sh --fixture ios_ocr_benchmark_reference --warmup 1
./scripts/run_benchmark.sh --accuracy-check --udid <device-udid> --measured-iterations 30
./scripts/run_benchmark.sh --accuracy-check --stop-on-accuracy-failure --udid <device-udid>
./scripts/run_benchmark.sh --output-dir ./benchmark-out --measured-iterations 30
./scripts/run_benchmark.sh --udid <device-udid> --ort-profiling
```

`--accuracy-check` runs before the benchmark in a separate XCTest invocation. Results are `PASS`, `FAIL`, or `ERROR`. By default, `FAIL` records a mismatch but continues; `ERROR` stops the pipeline. `--stop-on-accuracy-failure` also skips the benchmark on `FAIL`.

**ORT session profiling** (attachments `ort_profile_detection`, `ort_profile_recognition`): use `--ort-profiling`. Run profiling in a **separate** run from clean latency measurements.

After the benchmark **completes**, the script overwrites `run-status.json`, `on-device-performance.json`, `xctest-memory-metrics.json`, and `benchmark-report.md` under the output directory (`out/` by default). On **ERROR**, those files may be missing or partial.

The report includes model input tensor shape distributions, first measured run line count, inferred model preset, actual model format (`onnx` or `ort`), det/rec/total weight sizes, app executable size when resolvable, cold model load time, measured latency, memory, etc.

| Artifact | Producer | Purpose |
|---|---|---|
| `accuracy-reference.json` | `ocr_reference_run.py` | Optional reference JSON |
| `accuracy-result.xcresult` | `xcodebuild test` | Optional accuracy precheck |
| `ios-ocr-export.json` | `extract_xcresult_attachments.py` | Optional iOS accuracy payload |
| `accuracy-summary.json` | `compare_ocr_json.py` | Optional accuracy summary |
| `latency-result.xcresult` | `xcodebuild test` | Latency benchmark |
| `memory-result.xcresult` | `xcodebuild test` | Memory benchmark |
| `on-device-performance.json` | `extract_xcresult_attachments.py` | Latency stats from `latency-result.xcresult` |
| `ort_profile_detection`, `ort_profile_recognition` | `extract_xcresult_attachments.py` | Optional ORT profiling JSON |
| `xctest-memory-metrics.json` | `extract_xctest_metrics.py` | XCTest memory metrics |
| `logs/*.log` | `run_benchmark.sh` | Per-step logs |
| `run-status.json` | `run_benchmark.sh` | Step outcomes and metadata |
| `benchmark-report.md` | `generate_benchmark_report.py` | Human-readable report |

Exit **`0`** when the pipeline reaches **`COMPLETED`**; non-zero on **`ERROR`**. Accuracy `FAIL` changes the exit code only with `--stop-on-accuracy-failure`; `ERROR` always stops the pipeline.

### Advanced: XCTest environment

Most users should use `run_benchmark.sh` flags instead of setting XCTest environment variables directly. This section is for debugging the `PaddleOCRDemoTests` target from Xcode or a custom `xcodebuild test` command.

Benchmark tests read **`PADDLEOCR_BENCHMARK_*`** variables on the XCTest runner process.

| How you launch tests | What to configure |
| --- | --- |
| **Xcode** | Scheme **PaddleOCRDemo** → **Test** → **Arguments** → **Environment Variables** (`PADDLEOCR_BENCHMARK_…` as-is). |
| **Custom `xcodebuild test`** | **`TEST_RUNNER_` + variable name** (e.g. `TEST_RUNNER_PADDLEOCR_BENCHMARK_IMAGE_NAME=…`). |

| Variable | If unset | Role |
| --- | --- | --- |
| `PADDLEOCR_BENCHMARK_IMAGE_NAME` | **`ios_ocr_benchmark_reference`** | Bundled test image under **`Fixtures/`**. |
| `PADDLEOCR_BENCHMARK_WARMUP_ITERATIONS` | **3** | Untimed full OCR runs before timing. |
| `PADDLEOCR_BENCHMARK_MEASURED_ITERATIONS` | **10** | Timed runs. |
| `PADDLEOCR_BENCHMARK_ORT_EXECUTION_PROVIDER` | **CPU** | **`CPU`**, **`XNNPACK`**, or **`CORE_ML`**. |
| `PADDLEOCR_BENCHMARK_ORT_PROFILING` | (unset) | Enable ORT session profiling (`1` / `true` / `yes` / `on`). |
| `PADDLEOCR_BENCHMARK_ONLY_TESTING_SCOPE` | latency + memory benchmark tests | `run_benchmark.sh` only: comma-separated `-only-testing` scope. |

Non-negative integers for the two iteration variables.

## Third-party licenses

Bundled **Clipper** (polyclipping 6.4.2) is under the [Boost Software License 1.0](https://www.boost.org/LICENSE_1_0.txt); see `NOTICE` and `PaddleOCRDemo/ThirdParty/Clipper1/LICENSE`. CocoaPods pods are governed by their respective licenses (see `Podfile.lock` after `pod install`).
