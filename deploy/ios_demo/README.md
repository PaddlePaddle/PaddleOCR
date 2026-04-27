# iOS Demo

SwiftUI demo that runs OCR on device using exported ONNX models and [ONNX Runtime Objective-C API](https://onnxruntime.ai/docs/tutorials/mobile/).

## Layout

All app sources, bundled resources, and third-party **source** vendored for this demo live under **`PaddleOCRDemo/`**. Unit tests are in **`PaddleOCRDemoTests/`** next to the Xcode project. The project root also contains `Podfile`, **`scripts/`**, `README.md`, and `NOTICE`.

## Prerequisites

- macOS with Xcode (iOS 16+)
- CocoaPods (`gem install cocoapods` or Homebrew)
- `curl`, `tar`

## One-time asset setup

From the **project root**:

```bash
pod install
./scripts/fetch_ios_demo_models.sh
```

`scripts/fetch_ios_demo_models.sh` downloads ONNX bundles into **`PaddleOCRDemo/Models/`**. Intermediate `.tar` caches are stored under **`.fetch_ios_demo_models_work/`** at the project root.

Optionally, pass the **model preset** (bundle name such as `PP-OCRv6_small`) as a positional argument:

```bash
./scripts/fetch_ios_demo_models.sh PP-OCRv6_small
```

Currently, the supported model presets are `PP-OCRv6_small` and `PP-OCRv6_tiny`. The default preset is `PP-OCRv6_small`.

## Quantize ONNX models on the host (optional)

To build **INT8** variants using [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), ensure that Python 3.8 or newer is installed in your host environment.

First, install the required Python dependencies:

```bash
python3 -m pip install -r requirements-onnx-quantize.txt
```

Next, run the quantization script:

```bash
python3 scripts/quantize_onnx_model.py \
  --input-model-dir PaddleOCRDemo/Models/det \
  --output-model-dir /path/to/det_quant \
  --mode dynamic
```

* **`--mode dynamic`**
  Uses `quantize_dynamic`, which performs weight-only quantization and does **not** require a calibration dataset.

* **`--mode static`**
  Uses `quantize_static` (QDQ format). This mode **requires** a calibration dataset specified via `--calib-data-dir`.

  * The calibration directory should contain **float32 `.npy` files**, with **one file per sample**.
  * Each tensor must match the shape of the model’s **single input** (note: in this demo, both `det` and `rec` models have one input).
  * You can choose calibration methods such as *MinMax*, *Entropy*, or others supported by your installed `onnxruntime` version via `--calibration-method`.

**Building calibration `.npy` files (optional):** use **`scripts/build_onnx_calib_npy.py`** to turn a folder of images into tensors.

1. **Dependencies** (host Python):

   ```bash
   python3 -m pip install -r requirements-build-calib.txt
   ```

2. **Run the build script**:

   | Flag | Role |
   | --- | --- |
   | `--task` | `det` = detection model, `rec` = recognition model (must match `--model-dir`). |
   | `--model-dir` | Model directory (e.g. `PaddleOCRDemo/Models/det` or `.../rec`). |
   | `--image-dir` | Images to calibrate on (`.png` / `.jpg` / …). |
   | `--output-dir` | Where to write `.npy` files (created if missing). |
   | `--device` | Optional; e.g. `cpu` (default) or `gpu:0`. |
   | `--det-model-dir` | **Recognition only:** detection model directory. If set, each *full page* in `--image-dir` is run through det; every detected text line is rotated-cropped and written as a separate rec tensor (`…_box000.npy`, …), instead of one tensor per file. |
   | `--max-crops-per-image` | With `--det-model-dir`, limit how many boxes per page are used (default `0` = no cap). |

   For **detection** calibration, one `.npy` is produced per input image:

   ```bash
   python3 scripts/build_onnx_calib_npy.py --task det \
     --model-dir PaddleOCRDemo/Models/det \
     --image-dir /path/to/your/images \
     --output-dir /path/to/calib_npy
   ```

   For **recognition** calibration, you can either use **per-line** crops (if you have them) or pass **full pages** and point at the **det** model so line crops are generated automatically:

   ```bash
   python3 scripts/build_onnx_calib_npy.py --task rec \
     --model-dir PaddleOCRDemo/Models/rec \
     --det-model-dir PaddleOCRDemo/Models/det \
     --image-dir /path/to/full_page_images \
     --output-dir /path/to/calib_rec_crops
   ```

3. **Then static quantize** using that directory:

   ```bash
   python3 scripts/quantize_onnx_model.py \
     --input-model-dir PaddleOCRDemo/Models/det \
     --output-model-dir /path/to/det_int8 \
     --mode static \
     --calib-data-dir /path/to/calib_npy
   ```

4. **(Optional) QDQ debug**  
   For the same [ORT debugging story](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md#debugging) as the upstream *run_qdq_debug* flow: compare a **float** ONNX to your **QDQ** output using the same `.npy` inputs as calibration. This uses ORT’s `qdq_loss_debug` helpers (weight + activation SQNR).  
   * `--float-model` should be the same float graph you fed into `quantize_onnx_model.py` (if you use default `--ort-preprocess`, that means the [pre-processed](https://raw.githubusercontent.com/microsoft/onnxruntime-inference-examples/main/quantization/image_classification/cpu/ReadMe.md#pre-processing) `inference.onnx` in float form—save a copy before quant if you do not have it, or re-run pre-process once).  

   ```bash
   python3 scripts/debug_onnx_qdq.py \
     --float-model /path/to/float_for_quant.onnx \
     --qdq-model /path/to/det_int8/inference.onnx \
     --calib-data-dir /path/to/calib_npy \
     --json-report /tmp/qdq_debug.json
   ```

## Convert to ORT model format (optional)

To produce [ORT format](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html) weights, install the Python dependencies for the converter, then run the converter:

```bash
python3 -m pip install -r requirements-onnx-convert-ort.txt
./scripts/convert_onnx_to_ort.sh
```

By default, conversion writes `inference*.ort` next to each `inference.onnx` under `PaddleOCRDemo/Models/`, so both formats sit in the same tree and the bundle can grow. The demo loads `inference*.ort` when present, so you do not need to ship ONNX in the bundle. To keep only one weight file in the app, you can either manually delete the `inference.onnx` files, or use the `--out-dir` option to generate `inference*.ort` models in a separate directory. These generated models can then replace the originals in `PaddleOCRDemo/Models/`. Use **`--input-dir`** to point at any ONNX tree on the host; it defaults to `PaddleOCRDemo/Models` when omitted.

Place this script’s options first (`--input-dir`, `--out-dir`), then any ORT converter flags. The script inserts `--` before the model path internally so options like `--optimization_style` (which accept multiple values) are not misparsed. Example:

```bash
./scripts/convert_onnx_to_ort.sh --out-dir ./out/ort_bundles --optimization_style Runtime
```

## Open in Xcode

```bash
open PaddleOCRDemo.xcworkspace
```

If you use CocoaPods, run `pod install` in the project root first so the workspace is generated next to the `Podfile`.

Build the **PaddleOCRDemo** scheme. Ensure **`PaddleOCRDemo/Models/`** and **`PaddleOCRDemo/Resources/SampleImages/`** are included in the app target via folder references / **Copy Bundle Resources**, and **`PaddleOCRDemoTests/Fixtures/`** in the test target (as in the checked-in project). The built-in picker sample is **`general_ocr_002.jpg`**.

## Benchmark

This demo provides a benchmark pipeline for measuring on-device OCR latency and memory.

### Prerequisites

Complete [One-time asset setup](#one-time-asset-setup) first. For benchmark runs you also need:

- Xcode 16 or later (the benchmark extractor uses `xcresulttool get test-results`, introduced in 16.0).
- Optional accuracy precheck: PaddleOCR (with ONNX Runtime engine) for reference generation, and `python3 -m pip install -r requirements-accuracy.txt` for additional dependencies.

### Full pipeline

From the project root, use **`./scripts/run_benchmark.sh`**. This is the supported entry point for benchmark runs. The script resolves the input image, optionally runs an accuracy precheck, runs the XCTest benchmark on a simulator or device, extracts result artifacts, and writes the Markdown report.

The script always invokes `xcodebuild test` with `-configuration Release`.

**Configuring the run:** prefer script flags. `--image` copies an arbitrary image into the test fixtures for this run. `--fixture` selects an existing file under `PaddleOCRDemoTests/Fixtures/`. If neither is set, the script requires exactly one non-`local-*` image under `PaddleOCRDemoTests/Fixtures/` (a typical setup has one benchmark image such as `ios_ocr_benchmark_reference.jpg`).

| Intent | Flags |
| --- | --- |
| Destination | `--udid <id>` for a real device, or `--simulator <name>` |
| Image from an arbitrary path | `--image <path>` (copied into `Fixtures/` as `local-*` for this run) |
| Image already under `PaddleOCRDemoTests/Fixtures/` | `--fixture <name>` (stem or `stem.ext`) |
| Benchmark intensity | `--warmup <n>`, `--measured-iterations <n>` |
| ONNX Runtime EP | `--inference-backend CORE_ML`, `XNNPACK`, or `CPU` |
| Optional accuracy precheck | `--accuracy-check`, optionally `--accuracy-reference-json <path>` |
| Gate benchmark on accuracy `FAIL` | `--accuracy-check --stop-on-accuracy-failure` |
| Output directory | `--out-dir <dir>` (default: `out/`) |
| Clean previous artifacts | `--clean` (removes `Fixtures/local-*` and prior artifacts under the output directory) |

```bash
./scripts/run_benchmark.sh --udid <device-udid> --warmup 5 --measured-iterations 30
./scripts/run_benchmark.sh --udid <udid> --image /path/to/photo.png --inference-backend CPU
./scripts/run_benchmark.sh --fixture ios_ocr_benchmark_reference --warmup 2 --measured-iterations 20
./scripts/run_benchmark.sh
PADDLEOCR_BENCHMARK_MEASURED_ITERATIONS=30 ./scripts/run_benchmark.sh --warmup 0
./scripts/run_benchmark.sh --accuracy-check --udid <device-udid> --measured-iterations 30
./scripts/run_benchmark.sh --accuracy-check --stop-on-accuracy-failure --udid <device-udid>
./scripts/run_benchmark.sh --out-dir ./benchmark-out --measured-iterations 30
```

`--accuracy-check` runs before the benchmark in a separate XCTest invocation. Its result is reported as an accuracy precheck (`PASS`, `FAIL`, or `ERROR`). By default, `FAIL` records an accuracy mismatch but continues to the benchmark, while `ERROR` stops the pipeline because the precheck infrastructure did not produce a trustworthy result. Add `--stop-on-accuracy-failure` when you also want `FAIL` to skip the benchmark tests and return a non-zero exit code.

`PADDLEOCR_BENCHMARK_ORT_PROFILING=1` enables ONNX Runtime session profiling attachments (`ort_profile_detection`, `ort_profile_recognition`). Profiling changes runtime behavior and should be captured in a separate run from clean latency measurements.

The script writes artifacts under the output directory (`out/` by default, configurable via `--out-dir`). After the benchmark **completes**, it overwrites `run-status.json`, `on-device-performance.json`, `xctest-memory-metrics.json`, and `benchmark-report.md` there. If the benchmark pipeline **errors** earlier, those files may be missing or partial for this run (any existing copies are from a previous attempt).

The report includes model input tensor shape distributions, first measured run line count, inferred model preset, actual model format (`onnx` or `ort`), det/rec/total model weight sizes, app executable size when it can be resolved from Xcode build settings, cold model load time, measured latency, memory, etc. Shape distribution counts are counted per model invocation inside the measured loop: detection contributes one shape per full OCR run, while recognition contributes one shape per recognition batch.

Outputs under the configured output directory:

| Artifact | Producer | Purpose |
|---|---|---|
| `accuracy-reference.json` | `ocr_reference_run.py` | Optional generated reference JSON, unless `--accuracy-reference-json` is provided |
| `accuracy-result.xcresult` | `xcodebuild test` | Optional accuracy precheck XCTest run |
| `ios-ocr-export.json` | `extract_xcresult_attachments.py` | Optional iOS accuracy payload extracted from the accuracy precheck |
| `accuracy-summary.json` | `compare_ocr_json.py` | Optional accuracy precheck summary |
| `latency-result.xcresult` | `xcodebuild test` | Latency benchmark XCTest run |
| `memory-result.xcresult` | `xcodebuild test` | Memory benchmark XCTest run |
| `on-device-performance.json` | `extract_xcresult_attachments.py` | iOS latency stats extracted from `latency-result.xcresult` |
| `ort_profile_detection`, `ort_profile_recognition` | `extract_xcresult_attachments.py` | Optional ONNX Runtime **profiling** JSON from the latency benchmark run |
| `xctest-memory-metrics.json` | `extract_xctest_metrics.py` | XCTest memory metrics, normally extracted from `memory-result.xcresult` |
| `logs/*.log` | `run_benchmark.sh` | Per-step command logs, especially useful after failures |
| `run-status.json` | `run_benchmark.sh` | Per-step outcomes and benchmark metadata |
| `benchmark-report.md` | `generate_benchmark_report.py` | Human-readable report |

Exit **`0`** when the benchmark pipeline reaches **`COMPLETED`**; non-zero on benchmark pipeline **`ERROR`**. Optional accuracy precheck `FAIL` is recorded in `run-status.json` and `benchmark-report.md` and changes the exit code only when `--stop-on-accuracy-failure` is set. Accuracy precheck `ERROR` always stops the pipeline.

### Advanced: XCTest Environment

Most users should use `run_benchmark.sh` flags instead of setting XCTest environment variables directly. This section is only for debugging the `PaddleOCRDemoTests` target from Xcode or a custom `xcodebuild test` command.

The benchmark tests read settings through variables named **`PADDLEOCR_BENCHMARK_*`**. These variables apply to the XCTest runner process, not automatically to your interactive shell.

| How you launch tests | What to configure |
| --- | --- |
| **Xcode** | Scheme **PaddleOCRDemo** → **Test** (not Run) → **Arguments** → **Environment Variables**. Use the names below **as-is** (`PADDLEOCR_BENCHMARK_…`). |
| **Custom `xcodebuild test`** | Set **`TEST_RUNNER_` + the same name** (e.g. `TEST_RUNNER_PADDLEOCR_BENCHMARK_IMAGE_NAME=…`). `xcodebuild` injects them into the test runner and strips the prefix so the test still reads **`PADDLEOCR_BENCHMARK_…`**. |

| Variable | If unset | Role |
| --- | --- | --- |
| `PADDLEOCR_BENCHMARK_IMAGE_NAME` | Defaults to **`ios_ocr_benchmark_reference`**. Set explicitly to use another bundled image. | Bundled test image: **stem** or **`stem.ext`** under **`Fixtures/`**. |
| `PADDLEOCR_BENCHMARK_WARMUP_ITERATIONS` | **3** | Untimed full OCR runs before timing (warm caches / JIT). Used by latency and memory benchmark tests. |
| `PADDLEOCR_BENCHMARK_MEASURED_ITERATIONS` | **10** | Timed runs. Used by latency and memory benchmark tests. |
| `PADDLEOCR_BENCHMARK_INFERENCE_BACKEND` | **CORE_ML** | ONNX Runtime EP for **`OCRBenchmarkTests`**: **`CORE_ML`**, **`XNNPACK`**, or **`CPU`**. |
| `PADDLEOCR_BENCHMARK_ORT_PROFILING` | (unset) | Set to **`1`**, **`true`**, **`yes`**, or **`on`** to enable ONNX Runtime **session profiling** (JSON attachments). Profiling **distorts** wall-clock timings; use a separate run for clean latency. |
| `PADDLEOCR_BENCHMARK_ONLY_TESTING_SCOPE` | latency + memory benchmark tests | `run_benchmark.sh` only: comma-separated values passed to repeated `xcodebuild -only-testing` flags. |

Non-negative integers for the two iteration variables.

## Third-party licenses

Bundled **Clipper** (polyclipping 6.4.2) is under the [Boost Software License 1.0](https://www.boost.org/LICENSE_1_0.txt); see `NOTICE` and `PaddleOCRDemo/ThirdParty/Clipper1/LICENSE`. CocoaPods pods are governed by their respective licenses (see `Podfile.lock` after `pod install`).
