# iOS Demo

SwiftUI demo that runs OCR on device using exported ONNX models and [ONNX Runtime Objective-C API](https://onnxruntime.ai/docs/tutorials/mobile/).

## Layout

All app sources, bundled resources, and third-party **source** vendored for this demo live under **`PaddleOCRDemo/`**. Unit tests are in **`PaddleOCRDemoTests/`** next to the Xcode project. The project root also contains `Podfile`, **`Scripts/`**, `README.md`, and `NOTICE`.

> **Paths in this README** assume your shell’s working directory is the **project root** (the folder that contains `Podfile` and `Scripts/`). If you use a checkout of the full PaddleOCR repository, that folder is **`deploy/ios_demo/`**.

## Third-party licenses

Bundled **Clipper** (polyclipping 6.4.2) is under the [Boost Software License 1.0](https://www.boost.org/LICENSE_1_0.txt); see `NOTICE` and `PaddleOCRDemo/ThirdParty/Clipper1/LICENSE`. CocoaPods pods are governed by their respective licenses (see `Podfile.lock` after `pod install`).

## Prerequisites

- macOS with Xcode (iOS 16+)
- CocoaPods (`gem install cocoapods` or Homebrew)
- `curl`, `tar`

## One-time asset setup

From the **project root**:

```bash
pod install
./Scripts/fetch_ios_demo_assets.sh
```

`Scripts/fetch_ios_demo_assets.sh` downloads ONNX bundles into **`PaddleOCRDemo/Models/`** and fetches demo images into **`PaddleOCRDemo/Resources/SampleImages/`**. Intermediate `.tar` caches are stored under **`.fetch_ios_demo_assets_work/`** at the project root.

Optionally, pass the **model preset** (bundle name such as `PP-OCRv6_small`) as a positional argument after any options:

```bash
./Scripts/fetch_ios_demo_assets.sh PP-OCRv6_small
```

Currently, the supported model presets are `PP-OCRv6_small` and `PP-OCRv6_tiny`. The default preset is `PP-OCRv6_small`.

Flags:

| Flag | Meaning |
|------|---------|
| `--models-only` | ONNX models only |
| `--samples-only` | Sample image only |

## Open in Xcode

```bash
open PaddleOCRDemo.xcworkspace
```

If you use CocoaPods, run `pod install` in the project root first so the workspace is generated next to the `Podfile`.

Build the **PaddleOCRDemo** scheme. Ensure **`PaddleOCRDemo/Models/`** and **`PaddleOCRDemo/Resources/SampleImages/`** are included via folder references / **Copy Bundle Resources** (as in the checked-in project).

## Validation

Validation covers two independent tracks:

1. **Accuracy (quality)** — same image and ONNX semantics on **Python** (`engine="onnxruntime"`, no doc preprocessor modules) and **iOS**, then compare JSON with IoU + character error rate.
2. **On-device runtime performance** — **latency** (timing summaries in JSON / report) and **memory** (physical footprint: load state + per-inference samples).

You can run either track alone, or both. To archive results in one place, use **`Scripts/generate_validation_report.py`** (see below).

Shared Python extras:

```bash
python3 -m pip install -r Scripts/requirements-validation.txt
```

### Accuracy validation

**1) Python reference JSON**:

Run **`./Scripts/fetch_ios_demo_assets.sh`** first (see [One-time asset setup](#one-time-asset-setup)) so **`PaddleOCRDemo/Models/`** exists for the default **`--ios-models-root`**. Install PaddleOCR with the ONNX Runtime engine, then run:

```bash
python3 Scripts/ocr_reference_run.py \
  --image PaddleOCRDemo/Resources/SampleImages/general_ocr_002.png \
  --output /tmp/ref.json \
  --device cpu \
  --align-ios-defaults
```

**`--image`** may be any path to a test image (the example uses a sample from the asset script); use that **same** file in the iOS step when comparing.

**`--ios-models-root`** defaults to **`PaddleOCRDemo/Models`**. Override it if your ONNX tree lives elsewhere.

**2) iOS JSON export** — run **`OCRBenchmarkTests`** / **`testOCRExportJSONSchema`** on a device or simulator with models bundled. Export schema matches the reference (`schema_version`, `source`, `items[]` with `polygon`, `text`, `score`).

Environment variables (scheme → **Arguments → Environment Variables** or `xcodebuild`):

| Variable | Meaning |
|----------|---------|
| `PADDLEOCR_VALIDATION_IMAGE_PATH` | **Required.** Absolute path to the test image (PNG or JPEG); must be the **same file** as the Python `--image` argument so results are comparable. |
| `PADDLEOCR_VALIDATION_EXPORT_JSON` | Optional. Absolute path to write JSON from `testOCRExportJSONSchema` (for `compare_ocr_json.py`). If unset, the test still runs but does not write a file. |

**3) Compare** — write a JSON summary for the validation report:

```bash
python3 Scripts/compare_ocr_json.py /tmp/ref.json /tmp/ios.json \
  --iou-threshold 0.5 \
  --cer-threshold 0.08 \
  --json-summary-out /tmp/compare-summary.json
```

Exit code **`0`** means **PASS**; non-zero means thresholds were exceeded.

### On-device runtime performance

Run **`OCRBenchmarkTests`** / **`testOCRBenchmarkTimings`** on a **physical device** (recommended). After warmup, the test records **runtime performance** in two parts:

- **Latency**: mean / stdev / p90 (ms) for each timing field in the exported JSON .
- **Memory (resource footprint)**: `task_vm_info` **physical footprint** (`phys_footprint`) — aligned with Xcode’s Memory gauge — sampled before session setup, after model loading, and immediately before/after each measured inference. **Peak** = max of those samples per iteration; **mean** = mean of post-inference samples. This does **not** replace Instruments for allocator spikes inside native code; it gives repeatable regression numbers for load + steady inference.

Prefer **Release** and avoid the debugger when recording.

Environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `PADDLEOCR_VALIDATION_IMAGE_PATH` | — | **Required.** Absolute path to the image (PNG or JPEG) to run OCR on. |
| `PADDLEOCR_VALIDATION_WARMUP_ITERATIONS` | `3` | Non-negative integer. Warmup runs (excluded from timing and memory stats). |
| `PADDLEOCR_VALIDATION_MEASURED_ITERATIONS` | `10` | Non-negative integer. Measured runs for timing stats and inference memory stats. |
| `PADDLEOCR_VALIDATION_ON_DEVICE_PERFORMANCE_JSON_PATH` | — | Optional. If set, writes the full JSON (timing + memory + `thermalState`) to this path for `Scripts/generate_validation_report.py`. |

### Validation report

After you have optional inputs from the **accuracy** and/or **on-device** steps, merge them into a single Markdown file:

| Input | Produced by |
|-------|-------------|
| `--compare-summary` | `compare_ocr_json.py`, with `--json-summary-out` to capture the same JSON as stdout (includes `pass`) |
| `--on-device-performance-json` | `testOCRBenchmarkTimings`, when `PADDLEOCR_VALIDATION_ON_DEVICE_PERFORMANCE_JSON_PATH` is set |

Either flag may be omitted; missing sections appear as short placeholders in the report. **App Store / download size** is not part of this report — use Xcode’s **App Thinning Size Report** or **App Store Connect** ([Reducing your app’s size](https://developer.apple.com/documentation/xcode/reducing-your-app-s-size)).

```bash
python3 Scripts/generate_validation_report.py \
  --compare-summary /tmp/compare-summary.json \
  --on-device-performance-json /tmp/on-device-performance.json \
  --output out/validation-report.md
```
