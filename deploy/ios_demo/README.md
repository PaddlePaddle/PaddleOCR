# iOS Demo

SwiftUI demo that runs OCR on device using exported ONNX models and [ONNX Runtime Objective-C API](https://onnxruntime.ai/docs/tutorials/mobile/) (CoreML + XNNPACK execution providers).

## Third-party licenses

Bundled **Clipper** (polyclipping 6.4.2) is under the [Boost Software License 1.0](https://www.boost.org/LICENSE_1_0.txt); see `NOTICE` and `ThirdParty/Clipper1/LICENSE`. CocoaPods pods are governed by their respective licenses (see `Podfile.lock` after `pod install`).

## Prerequisites

- macOS with Xcode (iOS 16+)
- CocoaPods (`gem install cocoapods` or Homebrew)
- `curl`, `tar`

## One-time asset setup

```bash
pod install
./fetch_ios_demo_assets.sh
```

`fetch_ios_demo_assets.sh` downloads ONNX bundles into `Models/` and fetches demo images into `Resources/SampleImages/`.

Optionally, pass the model variant as a positional argument after any options (common CLI style):

```bash
./fetch_ios_demo_assets.sh PP-OCRv6_small
```

Currently, the supported model variants are `PP-OCRv6_small` and `PP-OCRv6_tiny`. The default variant is `PP-OCRv6_small` (see `ALLOWED_VARIANTS` in `fetch_ios_demo_assets.sh`).

Flags:

| Flag | Meaning |
|------|---------|
| `--models-only` | ONNX models only |
| `--samples-only` | Sample image only |

## Open in Xcode

```bash
open PaddleOCRDemo.xcworkspace
```

If you use CocoaPods, run `pod install` in this directory first so the workspace is generated next to the `Podfile`.

Build the **PaddleOCRDemo** scheme. Ensure **Models/** and **Resources/SampleImages/** are included via folder references / **Copy Bundle Resources** (as in the checked-in project).
