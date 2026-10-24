---
comments: true
---

# iOS 部署

PaddleOCR 在 [`deploy/ios_demo`](https://github.com/PaddlePaddle/PaddleOCR/tree/{{PADDLEOCR_GITHUB_REF}}/deploy/ios_demo) 提供基于 **SwiftUI** 的 iOS OCR 示例应用。应用在设备上使用导出的 **ONNX** 模型，通过 [ONNX Runtime Objective-C API](https://onnxruntime.ai/docs/tutorials/mobile/) 完成检测与识别推理。

本指南介绍该 Demo 的环境要求、模型资源准备、在 Xcode 中构建运行，以及可选的 ORT 格式转换与基准测试流程。

## 工程结构

应用源码、打包资源及为本 Demo 引入的第三方**源码**均位于 **`PaddleOCRDemo/`** 下。单元测试位于与 Xcode 工程同级的 **`PaddleOCRDemoTests/`**。工程根目录还包含 `Podfile`、`scripts/`、`README.md` 与 `NOTICE`。

## 环境要求

- 搭载 **macOS** 的开发机，**Xcode 16.0+**（部署目标 **iOS 16+**）
- **CocoaPods**（`gem install cocoapods` 或 Homebrew）
- `curl`、`tar`

## 支持的模型预设

`scripts/fetch_ios_demo_models.sh` 会将 ONNX 模型包下载到 **`PaddleOCRDemo/Models/`**。当前支持的预设为：

- **`PP-OCRv6_small`**（默认）
- **`PP-OCRv6_tiny`**
- **`PP-OCRv5_mobile`**

## 快速开始 {#快速开始}

在 **工程根目录**（`deploy/ios_demo`）执行：

```bash
pod install
./scripts/fetch_ios_demo_models.sh
```

可选：通过位置参数指定模型预设（即 bundle 名称，如 `PP-OCRv6_small`）：

```bash
./scripts/fetch_ios_demo_models.sh PP-OCRv6_small
```

中间 `.tar` 缓存目录为工程根下的 **`.fetch_ios_demo_models_work/`**。

使用 CocoaPods 时，需先执行 `pod install` 再打开工作区：

```bash
open PaddleOCRDemo.xcworkspace
```

构建 **PaddleOCRDemo** scheme。请确认 **`PaddleOCRDemo/Models/`** 与 **`PaddleOCRDemo/Resources/SampleImages/`** 已通过文件夹引用 / **Copy Bundle Resources** 加入 App target；**`PaddleOCRDemoTests/Fixtures/`** 已加入测试 target（与仓库中工程配置一致）。内置样例图为 **`general_ocr_002.jpg`**。

### ONNX Runtime 执行提供程序（EP）

界面上的 **CPU** / **XNNPACK** / **Core ML** 分段控件用于在创建 Session 时指定优先使用的 EP。当某一 EP 无法覆盖部分算子时，ONNX Runtime 仍可能回退到默认 **CPU** EP。选择 **CPU** 时不会注册 XNNPACK 或 Core ML EP。

## 转换为 ORT 模型格式（可选）

生成 [ORT 格式](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html) 权重：

```bash
python3 -m pip install -r requirements-onnx-convert-ort.txt
./scripts/convert_onnx_to_ort.sh
```

默认在 `PaddleOCRDemo/Models/` 下每个 `inference.onnx` 旁生成 `inference*.ort`；Demo 在存在 `.ort` 时会优先加载，因此可不将 ONNX 打入 App 包。若只希望保留一种权重文件，可手动删除 `inference.onnx`，或用 **`--output-dir`** 将 `.ort` 输出到单独目录后再替换 `PaddleOCRDemo/Models/` 中的文件。省略 **`--input-dir`** 时默认输入目录为 `PaddleOCRDemo/Models`。

默认使用 **`--optimization-style Fixed`**（每个 ONNX 对应一个 `.ort`）。底层 Python 工具若未指定，会同时生成 Fixed 与 Runtime 两种（`*.ort` 与 `*.with_runtime_opt.ort`）。需要 Runtime 风格（如面向 Core ML 的导出）时，传入 **`--optimization-style Runtime`**（别名 `--optimization_style`）；可同时传入 **`--optimization-style Fixed Runtime`**。

转换会生成 **`required_operators*.config`**（用于裁剪最小 ORT 构建）；**默认在脚本成功结束后删除**。使用 **`--keep-operator-config`** 可保留。这些文件**不参与** Demo 推理，仅加载 **`.ort`**（或 `.onnx`）权重。

示例（Runtime，输出到独立目录）：

```bash
./scripts/convert_onnx_to_ort.sh --output-dir ./out/ort_bundles --optimization-style Runtime
```

## 基准测试（Benchmark）

Demo 提供在真机或模拟器上测量 OCR 时延与内存的流水线。

### 前置条件

先完成 [快速开始](#快速开始) 中的资源准备。真机基准测试还需要：

- **代码签名（仅真机）**：`xcodebuild` 部署到物理设备需要有效的 Apple Development 证书与描述文件。建议在 Xcode 中打开 `PaddleOCRDemo.xcworkspace`，在 *Signing & Capabilities* 中选择 Development Team 并启用自动管理；之后 `run_benchmark.sh` 等命令可复用已缓存的配置。首次在设备上启动时，须在 **设置 → 通用 → VPN 与设备管理** 中信任开发者证书（每台设备一次）。模拟器无需签名。
- 可选精度预检：主机安装 PaddleOCR（ONNX Runtime 引擎）用于生成参考结果，并执行 `python3 -m pip install -r requirements-accuracy.txt`。

### 完整流水线

在工程根目录执行 **`./scripts/run_benchmark.sh`**。脚本会解析输入图像、可选执行精度预检、在模拟器或真机上运行 XCTest 基准、提取产物并生成 Markdown 报告。始终以 **Release** 配置调用 `xcodebuild test`。

**配置方式**：优先使用脚本参数。`--image` 将任意路径图像复制到本次运行的 `Fixtures/local-*`；`--fixture` 选择 `PaddleOCRDemoTests/Fixtures/` 下已有文件。若两者均未指定，则要求 `Fixtures/` 下恰好有一张非 `local-*` 的图像（常见为 `ios_ocr_benchmark_reference.jpg`）。

| 用途 | 参数 |
| --- | --- |
| 运行目标 | `--udid <id>`（真机）或 `--simulator <name>` |
| 任意路径图像 | `--image <path>` |
| 已有 Fixture | `--fixture <name>`（stem 或 `stem.ext`） |
| 强度 | `--warmup <n>`、`--measured-iterations <n>` |
| 优先 ORT EP | `--ort-execution-provider CPU`、`XNNPACK` 或 `CORE_ML` |
| ORT profiling JSON | `--ort-profiling` |
| 可选精度预检 | `--accuracy-check`，可选 `--accuracy-reference-json <path>` |
| 精度 FAIL 时中止 | `--accuracy-check --stop-on-accuracy-failure` |
| 输出目录 | `--output-dir <dir>`（默认 `out/`） |
| 清理旧产物 | `--clean` |

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

`--accuracy-check` 在基准测试前以独立 XCTest 运行。结果为 `PASS`、`FAIL` 或 `ERROR`。默认 `FAIL` 仅记录并继续基准；`ERROR` 会中止流水线。`--stop-on-accuracy-failure` 会在 `FAIL` 时跳过基准并返回非零退出码。

**ORT Session profiling**（附件 `ort_profile_detection`、`ort_profile_recognition`）：使用 `--ort-profiling`。Profiling 会改变运行时行为，应与干净时延测量**分开**运行。

基准**成功完成**后，脚本会在输出目录（默认 `out/`）覆盖写入 `run-status.json`、`on-device-performance.json`、`xctest-memory-metrics.json`、`benchmark-report.md`。若流水线提前 **ERROR**，这些文件可能缺失或不完整。

报告内容包括模型输入张量形状分布、首次测量运行的行数、推断的模型预设、实际模型格式（`onnx` 或 `ort`）、det/rec/总权重大小、在可解析时的 App 可执行文件大小、冷启动模型加载时间、测量时延与内存等。形状统计在测量循环内按模型调用计数：每次完整 OCR 检测贡献 1 次，识别按 batch 各贡献 1 次。

| 产物 | 来源 | 用途 |
|---|---|---|
| `accuracy-reference.json` | `ocr_reference_run.py` | 可选参考 JSON（除非提供 `--accuracy-reference-json`） |
| `accuracy-result.xcresult` | `xcodebuild test` | 可选精度预检 |
| `ios-ocr-export.json` | `extract_xcresult_attachments.py` | 从精度预检提取的 iOS 精度 payload |
| `accuracy-summary.json` | `compare_ocr_json.py` | 精度预检摘要 |
| `latency-result.xcresult` | `xcodebuild test` | 时延基准 |
| `memory-result.xcresult` | `xcodebuild test` | 内存基准 |
| `on-device-performance.json` | `extract_xcresult_attachments.py` | 从 `latency-result.xcresult` 提取的时延统计 |
| `ort_profile_detection`、`ort_profile_recognition` | `extract_xcresult_attachments.py` | 可选 ORT profiling JSON |
| `xctest-memory-metrics.json` | `extract_xctest_metrics.py` | XCTest 内存指标 |
| `logs/*.log` | `run_benchmark.sh` | 各步骤日志 |
| `run-status.json` | `run_benchmark.sh` | 步骤状态与元数据 |
| `benchmark-report.md` | `generate_benchmark_report.py` | 可读报告 |

流水线状态为 **`COMPLETED`** 时退出码 **`0`**；**`ERROR`** 时非零。精度预检 `FAIL` 仅在使用 `--stop-on-accuracy-failure` 时改变退出码；`ERROR` 始终中止。

### 进阶：XCTest 环境变量

多数场景应使用 `run_benchmark.sh` 参数，而非直接设置环境变量。本节适用于在 Xcode 或自定义 `xcodebuild test` 中调试 **PaddleOCRDemoTests**。

基准测试通过 **`PADDLEOCR_BENCHMARK_*`** 读取配置（作用于 XCTest runner，不会自动传入交互式 shell）。

| 启动方式 | 配置方法 |
| --- | --- |
| **Xcode** | Scheme **PaddleOCRDemo** → **Test** → **Arguments** → **Environment Variables**，变量名如下。 |
| **自定义 `xcodebuild test`** | 使用 **`TEST_RUNNER_` + 变量名**（如 `TEST_RUNNER_PADDLEOCR_BENCHMARK_IMAGE_NAME=…`）。 |

| 变量 | 未设置时 | 作用 |
| --- | --- | --- |
| `PADDLEOCR_BENCHMARK_IMAGE_NAME` | **`ios_ocr_benchmark_reference`** | `Fixtures/` 下 bundled 图像（stem 或 `stem.ext`）。 |
| `PADDLEOCR_BENCHMARK_WARMUP_ITERATIONS` | **3** | 计时时前的预热完整 OCR 次数。 |
| `PADDLEOCR_BENCHMARK_MEASURED_ITERATIONS` | **10** | 计时迭代次数。 |
| `PADDLEOCR_BENCHMARK_ORT_EXECUTION_PROVIDER` | **CPU** | **`CPU`**、**`XNNPACK`** 或 **`CORE_ML`**。 |
| `PADDLEOCR_BENCHMARK_ORT_PROFILING` | （未设置） | 启用 ORT session profiling（`1` / `true` / `yes` / `on`）。 |
| `PADDLEOCR_BENCHMARK_ONLY_TESTING_SCOPE` | latency + memory 基准测试 | 仅 `run_benchmark.sh`：传给多次 `xcodebuild -only-testing` 的逗号分隔列表。 |

两个迭代相关变量须为非负整数。

## 第三方许可

随 Demo 打包的 **Clipper**（polyclipping 6.4.2）采用 [Boost Software License 1.0](https://www.boost.org/LICENSE_1_0.txt)；详见 `NOTICE` 与 `PaddleOCRDemo/ThirdParty/Clipper1/LICENSE`。CocoaPods 依赖的许可以 `pod install` 后的 `Podfile.lock` 及各自仓库为准。
