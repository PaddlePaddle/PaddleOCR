# PaddleOCR-VL Documentation IA Validation

## Structural checks

- [x] Main guides expose `## Start Here` / `## 从这里开始`
- [x] Main guides expose `## Workflow Guide for This Tutorial` / `## 本教程支持的使用目标`
- [x] All English hardware guides expose `## Workflow Guide for This Hardware`
- [x] All Chinese hardware guides expose `## 本硬件支持的使用目标`
- [x] The old main-guide Mermaid flowchart is gone
- [x] The old hardware-guide round-trip tips are gone
- Evidence:
  - The required structural `rg` validation returned only the new main-guide and hardware-guide headings across the bilingual guide set.
  - That same validation returned no `flowchart TD`, `Before reading this hardware-specific tutorial`, or `建议先阅读 .*流程导览` matches.

## Execution artifacts

- [x] Support truth table still matches the edited docs
- [x] Old-to-new path mapping still matches the edited docs
- [x] The main-guide support matrix, support truth table, and Ascend/Intel hardware workflow tables all agree that Huawei Ascend NPU and Intel Arc GPU do not currently support local PaddlePaddle direct inference
- Evidence:
  - The artifact-consistency `rg` validation preserved the expected Apple manual-only route, the Blackwell Docker Compose or manual route, and the compose-based full-API routes in the relevant hardware guides.
  - English revalidation: `docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md` marks PaddlePaddle as `🚧` for Huawei Ascend NPU and Intel Arc GPU in the support matrix, while `docs/version3.x/pipeline_usage/PaddleOCR-VL-Huawei-Ascend-NPU.en.md` and `docs/version3.x/pipeline_usage/PaddleOCR-VL-Intel-Arc-GPU.en.md` both say `Not currently supported with local PaddlePaddle inference on this hardware` and route the supported path to Section 3.
  - Chinese revalidation: `docs/version3.x/pipeline_usage/PaddleOCR-VL.md` marks PaddlePaddle as `🚧` for `华为昇腾 NPU` and `Intel Arc GPU` in the support matrix, while `docs/version3.x/pipeline_usage/PaddleOCR-VL-Huawei-Ascend-NPU.md` and `docs/version3.x/pipeline_usage/PaddleOCR-VL-Intel-Arc-GPU.md` both say `当前不支持通过本地 PaddlePaddle 推理方式在本硬件上运行` and route the supported path to Section 3.
  - `docs/superpowers/plans/2026-04-22-paddleocr-vl-doc-ia-support-truth-table.md` still records both hardware paths as not supporting local PaddlePaddle direct inference, so the artifact remains aligned with all six bilingual doc pages above.

## Navigation and link sanity

- [x] New main-guide hardware entry links were rechecked in English and Chinese
- [x] Hardware-guide links back to the main support matrix were rechecked in English and Chinese
- [x] Apple Silicon manual-deployment handoff to the main guide was rechecked in English and Chinese
- Evidence:
  - `docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md` and `docs/version3.x/pipeline_usage/PaddleOCR-VL.md` still expose the new hardware entry tables with sibling-guide links for Blackwell, Apple Silicon, Kunlunxin XPU, Hygon DCU, MetaX GPU, Iluvatar GPU, Huawei Ascend NPU, AMD GPU, and Intel Arc GPU.
  - The support-matrix destination headings are still present as `## Inference Device Support for PaddleOCR-VL` in `docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md` and `## PaddleOCR-VL 对推理设备的支持情况` in `docs/version3.x/pipeline_usage/PaddleOCR-VL.md`, and the Apple/Ascend/Intel hardware guides in both languages still link back to those main-guide support-matrix anchors.
  - `docs/version3.x/pipeline_usage/PaddleOCR-VL-Apple-Silicon.en.md` still hands manual-deployment readers to `docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md` Section `4.2 Method 2: Manual Deployment`, and `docs/version3.x/pipeline_usage/PaddleOCR-VL-Apple-Silicon.md` still hands them to `docs/version3.x/pipeline_usage/PaddleOCR-VL.md` Section `4.2 方法二：手动部署`; both target sections are still present in the main guides.

## Walkthrough checks

- [x] `x64 CPU -> local direct inference`
- [x] `NVIDIA GPU (except Blackwell) -> full API service`
- [x] `NVIDIA Blackwell -> client + VLM service`
- [x] `Apple Silicon -> full API service`
- [x] `AMD GPU -> full API service`
- [x] `Kunlunxin XPU -> client + VLM service`
- [x] `Huawei Ascend NPU -> client + VLM inference service`
- [x] `Intel Arc GPU -> client + VLM inference service`
- Evidence:
  - The main guide workflow table keeps the x64 CPU local direct-inference route in Sections 1-2 and routes non-Blackwell NVIDIA full-API users to Section 4 with the documented Docker Compose or manual split.
  - The Blackwell guide keeps the client + VLM service route in Section 3 after local direct inference.
  - The Apple Silicon guide marks full API service as manual-deployment-only and sends the reader to shared manual deployment details without routing back through a global process guide.
  - The AMD GPU guide keeps full API service local to the hardware guide through Section 4.
  - The Kunlunxin XPU guide keeps the client + VLM service route local to Section 3 after local direct inference.
  - The Huawei Ascend NPU and Intel Arc GPU guides mark local direct inference unsupported on current local PaddlePaddle hardware paths, while keeping the supported client + VLM inference service path in Section 3.

## Result

- [x] All checks passed without sending the reader back to a global process guide
- Scope note: this record is limited to the targeted routing rewrite validation. Known pre-existing full-site MkDocs strict-mode warnings, mainly historical `version2.x` link issues, were treated as background noise and not counted as failures for this note.
