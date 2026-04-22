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
  - `PaddleOCR-VL.en.md` marks PaddlePaddle as `🚧` for Huawei Ascend NPU and Intel Arc GPU in the support matrix, the support truth table records local direct inference as not supported on those hardware paths, and both hardware workflow tables route readers from that limitation to the supported client + VLM service path in Section 3.

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
