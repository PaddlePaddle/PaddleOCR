# PaddleOCR iOS Demo

## What This Is

An iOS demo application for PaddleOCR that showcases PP-OCRv5 model deployment on iOS devices. The demo lives in `deploy/ios_demo/` (replacing the existing placeholder) and targets developers who want to integrate PaddleOCR into their own iOS apps.

## Core Value

Developers can see PP-OCRv5 text detection and recognition running on an iOS device with clear, understandable code they can adapt for their own apps.

## Requirements

### Validated

- ✓ PP-OCRv5 models exist and work for text detection + recognition — existing
- ✓ PaddleOCR has a model export pipeline (Paddle → ONNX) — existing
- ✓ Android demo exists in `deploy/android_demo/` as reference pattern — existing
- ✓ C++ inference exists in `deploy/cpp_infer/` — existing

### Active

- [ ] iOS app that runs PP-OCRv5 text detection on device
- [ ] iOS app that runs PP-OCRv5 text recognition on device
- [ ] End-to-end OCR pipeline (detect → crop → recognize) on iOS
- [ ] Image picker to select photos from album and run OCR
- [ ] Visual display of detection results (bounding boxes + recognized text)
- [ ] Model conversion pipeline documented (PP-OCRv5 → iOS-compatible format)
- [ ] README with build instructions, architecture explanation, and integration guide
- [ ] Pre/post-processing logic matching the Python reference implementation

### Out of Scope

- PaddleLite integration — PaddleLite is no longer maintained
- Real-time camera OCR — not in v1 scope, can be added later
- Document structure analysis (PP-StructureV3) — focus on core OCR first
- Android demo updates — separate effort
- Model training on device — inference only
- App Store distribution — developer demo, not a product

## Context

- **Existing iOS demo**: `deploy/ios_demo/` currently only has a README pointing to the old Paddle-Lite-Demo external repo. No actual code. Will be replaced entirely.
- **PaddleOCR ecosystem**: Built on PaddlePaddle, delegates to PaddleX for pipeline execution. Models are in Paddle format and can be exported to ONNX.
- **PP-OCRv5**: Latest OCR model version in PaddleOCR. Detection (det) + recognition (rec) pipeline.
- **Reference implementations**: `deploy/android_demo/` (Java/Kotlin + PaddleLite), `deploy/cpp_infer/` (C++ with Paddle Inference). The android demo also uses PaddleLite, so it's not a direct template for the iOS approach.
- **Inference framework**: Needs research — must find a well-maintained iOS inference framework (not PaddleLite). Candidates include ONNX Runtime, Core ML, or others.

## Constraints

- **No PaddleLite**: Explicitly excluded — unmaintained. Must use an alternative inference framework.
- **Developer-facing**: Code clarity and documentation quality matter more than UI polish.
- **English UI**: Demo interface in English.
- **PP-OCRv5 models**: Must support the latest model version specifically.
- **Location**: Must live in `deploy/ios_demo/` within the PaddleOCR repo.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Replace `deploy/ios_demo/` instead of creating new folder | Existing folder is just a placeholder README, no real code to preserve | — Pending |
| Exclude PaddleLite | No longer maintained by Baidu | — Pending |
| Inference framework | Research needed — ONNX Runtime vs CoreML vs others | — Pending |
| UIKit vs SwiftUI | Research will recommend based on compatibility | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-03 after initialization*
