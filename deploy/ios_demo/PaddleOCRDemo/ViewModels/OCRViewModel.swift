// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
import UIKit

// MARK: - App State

/// The complete set of states the app can be in.
/// Drives all UI rendering via a single `@Published` property on `OCRViewModel`.
enum AppState: Equatable {
    /// Models are being loaded from the bundle.
    case loadingModels
    /// Models are loaded and ready; waiting for the user to select an image.
    case ready
    /// An image is being processed by OCR.
    case processing(UIImage)
    /// OCR completed successfully with results and the processed image.
    /// `runId` distinguishes successive successful runs for ``Equatable`` without requiring ``OCRRunResult`` to conform.
    case results(OCRRunResult, UIImage, runId: UUID)
    /// An error occurred during model loading or inference.
    case error(AppError)

    static func == (lhs: AppState, rhs: AppState) -> Bool {
        switch (lhs, rhs) {
        case (.loadingModels, .loadingModels): return true
        case (.ready, .ready): return true
        case (.processing, .processing): return true
        case (.results(_, _, let runIdL), .results(_, _, let runIdR)): return runIdL == runIdR
        case (.error(let a), .error(let b)): return a == b
        default: return false
        }
    }
}

// MARK: - App Error

/// User-facing errors with recovery guidance.
enum AppError: Equatable, LocalizedError {
    case modelLoadFailed(String)
    case inferenceFailed(String)
    case imageLoadFailed

    var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let msg): return msg
        case .inferenceFailed(let msg): return msg
        case .imageLoadFailed: return "Could not load the selected image. Try a different one."
        }
    }

    /// Whether this error is recoverable by retrying model load (vs just selecting a new image).
    var isModelError: Bool {
        if case .modelLoadFailed = self { return true }
        return false
    }
}

// MARK: - OCRViewModel

/// Central view model that manages the complete app lifecycle.
///
/// State transitions:
/// ```
/// loadingModels --> ready --> processing --> results
///       |            ^           |            |
///       v            |           v            v
///     error ----retry---->    error       ready (reset)
/// ```
///
/// All state mutations happen on `@MainActor` through the single
/// `@Published var state` property, giving SwiftUI a single source of truth.
@MainActor
class OCRViewModel: ObservableObject {
    @Published var state: AppState = .loadingModels
    @Published var copiedFeedback: Bool = false
    @Published var runtimeParams = OCRRuntimeParams.noOverrides
    /// ONNX Runtime presets and tuning shown in UI (draft). Use **Apply & reload** in Advanced settings to reload sessions (`loadModels()`).
    @Published var ortPrimaryExecutionProvider: ORTPrimaryExecutionProvider = .cpu
    @Published var ortSessionTuning = ORTSessionTuningOptions.default
    /// Snapshot of Ort settings actually used by the loaded ``ocrEngine``.
    private var ortLoadedPrimaryExecutionProvider: ORTPrimaryExecutionProvider = .cpu
    private var ortLoadedSessionTuning: ORTSessionTuningOptions = .default
    @Published private(set) var resolvedRuntimeBaseline: ResolvedOCRRuntimeParams?

    private var sessionManager: ORTSessionManager?
    private var ocrEngine: OCREngine?
    /// Last image run through OCR (normalized orientation), for **Re-run OCR** without re-picking.
    private var lastNormalizedImage: UIImage?
    /// OCR runtime parameters used for the last successful run (drives ``hasPendingOcrChanges``).
    private var lastRunRuntimeParams: OCRRuntimeParams?

    /// Bundled sample resource names (stem or `stem.ext`) for ``selectSampleImage(named:)``.
    let sampleImageNames: [String] = [DemoAssets.defaultSampleImageStem]

    /// Draft Ort settings differ from the configuration used by loaded sessions.
    var hasUnappliedOrtSettings: Bool {
        ortPrimaryExecutionProvider != ortLoadedPrimaryExecutionProvider
            || ortSessionTuning != ortLoadedSessionTuning
    }

    /// `runtimeParams` differs from the last successful OCR run (only meaningful in ``AppState/results``).
    var hasPendingOcrChanges: Bool {
        guard case .results = state else { return false }
        guard let last = lastRunRuntimeParams else { return false }
        return runtimeParams != last
    }

    /// When `false`, disable Apply/Revert (reload would conflict with loading or OCR in flight).
    var allowsOrtApplyControls: Bool {
        switch state {
        case .loadingModels, .processing: return false
        case .ready, .results, .error: return true
        }
    }

    /// Restores ONNX Runtime controls to match the currently loaded sessions.
    func revertOrtDraftToLoaded() {
        ortPrimaryExecutionProvider = ortLoadedPrimaryExecutionProvider
        ortSessionTuning = ortLoadedSessionTuning
    }

    /// Photo picker or file load produced unusable image data.
    func reportImageLoadFailed() {
        state = .error(.imageLoadFailed)
    }

    // MARK: - Lifecycle

    /// Load ORT models from the app bundle.
    ///
    /// Creates an `ORTSessionManager`, loads both detection and recognition models,
    /// then creates the `OCREngine` for end-to-end OCR.
    func loadModels() async {
        state = .loadingModels
        do {
            let manager = ORTSessionManager()
            try await manager.loadModels(executionProvider: ortPrimaryExecutionProvider, tuning: ortSessionTuning)
            let engine = try OCREngine(sessionManager: manager)
            self.sessionManager = manager
            self.ocrEngine = engine
            self.resolvedRuntimeBaseline = engine.baselineRuntimeDefaults()
            ortLoadedPrimaryExecutionProvider = ortPrimaryExecutionProvider
            ortLoadedSessionTuning = ortSessionTuning
            state = .ready
        } catch {
            state = .error(.modelLoadFailed(error.localizedDescription))
        }
    }

    // MARK: - Image Processing

    /// Run full OCR on a user-selected image.
    ///
    /// Normalizes orientation first (camera photos often have EXIF rotation),
    /// then delegates to `OCREngine.run(_:)`.
    func processImage(_ uiImage: UIImage) async {
        let normalized = normalizeOrientation(uiImage)
        lastNormalizedImage = normalized
        state = .processing(normalized)
        guard let cgImage = normalized.cgImage else {
            state = .error(.imageLoadFailed)
            return
        }
        guard let engine = ocrEngine else {
            state = .error(.modelLoadFailed("OCR engine is not ready."))
            return
        }
        do {
            let result = try await engine.run(cgImage, params: runtimeParams)
            lastRunRuntimeParams = runtimeParams
            state = .results(result, normalized, runId: UUID())
        } catch {
            lastNormalizedImage = nil
            state = .error(.inferenceFailed(error.localizedDescription))
        }
    }

    /// Runs OCR again on the **last image** using the current `runtimeParams` (e.g. after tuning sliders).
    func rerunOCR() async {
        guard let image = lastNormalizedImage else { return }
        await processImage(image)
    }

    /// Load a bundled sample image by name and run OCR on it.
    func selectSampleImage(named name: String) async {
        guard let path = Bundle.main.path(forBundledImageNamed: name, subdirectory: "SampleImages") else {
            state = .error(.imageLoadFailed)
            return
        }
        do {
            guard let uiFromFile = UIImage(contentsOfFile: path) else {
                state = .error(.imageLoadFailed)
                return
            }
            let normalized = normalizeOrientation(uiFromFile)
            lastNormalizedImage = normalized
            state = .processing(normalized)
            guard let cgImage = normalized.cgImage else {
                state = .error(.imageLoadFailed)
                return
            }
            guard let engine = ocrEngine else {
                state = .error(.modelLoadFailed("OCR engine is not ready."))
                return
            }
            let result = try await engine.run(cgImage, params: runtimeParams)
            lastRunRuntimeParams = runtimeParams
            state = .results(result, normalized, runId: UUID())
        } catch {
            lastNormalizedImage = nil
            state = .error(.inferenceFailed(error.localizedDescription))
        }
    }

    // MARK: - Result Actions

    /// Copy all recognized text to the system clipboard, with brief visual feedback.
    func copyResultsToClipboard() {
        guard case .results(let result, _, _) = state else { return }
        let allText = result.results.map(\.text).joined(separator: "\n")
        UIPasteboard.general.string = allText
        copiedFeedback = true
        Task {
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            copiedFeedback = false
        }
    }

    // MARK: - Error Recovery

    /// Retry after an error. Model errors reload models; other errors return to ready state.
    func retry() async {
        guard case .error(let error) = state else { return }
        if error.isModelError {
            await loadModels()
        } else {
            state = .ready
        }
    }

    /// Return to the ready state (e.g. after viewing results, user wants to pick a new image).
    func reset() {
        lastNormalizedImage = nil
        lastRunRuntimeParams = nil
        state = .ready
    }
}

// MARK: - Image Orientation

/// Normalize EXIF orientation by redrawing the image in `.up` orientation.
///
/// Camera photos from the photo picker often carry non-`.up` orientation metadata.
/// CoreGraphics ignores EXIF orientation, so we must flatten it before OCR processing.
private func normalizeOrientation(_ image: UIImage) -> UIImage {
    guard image.imageOrientation != .up else { return image }
    let format = UIGraphicsImageRendererFormat()
    format.scale = image.scale
    let renderer = UIGraphicsImageRenderer(size: image.size, format: format)
    return renderer.image { _ in
        image.draw(in: CGRect(origin: .zero, size: image.size))
    }
}
