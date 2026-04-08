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

import PhotosUI
import SwiftUI

/// Root view that coordinates all UI states based on `OCRViewModel.state`.
///
/// This view acts as a state router: it renders the appropriate sub-view
/// for each `AppState` case and wires up the PhotosPicker + sample image
/// selection to the view model's processing pipeline.
///
/// Plan 02 will replace the placeholder results view with proper
/// ResultImageView, ResultsListView, TimingView, and ErrorView components.
struct ContentView: View {
    @StateObject private var viewModel = OCRViewModel()
    @State private var selectedItem: PhotosPickerItem?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    contentForState
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 16)
            }
            .navigationTitle("PaddleOCR Demo")
            .navigationBarTitleDisplayMode(.inline)
        }
        .task {
            await viewModel.loadModels()
        }
        .onChange(of: selectedItem) { _, newItem in
            guard let item = newItem else { return }
            Task {
                guard let data = try? await item.loadTransferable(type: Data.self),
                      let uiImage = UIImage(data: data) else {
                    viewModel.state = .error(.imageLoadFailed)
                    return
                }
                selectedItem = nil // Reset for re-selection
                await viewModel.processImage(uiImage)
            }
        }
    }

    // MARK: - State Router

    @ViewBuilder
    private var contentForState: some View {
        switch viewModel.state {
        case .loadingModels:
            loadingModelsView
        case .ready:
            readyView
        case .processing(let image):
            processingView(image: image)
        case .results(let result, let image):
            resultsView(result: result, image: image)
        case .error(let error):
            errorView(error: error)
        }
    }

    // MARK: - Loading Models

    private var loadingModelsView: some View {
        VStack(spacing: 12) {
            ProgressView()
                .scaleEffect(1.5)
                .padding(.bottom, 4)
            Text("Loading Models")
                .font(.title2)
                .fontWeight(.semibold)
            Text("Preparing detection and recognition models...")
                .font(.body)
                .foregroundColor(Color(.secondaryLabel))
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 80)
    }

    // MARK: - Ready

    private var readyView: some View {
        VStack(spacing: 16) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 36))
                .foregroundColor(.green)
            Text("Models Ready")
                .font(.headline)
            Text("Select a photo or tap a sample image to start.")
                .font(.subheadline)
                .foregroundColor(Color(.secondaryLabel))
                .multilineTextAlignment(.center)

            ImagePickerSection(
                selectedItem: $selectedItem,
                sampleImageNames: viewModel.sampleImageNames,
                onSampleSelected: { name in
                    Task { await viewModel.selectSampleImage(named: name) }
                }
            )
        }
        .padding(.top, 40)
    }

    // MARK: - Processing

    private func processingView(image: UIImage) -> some View {
        VStack(spacing: 16) {
            ZStack {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 400)
                    .clipShape(RoundedRectangle(cornerRadius: 12))

                // Overlay spinner
                VStack(spacing: 8) {
                    ProgressView()
                        .scaleEffect(1.5)
                        .tint(.white)
                    Text("Running OCR...")
                        .font(.headline)
                        .foregroundColor(.white)
                }
                .padding(24)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            }
        }
    }

    // MARK: - Results (placeholder -- Plan 02 replaces with proper visualization)

    private func resultsView(result: OCRPipelineResult, image: UIImage) -> some View {
        VStack(spacing: 16) {
            // Image display (no polygon overlay yet -- Plan 02)
            Image(uiImage: image)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxHeight: 400)
                .clipShape(RoundedRectangle(cornerRadius: 12))

            // Results heading
            Text("Results (\(result.results.count))")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            // Timing summary
            Text("Detection: \(ms(result.detectionTime)) ms | Recognition: \(ms(result.recognitionTime)) ms | Total: \(ms(result.totalTime)) ms")
                .font(.caption)
                .foregroundColor(Color(.secondaryLabel))
                .frame(maxWidth: .infinity, alignment: .leading)

            // Text results list
            if result.results.isEmpty {
                Text("No text detected.")
                    .font(.body)
                    .foregroundColor(Color(.secondaryLabel))
                    .padding(.vertical, 8)
            } else {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(Array(result.results.enumerated()), id: \.offset) { idx, item in
                        Text("\(idx + 1). \(item.text) -- \(String(format: "%.1f%%", item.confidence * 100))")
                            .font(.body)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }

            // Image picker at bottom for selecting a new image
            ImagePickerSection(
                selectedItem: $selectedItem,
                sampleImageNames: viewModel.sampleImageNames,
                onSampleSelected: { name in
                    Task { await viewModel.selectSampleImage(named: name) }
                }
            )
        }
    }

    // MARK: - Error

    private func errorView(error: AppError) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 48))
                .foregroundColor(.red)

            Text(errorHeading(for: error))
                .font(.headline)

            Text(error.localizedDescription ?? "An unknown error occurred.")
                .font(.body)
                .foregroundColor(Color(.secondaryLabel))
                .multilineTextAlignment(.center)

            Button {
                Task { await viewModel.retry() }
            } label: {
                Label("Retry", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.borderedProminent)
            .padding(.top, 4)

            // If not a model error, let user try a different image
            if !error.isModelError {
                ImagePickerSection(
                    selectedItem: $selectedItem,
                    sampleImageNames: viewModel.sampleImageNames,
                    onSampleSelected: { name in
                        Task { await viewModel.selectSampleImage(named: name) }
                    }
                )
            }
        }
        .padding(.top, 40)
    }

    // MARK: - Helpers

    private func errorHeading(for error: AppError) -> String {
        switch error {
        case .modelLoadFailed: return "Model Loading Failed"
        case .inferenceFailed: return "OCR Failed"
        case .imageLoadFailed: return "Image Error"
        }
    }

    private func ms(_ t: TimeInterval) -> String {
        String(format: "%.0f", t * 1000)
    }
}
