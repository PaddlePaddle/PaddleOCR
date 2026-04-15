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
        .onChange(of: selectedItem) { newItem in
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

    // MARK: - Results

    private func resultsView(result: OCRPipelineResult, image: UIImage) -> some View {
        VStack(spacing: 24) {
            // Image with polygon overlays
            ResultImageView(image: image, results: result.results)

            // Timing breakdown card
            TimingView(result: result)

            // Results list with copy button
            ResultsListView(
                results: result.results,
                copiedFeedback: viewModel.copiedFeedback,
                onCopy: { viewModel.copyResultsToClipboard() }
            )

            // Image picker for selecting a new image
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
        VStack(spacing: 24) {
            ErrorView(error: error, onRetry: {
                Task { await viewModel.retry() }
            })

            // If not a model error, show image picker so user can try different image
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

    private func ms(_ t: TimeInterval) -> String {
        String(format: "%.0f", t * 1000)
    }
}
