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

/// Root view: routes `AppState` and lays out **input (parameters + image)** vs **output (preview + text)** clearly.
struct ContentView: View {
    @StateObject private var viewModel = OCRViewModel()
    @State private var selectedItem: PhotosPickerItem?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 0) {
                    contentForState
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 20)
            }
            .background(Color(.systemGroupedBackground))
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
                selectedItem = nil
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
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.3)
                .tint(.accentColor)
            Text("Loading models")
                .font(.title3.weight(.semibold))
            Text("Preparing detection and recognition…")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 72)
        .padding(.bottom, 40)
    }

    // MARK: - Ready

    private var readyView: some View {
        VStack(spacing: 22) {
            statusPill(icon: "checkmark.circle.fill", tint: .green) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Ready")
                        .font(.title3.weight(.semibold))
                    Text("Set parameters, then choose a photo or open a sample.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            VStack(alignment: .leading, spacing: 10) {
                DemoSectionHeader(
                    title: "Parameters",
                    subtitle: "These apply to the next OCR run."
                )
                DemoCard {
                    OCRParametersPanel(
                        params: $viewModel.runtimeParams,
                        baseline: viewModel.thresholdBaseline ?? .fallbackForUI
                    )
                }
            }

            VStack(alignment: .leading, spacing: 10) {
                DemoSectionHeader(
                    title: "Image",
                    subtitle: "Primary action: choose from your library."
                )
                ImagePickerSection(
                    selectedItem: $selectedItem,
                    sampleImageNames: viewModel.sampleImageNames,
                    onSampleSelected: { name in
                        Task { await viewModel.selectSampleImage(named: name) }
                    }
                )
            }
        }
    }

    // MARK: - Processing

    private func processingView(image: UIImage) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Running OCR")
                .font(.title3.weight(.semibold))
            Text("Hold on — detection and recognition are in progress.")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            ZStack {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 360)
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))

                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(.ultraThinMaterial)

                VStack(spacing: 10) {
                    ProgressView()
                        .scaleEffect(1.25)
                        .tint(.accentColor)
                    Text("Working…")
                        .font(.headline)
                }
            }
            .frame(maxWidth: .infinity)
        }
    }

    // MARK: - Results

    private func resultsView(result: OCRPipelineResult, image: UIImage) -> some View {
        VStack(spacing: 22) {
            // —— Output zone: what you see —
            VStack(alignment: .leading, spacing: 10) {
                DemoSectionHeader(
                    title: "Preview",
                    subtitle: "Green overlay shows each detected text region."
                )
                DemoCard {
                    ResultImageView(image: image, results: result.results)
                        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                }
            }

            TimingView(result: result)
                .padding(.vertical, 4)

            // —— Control zone: tune & repeat on same image —
            VStack(alignment: .leading, spacing: 10) {
                DemoSectionHeader(
                    title: "Parameters",
                    subtitle: "Adjust, then re-run without picking a new photo."
                )
                DemoCard {
                    OCRParametersPanel(
                        params: $viewModel.runtimeParams,
                        baseline: viewModel.thresholdBaseline ?? .fallbackForUI
                    )
                }
            }

            Button {
                Task { await viewModel.rerunOCR() }
            } label: {
                Label("Re-run OCR on this image", systemImage: "arrow.clockwise.circle.fill")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 4)
            }
            .buttonStyle(.bordered)
            .controlSize(.large)

            // —— Text output —
            VStack(alignment: .leading, spacing: 10) {
                DemoSectionHeader(title: "Results", subtitle: nil)
                DemoCard {
                    ResultsListView(
                        results: result.results,
                        copiedFeedback: viewModel.copiedFeedback,
                        onCopy: { viewModel.copyResultsToClipboard() }
                    )
                }
            }

            VStack(alignment: .leading, spacing: 10) {
                DemoSectionHeader(
                    title: "New image",
                    subtitle: "Same controls as before — choose photo or try a sample."
                )
                ImagePickerSection(
                    selectedItem: $selectedItem,
                    sampleImageNames: viewModel.sampleImageNames,
                    onSampleSelected: { name in
                        Task { await viewModel.selectSampleImage(named: name) }
                    }
                )
            }
        }
    }

    // MARK: - Error

    private func errorView(error: AppError) -> some View {
        VStack(spacing: 22) {
            ErrorView(error: error, onRetry: {
                Task { await viewModel.retry() }
            })

            if !error.isModelError {
                VStack(alignment: .leading, spacing: 10) {
                    DemoSectionHeader(title: "Try another image", subtitle: nil)
                    ImagePickerSection(
                        selectedItem: $selectedItem,
                        sampleImageNames: viewModel.sampleImageNames,
                        onSampleSelected: { name in
                            Task { await viewModel.selectSampleImage(named: name) }
                        }
                    )
                }
            }
        }
        .padding(.top, 24)
    }

    // MARK: - Small chrome

    private func statusPill<Content: View>(icon: String, tint: Color, @ViewBuilder content: () -> Content) -> some View {
        HStack(alignment: .top, spacing: 14) {
            Image(systemName: icon)
                .font(.system(size: 28))
                .foregroundStyle(tint)
                .symbolRenderingMode(.hierarchical)
            content()
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        }
        .overlay {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(Color(.separator).opacity(0.35), lineWidth: 0.5)
        }
    }
}
