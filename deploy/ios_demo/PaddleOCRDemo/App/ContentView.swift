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

/// Root view: routes `AppState` and lays out **input (photo/sample)** vs **output (preview + text)**; engine tuning lives in ``EngineSettingsSheet``.
struct ContentView: View {
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @StateObject private var viewModel = OCRViewModel()
    @State private var selectedItem: PhotosPickerItem?
    @State private var showEngineSettings = false

    private var navigationTitleDisplayMode: NavigationBarItem.TitleDisplayMode {
        switch viewModel.state {
        case .ready: return .large
        case .error(let err) where !err.isModelError: return .large
        default: return .inline
        }
    }

    private var showsAdvancedToolbarItem: Bool {
        switch viewModel.state {
        case .loadingModels, .processing: return false
        case .ready, .results, .error: return true
        }
    }

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
            .navigationBarTitleDisplayMode(navigationTitleDisplayMode)
            .toolbar {
                if showsAdvancedToolbarItem {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button {
                            showEngineSettings = true
                        } label: {
                            Image(systemName: "slider.horizontal.3")
                        }
                        .accessibilityLabel("Advanced settings")
                    }
                }
            }
        }
        .sheet(isPresented: $showEngineSettings) {
            EngineSettingsSheet(viewModel: viewModel)
        }
        .task {
            await viewModel.loadModels()
        }
        .onChange(of: selectedItem) { newItem in
            guard let item = newItem else { return }
            Task {
                guard let data = try? await item.loadTransferable(type: Data.self),
                      let uiImage = UIImage(data: data) else {
                    viewModel.reportImageLoadFailed()
                    return
                }
                selectedItem = nil
                await viewModel.processImage(uiImage)
            }
        }
    }

    private func imageInputSection(title: String, subtitle: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            DemoSectionHeader(title: title, subtitle: subtitle)
            ImagePickerSection(
                selectedItem: $selectedItem,
                sampleImageNames: viewModel.sampleImageNames,
                onSampleSelected: { name in
                    Task { await viewModel.selectSampleImage(named: name) }
                }
            )
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
        case .results(let result, let image, _):
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
                    Text("Choose a photo or open a sample. Engine and OCR tuning are under Advanced in the toolbar.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            imageInputSection(
                title: "Image",
                subtitle: "Choose from your library or a bundled sample."
            )
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

    private func resultsView(result: OCRRunResult, image: UIImage) -> some View {
        // Only use two columns when explicitly regular (e.g. iPad); treat nil as compact.
        let isCompact = horizontalSizeClass != .regular
        return VStack(spacing: 22) {
            if isCompact {
                previewBlock(result: result, image: image)
                TimingView(result: result)
                    .padding(.vertical, 4)
            } else {
                TimingView(result: result)
                    .padding(.vertical, 4)
                HStack(alignment: .top, spacing: 16) {
                    previewBlock(result: result, image: image)
                        .frame(maxWidth: .infinity)
                        .layoutPriority(1)
                    resultsListBlock(result: result)
                        .frame(minWidth: 280, maxWidth: .infinity)
                }
            }

            if isCompact {
                resultsListBlock(result: result)
            }

            imageInputSection(
                title: "New image",
                subtitle: "Choose another photo or sample. Re-run uses the current image and OCR parameters."
            )

            rerunOCRButton()
        }
    }

    @ViewBuilder
    private func rerunOCRButton() -> some View {
        if viewModel.hasPendingOcrChanges {
            Button {
                Task { await viewModel.rerunOCR() }
            } label: {
                Label("Re-run OCR on this image", systemImage: "arrow.clockwise.circle.fill")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 4)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        } else {
            Button {
                Task { await viewModel.rerunOCR() }
            } label: {
                Label("Re-run OCR on this image", systemImage: "arrow.clockwise.circle.fill")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 4)
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
        }
    }

    private func previewBlock(result: OCRRunResult, image: UIImage) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            DemoSectionHeader(
                title: "Preview",
                subtitle: "Accent overlay shows each detected text region."
            )
            DemoCard {
                ResultImageView(image: image, results: result.results)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            }
        }
    }

    private func resultsListBlock(result: OCRRunResult) -> some View {
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
