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

import SwiftUI

/// Advanced ONNX Runtime and OCR parameters, presented as a sheet from the main screen toolbar.
struct EngineSettingsSheet: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var viewModel: OCRViewModel
    @State private var ortSessionReloadConfirmed = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    DemoSectionHeader(
                        title: "ONNX Runtime",
                        subtitle: "Choose execution provider and options, then reload to apply."
                    )
                    DemoCard {
                        OnnxRuntimeSettingsPanel(
                            primaryExecutionProvider: $viewModel.ortPrimaryExecutionProvider,
                            tuning: $viewModel.ortSessionTuning
                        )
                    }

                    DemoSectionHeader(
                        title: "OCR parameters",
                        subtitle: "These apply to the next OCR run."
                    )
                    DemoCard {
                        OCRParametersPanel(
                            params: $viewModel.runtimeParams,
                            resolvedBaseline: viewModel.resolvedRuntimeBaseline ?? .fallbackForUI
                        )
                    }
                }
                .padding(20)
                .padding(.bottom, viewModel.hasUnappliedOrtSettings ? 120 : 24)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Engine & parameters")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                if viewModel.hasUnappliedOrtSettings {
                    VStack(alignment: .leading, spacing: 12) {
                        Divider()
                        Text("Sessions still use the previous ONNX Runtime choices until you reload.")
                            .font(.footnote)
                            .foregroundStyle(Color.orange.opacity(0.95))
                            .fixedSize(horizontal: false, vertical: true)

                        HStack(spacing: 12) {
                            Button("Revert draft") {
                                viewModel.revertOrtDraftToLoaded()
                            }
                            .disabled(!viewModel.allowsOrtApplyControls || !viewModel.hasUnappliedOrtSettings)

                            Button {
                                ortSessionReloadConfirmed = true
                            } label: {
                                Label("Apply & reload", systemImage: "arrow.clockwise.circle")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(!viewModel.allowsOrtApplyControls || !viewModel.hasUnappliedOrtSettings)
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.vertical, 12)
                    .background(.ultraThinMaterial)
                }
            }
        }
        .confirmationDialog(
            "Rebuild ONNX Runtime sessions?",
            isPresented: $ortSessionReloadConfirmed,
            titleVisibility: .visible
        ) {
            Button("Reload models") {
                Task { await viewModel.loadModels() }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Sessions are recreated from the bundle. Expect a short pause.")
        }
    }
}
