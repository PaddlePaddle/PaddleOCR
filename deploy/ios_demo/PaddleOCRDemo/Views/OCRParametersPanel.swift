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

/// Sliders for detection / recognition runtime parameters (content only; wrap with `DemoCard` in parent).
struct OCRParametersPanel: View {
    @Binding var params: OCRRuntimeParams
    let resolvedBaseline: ResolvedOCRRuntimeParams

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            groupTitle("Detection preprocess")
            intParamRow(
                title: "Limit side length",
                caption: "Side length limit for detection resize",
                range: 32...2048,
                value: intOverrideBinding(\.textDetLimitSideLen, fallback: resolvedBaseline.textDetLimitSideLen)
            )
            limitTypeRow(
                selection: stringOverrideBinding(\.textDetLimitType, fallback: resolvedBaseline.textDetLimitType)
            )
            intParamRow(
                title: "Max side length cap",
                caption: "After limit-side resize, caps the longer side if it is still above this",
                range: 500...8000,
                value: intOverrideBinding(\.textDetMaxSideLimit, fallback: resolvedBaseline.textDetMaxSideLimit)
            )

            groupTitle("Detection postprocess")
            paramRow(
                title: "Map threshold",
                caption: "Binarization on the detector heatmap",
                range: 0.05...0.95,
                value: floatOverrideBinding(\.textDetThresh, fallback: resolvedBaseline.textDetThresh)
            )
            paramRow(
                title: "Box score threshold",
                caption: "Minimum score to keep a box",
                range: 0.1...0.95,
                value: floatOverrideBinding(\.textDetBoxThresh, fallback: resolvedBaseline.textDetBoxThresh)
            )
            paramRow(
                title: "Unclip ratio",
                caption: "Expand recovered polygons",
                range: 0.5...3.0,
                value: floatOverrideBinding(\.textDetUnclipRatio, fallback: resolvedBaseline.textDetUnclipRatio)
            )

            Divider()
                .padding(.vertical, 12)

            groupTitle("Recognition")
            intParamRow(
                title: "Recognition batch size",
                caption: "Number of text lines per recognition run",
                range: 1...32,
                value: intOverrideBinding(\.textRecBatchSize, fallback: resolvedBaseline.textRecBatchSize)
            )
            paramRow(
                title: "Line confidence",
                caption: "Discard lines below this score",
                range: 0...1,
                value: floatOverrideBinding(\.textRecScoreThresh, fallback: resolvedBaseline.textRecScoreThresh)
            )

            HStack {
                Spacer()
                Button("Reset to defaults") {
                    params = .noOverrides
                }
                .font(.subheadline.weight(.medium))
            }
            .padding(.top, 4)
        }
    }

    private func floatOverrideBinding(
        _ keyPath: WritableKeyPath<OCRRuntimeParams, Float?>,
        fallback: Float
    ) -> Binding<Float> {
        Binding(
            get: { params[keyPath: keyPath] ?? fallback },
            set: { newValue in
                var next = params
                next[keyPath: keyPath] = newValue
                params = next
            }
        )
    }

    private func intOverrideBinding(
        _ keyPath: WritableKeyPath<OCRRuntimeParams, Int?>,
        fallback: Int
    ) -> Binding<Int> {
        Binding(
            get: { params[keyPath: keyPath] ?? fallback },
            set: { newValue in
                var next = params
                next[keyPath: keyPath] = newValue
                params = next
            }
        )
    }

    private func stringOverrideBinding(
        _ keyPath: WritableKeyPath<OCRRuntimeParams, String?>,
        fallback: String
    ) -> Binding<String> {
        Binding(
            get: { params[keyPath: keyPath] ?? fallback },
            set: { newValue in
                var next = params
                next[keyPath: keyPath] = newValue
                params = next
            }
        )
    }

    private func groupTitle(_ text: String) -> some View {
        Text(text)
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.primary)
            .padding(.bottom, 8)
    }

    private func paramRow(
        title: String,
        caption: String,
        range: ClosedRange<Float>,
        value: Binding<Float>
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.subheadline)
                    Text(caption)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text(String(format: "%.3f", value.wrappedValue))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color(.tertiarySystemFill), in: Capsule())
            }
            Slider(
                value: Binding(
                    get: { Double(value.wrappedValue) },
                    set: { value.wrappedValue = Float($0) }
                ),
                in: Double(range.lowerBound)...Double(range.upperBound)
            )
            .tint(Color.accentColor)
        }
        .padding(.bottom, 14)
    }

    private func intParamRow(
        title: String,
        caption: String,
        range: ClosedRange<Int>,
        value: Binding<Int>
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.subheadline)
                    Text(caption)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text("\(value.wrappedValue)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color(.tertiarySystemFill), in: Capsule())
            }
            Slider(
                value: Binding(
                    get: { Double(value.wrappedValue) },
                    set: { value.wrappedValue = Int($0.rounded()) }
                ),
                in: Double(range.lowerBound)...Double(range.upperBound),
                step: 1
            )
            .tint(Color.accentColor)
        }
        .padding(.bottom, 14)
    }

    private func limitTypeRow(selection: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Limit type")
                    .font(.subheadline)
                Text("min: shortest side ≥ limit length; max: longest side ≤ limit length")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Picker("Limit type", selection: selection) {
                Text("min").tag("min")
                Text("max").tag("max")
            }
            .pickerStyle(.segmented)
        }
        .padding(.bottom, 14)
    }
}
