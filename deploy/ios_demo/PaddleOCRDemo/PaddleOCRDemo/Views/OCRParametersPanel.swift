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

/// Sliders for DB detection + recognition score threshold (content only; wrap with `DemoCard` in parent).
struct OCRParametersPanel: View {
    @Binding var params: OCRRuntimeParams
    let baseline: ResolvedOCRRuntimeParams

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .firstTextBaseline) {
                Text("Parameters")
                    .font(.headline)
                Spacer()
                Button("Reset") {
                    params = .noOverrides
                }
                .font(.subheadline.weight(.medium))
            }
            .padding(.bottom, 4)

            groupTitle("Detection")
            paramRow(
                title: "Map threshold",
                caption: "Binarization on the detector heatmap",
                range: 0.05...0.95,
                value: overrideBinding(\.textDetThresh, fallback: baseline.textDetThresh)
            )
            paramRow(
                title: "Box score threshold",
                caption: "Minimum score to keep a box",
                range: 0.1...0.95,
                value: overrideBinding(\.textDetBoxThresh, fallback: baseline.textDetBoxThresh)
            )
            paramRow(
                title: "Unclip ratio",
                caption: "Expand recovered polygons",
                range: 0.5...3.0,
                value: overrideBinding(\.textDetUnclipRatio, fallback: baseline.textDetUnclipRatio)
            )

            Divider()
                .padding(.vertical, 12)

            groupTitle("Recognition")
            paramRow(
                title: "Line confidence",
                caption: "Discard lines below this score",
                range: 0...1,
                value: overrideBinding(\.textRecScoreThresh, fallback: baseline.textRecScoreThresh)
            )
        }
    }

    private func overrideBinding(
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
}
