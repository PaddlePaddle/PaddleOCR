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

/// Recognized lines + copy action (intended to sit inside `DemoCard` from the parent).
struct ResultsListView: View {
    let results: [OCRResult]
    let copiedFeedback: Bool
    let onCopy: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .firstTextBaseline) {
                Text("Recognized text")
                    .font(.headline)
                Spacer()
                Button(action: onCopy) {
                    Label(
                        copiedFeedback ? "Copied" : "Copy all",
                        systemImage: copiedFeedback ? "checkmark.circle.fill" : "doc.on.doc"
                    )
                    .labelStyle(.titleAndIcon)
                }
                .font(.subheadline.weight(.medium))
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(results.isEmpty)
            }
            .padding(.bottom, 12)

            if results.isEmpty {
                emptyState
            } else {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(results.enumerated()), id: \.offset) { index, result in
                        resultRow(index: index, result: result)
                        if index < results.count - 1 {
                            Divider()
                                .padding(.vertical, 10)
                        }
                    }
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "text.magnifyingglass")
                .font(.system(size: 32))
                .foregroundStyle(.secondary)
                .symbolRenderingMode(.hierarchical)
            Text("No text to show")
                .font(.subheadline.weight(.semibold))
            Text("Try lowering thresholds or use an image with clearer text.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 20)
    }

    private func resultRow(index: Int, result: OCRResult) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text("\(index + 1)")
                .font(.caption.weight(.bold))
                .foregroundStyle(.secondary)
                .frame(width: 22, height: 22)
                .background(Color(.tertiarySystemFill), in: Circle())

            Text(result.text)
                .font(.body)
                .foregroundStyle(.primary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .lineLimit(5)

            Text(String(format: "%.0f%%", result.confidence * 100))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color(.tertiarySystemFill), in: Capsule())
        }
    }
}
