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

/// Displays the list of recognized text results with confidence scores and a "Copy All" button.
struct ResultsListView: View {
    let results: [OCRResult]
    let copiedFeedback: Bool
    let onCopy: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header row: "Results (N)" + Copy All button
            HStack {
                Text("Results (\(results.count))")
                    .font(.headline)
                Spacer()
                Button(action: onCopy) {
                    Label(
                        copiedFeedback ? "Copied!" : "Copy All",
                        systemImage: "doc.on.doc"
                    )
                }
                .buttonStyle(.bordered)
                .disabled(results.isEmpty)
            }

            if results.isEmpty {
                // Empty state: no text detected (informational, not error)
                VStack(spacing: 8) {
                    Image(systemName: "text.magnifyingglass")
                        .font(.system(size: 36))
                        .foregroundColor(Color(.secondaryLabel))
                    Text("No Text Detected")
                        .font(.headline)
                    Text("No text was found in this image. Try a different image with visible text.")
                        .font(.subheadline)
                        .foregroundColor(Color(.secondaryLabel))
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
            } else {
                // Results list
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(Array(results.enumerated()), id: \.offset) { index, result in
                        HStack(alignment: .top) {
                            Text("\(index + 1).")
                                .font(.subheadline)
                                .foregroundColor(Color(.secondaryLabel))
                                .frame(width: 28, alignment: .trailing)

                            Text(result.text)
                                .font(.body)
                                .lineLimit(3)

                            Spacer()

                            Text(String(format: "%.1f%%", result.confidence * 100))
                                .font(.subheadline)
                                .foregroundColor(Color(.secondaryLabel))
                        }
                    }
                }
            }
        }
    }
}
