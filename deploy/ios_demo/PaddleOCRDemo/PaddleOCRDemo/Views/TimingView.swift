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

/// Displays per-stage timing breakdown as a horizontal 3-column card.
struct TimingView: View {
    let result: OCRPipelineResult

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Performance")
                .font(.headline)

            HStack {
                timingColumn("Detection", result.detectionTime)
                Spacer()
                timingColumn("Recognition", result.recognitionTime)
                Spacer()
                timingColumn("Total", result.totalTime)
            }
        }
        .padding(16)
        .background(Color(.secondarySystemBackground))
        .cornerRadius(8)
    }

    private func timingColumn(_ label: String, _ time: TimeInterval) -> some View {
        VStack(spacing: 4) {
            Text(String(format: "%.0f ms", time * 1000))
                .font(.system(.body, design: .monospaced))
                .fontWeight(.semibold)
            Text(label)
                .font(.caption)
                .foregroundColor(Color(.secondaryLabel))
        }
    }
}
