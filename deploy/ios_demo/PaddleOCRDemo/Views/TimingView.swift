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

/// Compact per-stage timing chips (fits the “output” section below the preview).
struct TimingView: View {
    let result: OCRRunResult

    var body: some View {
        HStack(spacing: 10) {
            chip(title: "Detect", ms: result.detectionTime)
            chip(title: "Recognize", ms: result.recognitionTime)
            chip(title: "Total", ms: result.totalTime, emphasized: true)
        }
        .frame(maxWidth: .infinity)
    }

    private func chip(title: String, ms: TimeInterval, emphasized: Bool = false) -> some View {
        VStack(spacing: 6) {
            Text(title)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(String(format: "%.0f", ms * 1000))
                .font(.system(.body, design: .rounded).weight(emphasized ? .semibold : .regular))
                .monospacedDigit()
            Text("ms")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .padding(.horizontal, 8)
        .background {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(emphasized ? Color.accentColor.opacity(0.12) : Color(.tertiarySystemFill))
        }
        .overlay {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Color(.separator).opacity(0.25), lineWidth: 0.5)
        }
    }
}
