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

/// Displays an error state with icon, message, and retry button.
struct ErrorView: View {
    let error: AppError
    let onRetry: () -> Void

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 48))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(errorSeverityColor)

            Text(errorTitle)
                .font(.title2)
                .fontWeight(.semibold)

            Text(error.localizedDescription)
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            Button(action: onRetry) {
                Label("Retry", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.borderedProminent)
            .padding(.top, 8)
        }
    }

    private var errorTitle: String {
        switch error {
        case .modelLoadFailed: return "Model Loading Failed"
        case .inferenceFailed: return "OCR Failed"
        case .imageLoadFailed: return "Image Error"
        }
    }

    private var errorSeverityColor: Color {
        switch error {
        case .imageLoadFailed: return Color(.systemOrange)
        case .modelLoadFailed, .inferenceFailed: return Color(.systemRed)
        }
    }
}
