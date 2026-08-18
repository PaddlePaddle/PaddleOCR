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

/// Image selection UI: a prominent "Choose Photo" button backed by PhotosPicker,
/// plus a row of sample image thumbnails for quick demo access.
struct ImagePickerSection: View {
    @Binding var selectedItem: PhotosPickerItem?
    let sampleImageNames: [String]
    let onSampleSelected: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Primary CTA: system photo picker
            PhotosPicker(
                selection: $selectedItem,
                matching: .images,
                photoLibrary: .shared()
            ) {
                Label("Choose Photo", systemImage: "photo.on.rectangle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)

            // Sample images row
            if !sampleImageNames.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Or try a sample:")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    HStack(spacing: 8) {
                        ForEach(sampleImageNames, id: \.self) { name in
                            Button {
                                onSampleSelected(name)
                            } label: {
                                sampleThumbnail(named: name)
                            }
                            .buttonStyle(.plain)
                            .frame(minWidth: 44, minHeight: 44)
                            .contentShape(Rectangle())
                            .accessibilityLabel("Sample image: \(name)")
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func sampleThumbnail(named name: String) -> some View {
        if let path = Bundle.main.path(forBundledImageNamed: name, subdirectory: "SampleImages"),
           let uiImage = UIImage(contentsOfFile: path) {
            Image(uiImage: uiImage)
                .resizable()
                .aspectRatio(contentMode: .fill)
                .frame(width: 60, height: 60)
                .clipShape(RoundedRectangle(cornerRadius: 8))
        } else {
            // Fallback for when image can't be loaded
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(.tertiarySystemFill))
                .frame(width: 60, height: 60)
                .overlay(
                    Image(systemName: "photo")
                        .foregroundStyle(.secondary)
                )
        }
    }
}
