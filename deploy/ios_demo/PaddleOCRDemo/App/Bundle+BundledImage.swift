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

import Foundation

extension Bundle {
    /// Raster extensions tried when resolving a bundled image by stem, or after an explicit extension misses.
    static let bundledRasterImageExtensions: [String] = [
        "jpg", "jpeg", "png", "heic", "heif", "webp", "tiff", "tif", "bmp", "gif",
    ]

    /// Resolves a filesystem path for an image under `subdirectory`.
    ///
    /// `name` may be a resource stem (e.g. `photo`) or include a suffix (e.g. `photo.jpg`).
    /// Extensions are matched case-insensitively when provided; otherwise each entry in
    /// ``bundledRasterImageExtensions`` is tried in order.
    func path(forBundledImageNamed name: String, subdirectory: String) -> String? {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let ns = trimmed as NSString
        let extGiven = ns.pathExtension.lowercased()
        let stem = ns.deletingPathExtension
        let resourceStem = stem.isEmpty ? trimmed : stem

        if !extGiven.isEmpty {
            if let path = path(forResource: resourceStem, ofType: extGiven, inDirectory: subdirectory) {
                return path
            }
        }
        for ext in Self.bundledRasterImageExtensions where ext != extGiven {
            if let path = path(forResource: resourceStem, ofType: ext, inDirectory: subdirectory) {
                return path
            }
        }
        return nil
    }
}
