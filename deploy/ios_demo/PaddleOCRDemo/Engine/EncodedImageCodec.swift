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

import CoreGraphics
import Foundation
import ImageIO

enum EncodedImageCodecError: LocalizedError {
    case emptyData
    case createImageSourceFailed
    case createImageFailed

    var errorDescription: String? {
        switch self {
        case .emptyData:
            return "Image data is empty"
        case .createImageSourceFailed:
            return "Could not create image source from data"
        case .createImageFailed:
            return "Could not decode image from data"
        }
    }
}

/// Decodes PNG/JPEG/WebP and other formats supported by Image I/O into a ``CGImage``.
enum EncodedImageCodec {
    static func cgImage(fromEncodedData data: Data) throws -> CGImage {
        guard !data.isEmpty else {
            throw EncodedImageCodecError.emptyData
        }
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            throw EncodedImageCodecError.createImageSourceFailed
        }
        let options: [CFString: Any] = [kCGImageSourceShouldCache: true]
        guard let cgImage = CGImageSourceCreateImageAtIndex(source, 0, options as CFDictionary) else {
            throw EncodedImageCodecError.createImageFailed
        }
        return cgImage
    }
}
