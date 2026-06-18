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

// MARK: - BGR ↔ CGImage

/// Builds a ``CGImage`` from row-major BGR uint8 (8-bit three-channel).
enum BGRImageCodec {
    static func makeCGImageFromBGR(pixels: [UInt8], width: Int, height: Int) throws -> CGImage {
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel

        let pixelCount = width * height
        guard pixels.count == pixelCount * 3 else {
            throw QuadTextCropError.imageCreationFailed
        }

        var rgbaData = [UInt8](repeating: 255, count: pixelCount * bytesPerPixel)
        for i in 0..<pixelCount {
            let b = pixels[i * 3]
            let g = pixels[i * 3 + 1]
            let r = pixels[i * 3 + 2]
            rgbaData[i * 4] = r
            rgbaData[i * 4 + 1] = g
            rgbaData[i * 4 + 2] = b
        }

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                  data: &rgbaData,
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bytesPerRow: bytesPerRow,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
              ),
              let cgImage = context.makeImage() else {
            throw QuadTextCropError.imageCreationFailed
        }

        return cgImage
    }
}

// MARK: - Errors

enum QuadTextCropError: LocalizedError {
    case invalidPolygon(String)
    case imageCreationFailed
    case openCVFailed

    var errorDescription: String? {
        switch self {
        case .invalidPolygon(let detail):
            return "Invalid polygon for quad text crop: \(detail)"
        case .imageCreationFailed:
            return "Failed to create output CGImage"
        case .openCVFailed:
            return "OpenCV quad text-line crop failed"
        }
    }
}

// MARK: - QuadTextCrop

/// Minimum-area quadrilateral, corner ordering, perspective warp, then optional 90° CCW rotation when height/width ≥ 1.5.
struct QuadTextCrop {

    /// Crop a text region from the source image using the detector quadrilateral.
    static func crop(_ image: CGImage, polygon: [[Int32]]) throws -> CGImage {
        guard polygon.count == 4, polygon.allSatisfy({ $0.count == 2 }) else {
            throw QuadTextCropError.invalidPolygon(
                "Expected 4 points with 2 coordinates each, got \(polygon.count) points"
            )
        }

        let srcPixels = try extractBGRPixels(from: image)
        let srcW = image.width
        let srcH = image.height

        let polyValues: [NSValue] = polygon.map {
            NSValue(cgPoint: CGPoint(x: CGFloat($0[0]), y: CGFloat($0[1])))
        }
        guard let out = PDBOpenCVImageBridge.quadTextLineCropBGR(
            Data(srcPixels),
            srcWidth: srcW,
            srcHeight: srcH,
            quad: polyValues
        ) else {
            throw QuadTextCropError.openCVFailed
        }

        let bytes = [UInt8](out.rgbData)
        return try BGRImageCodec.makeCGImageFromBGR(pixels: bytes, width: Int(out.width), height: Int(out.height))
    }
}

// MARK: - Pixel Extraction & Image Creation

private extension QuadTextCrop {

    /// Row-major HWC **BGR**
    static func extractBGRPixels(from image: CGImage) throws -> [UInt8] {
        let width = image.width
        let height = image.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var rgbaData = [UInt8](repeating: 0, count: height * bytesPerRow)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                  data: &rgbaData,
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bytesPerRow: bytesPerRow,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
              ) else {
            throw QuadTextCropError.imageCreationFailed
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        let pixelCount = width * height
        var bgrData = [UInt8](repeating: 0, count: pixelCount * 3)
        for i in 0..<pixelCount {
            let r = rgbaData[i * 4]
            let g = rgbaData[i * 4 + 1]
            let b = rgbaData[i * 4 + 2]
            bgrData[i * 3] = b
            bgrData[i * 3 + 1] = g
            bgrData[i * 3 + 2] = r
        }

        return bgrData
    }
}
