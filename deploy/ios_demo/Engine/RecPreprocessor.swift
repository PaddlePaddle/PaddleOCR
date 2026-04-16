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

// MARK: - Errors

enum RecPreprocessorError: LocalizedError {
    case invalidImage
    case configMissing(String)
    case pixelExtractionFailed

    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Invalid input image: width or height is zero"
        case .configMissing(let detail):
            return "Required recognition preprocessing config missing: \(detail)"
        case .pixelExtractionFailed:
            return "Failed to extract pixel data from image"
        }
    }
}

// MARK: - Preprocessing Result

/// The output of recognition preprocessing: a float32 tensor and metadata.
struct RecPreprocessResult {
    /// Flat CHW array of shape [1, imgC, imgH, targetW].
    let tensorData: [Float]
    /// Tensor dimensions: [1, imgC, imgH, targetW].
    let tensorShape: [Int]
    /// Original image dimensions before any resizing.
    let originalSize: (width: Int, height: Int)
    /// Actual content width before zero-padding (the resized image width).
    let resizedWidth: Int
}

// MARK: - RecPreprocessor

/// Recognition resize + normalize + pad: dynamic width, fixed height, scale to [-1, 1],
/// CHW layout, right-pad to target width.
///
/// Recognition resize, per-channel normalization, HWC→CHW, and width padding.
/// `DecodeImage.img_mode` is applied via `InferenceConfig.decodeImageChannelOrder`. Resize uses OpenCV `INTER_LINEAR`.
///
/// Algorithm:
/// 1. Compute target width from aspect ratio.
/// 2. Resize `(resized_w, imgH)` with OpenCV.
/// 3. Normalize HWC `/255`, `-0.5`, `/0.5`.
/// 4. HWC → CHW; plane 0 is the first channel in memory (B or R depending on `img_mode`).
/// 5. Right-pad with zeros to target width.
struct RecPreprocessor {
    /// Channel count from `RecResizeImg.image_shape` (3).
    private let imgC: Int
    /// Target height from config RecResizeImg.image_shape (e.g., 48).
    private let imgH: Int
    /// Default width from config RecResizeImg.image_shape (e.g., 320).
    private let imgW: Int
    /// Absolute maximum width cap to prevent excessive memory usage.
    private let maxImgW: Int = 3200
    /// Channel order from `DecodeImage.img_mode` in the model config file.
    private let channelOrder: InferenceImageChannelOrder

    /// Creates a RecPreprocessor by extracting image_shape from the RecResizeImg transform op.
    ///
    /// - Parameter config: A parsed InferenceConfig from the recognition model's config file.
    /// - Throws: `RecPreprocessorError.configMissing` if RecResizeImg with image_shape is absent.
    init(config: InferenceConfig) throws {
        var foundImageShape: [Int]?

        for op in config.preProcess.transformOps {
            switch op {
            case .decodeImage, .detResizeForTest, .normalizeImage, .toCHWImage, .unknown:
                break
            case .recResizeImg(let imageShape):
                foundImageShape = imageShape
            }
        }

        guard let imageShape = foundImageShape, imageShape.count >= 3 else {
            throw RecPreprocessorError.configMissing("RecResizeImg with image_shape")
        }

        self.imgC = imageShape[0]
        self.imgH = imageShape[1]
        self.imgW = imageShape[2]
        self.channelOrder = config.decodeImageChannelOrder
    }

    /// Runs recognition preprocessing on a `CGImage`.
    ///
    /// Steps: extract pixels → compute dimensions → resize → normalize → HWC-to-CHW → pad
    ///
    /// - Parameter image: The input image (typically a cropped text region).
    /// - Returns: A `RecPreprocessResult` with the float32 CHW tensor and metadata.
    func preprocess(_ image: CGImage) throws -> RecPreprocessResult {
        let originalW = image.width
        let originalH = image.height

        guard originalW > 0, originalH > 0 else {
            throw RecPreprocessorError.invalidImage
        }

        // Step 0: HWC (BGR or RGB per `DecodeImage.img_mode`).
        let pixelBytes = try extractHWCPixels(from: image, width: originalW, height: originalH, order: channelOrder)

        // Step 1: Target canvas width from aspect ratio and `image_shape`.
        let whRatio = Float(originalW) / Float(originalH)
        let maxWhRatio = max(Float(imgW) / Float(imgH), whRatio)
        var targetW = Int(Float(imgH) * maxWhRatio)
        if targetW > maxImgW {
            targetW = maxImgW
        }

        // Step 2: Content width after resize (before right-padding).
        let ratio = Float(originalW) / Float(originalH)
        let resizedW: Int
        if targetW > maxImgW {
            resizedW = maxImgW
        } else if Int(ceil(Float(imgH) * ratio)) > targetW {
            resizedW = targetW
        } else {
            resizedW = Int(ceil(Float(imgH) * ratio))
        }

        // Step 3: OpenCV `INTER_LINEAR` resize on 3-channel row-major data.
        let resizedPixels = resizeWithOpenCV(pixelBytes, srcW: originalW, srcH: originalH, dstW: resizedW, dstH: imgH)

        // Step 4: Normalize with recognition formula: pixel/255.0, subtract 0.5, divide by 0.5
        // Equivalent to: pixel / 127.5 - 1.0, mapping [0, 255] to [-1.0, 1.0]
        let normalizedHWC = normalizePixels(resizedPixels, width: resizedW, height: imgH)

        // Step 5: HWC [H,W,3] -> CHW [3,H,W]
        let chwData = hwcToCHW(normalizedHWC, width: resizedW, height: imgH)

        // Step 6: Right-pad with zeros to targetW
        let paddedData = padToTargetWidth(chwData, contentWidth: resizedW, targetWidth: targetW, height: imgH, channels: imgC)

        return RecPreprocessResult(
            tensorData: paddedData,
            tensorShape: [1, imgC, imgH, targetW],
            originalSize: (width: originalW, height: originalH),
            resizedWidth: resizedW
        )
    }

    // MARK: - Step 0: Pixel extraction (BGR / RGB)

    private func extractHWCPixels(
        from image: CGImage,
        width: Int,
        height: Int,
        order: InferenceImageChannelOrder
    ) throws -> [UInt8] {
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
            throw RecPreprocessorError.pixelExtractionFailed
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var out = [UInt8](repeating: 0, count: height * width * 3)
        for i in 0..<(height * width) {
            let r = rgbaData[i * 4]
            let g = rgbaData[i * 4 + 1]
            let b = rgbaData[i * 4 + 2]
            switch order {
            case .bgr:
                out[i * 3] = b
                out[i * 3 + 1] = g
                out[i * 3 + 2] = r
            case .rgb:
                out[i * 3] = r
                out[i * 3 + 1] = g
                out[i * 3 + 2] = b
            }
        }
        return out
    }

    // MARK: - Step 3: Resize (OpenCV)

    private func resizeWithOpenCV(
        _ pixels: [UInt8], srcW: Int, srcH: Int, dstW: Int, dstH: Int
    ) -> [UInt8] {
        if srcW == dstW && srcH == dstH {
            return pixels
        }
        let out = PDBOpenCVImageBridge.resizeRGBU8(
            Data(pixels), srcWidth: srcW, srcHeight: srcH, dstWidth: dstW, dstHeight: dstH
        )
        let expected = dstW * dstH * 3
        guard out.count == expected else {
            return [UInt8](repeating: 0, count: expected)
        }
        return [UInt8](out)
    }

    // MARK: - Step 4: NormalizeImage (Recognition Formula)

    /// Applies recognition normalization: pixel / 255.0, then (x - 0.5) / 0.5.
    /// Equivalent to pixel / 127.5 - 1.0, mapping [0, 255] -> [-1.0, 1.0].
    ///
    /// This is NOT the same as detection normalization (which uses ImageNet mean/std).
    /// The recognition normalization is fixed and not parameterized from model config.
    private func normalizePixels(_ pixels: [UInt8], width: Int, height: Int) -> [Float] {
        let count = width * height * 3
        var result = [Float](repeating: 0, count: count)

        for i in 0..<count {
            result[i] = Float(pixels[i]) / 127.5 - 1.0
        }

        return result
    }

    // MARK: - Step 5: HWC → CHW

    private func hwcToCHW(_ hwcData: [Float], width: Int, height: Int) -> [Float] {
        let channelSize = height * width
        var chwData = [Float](repeating: 0, count: 3 * channelSize)

        for y in 0..<height {
            for x in 0..<width {
                let hwcIdx = (y * width + x) * 3
                let pixelOffset = y * width + x
                chwData[0 * channelSize + pixelOffset] = hwcData[hwcIdx]
                chwData[1 * channelSize + pixelOffset] = hwcData[hwcIdx + 1]
                chwData[2 * channelSize + pixelOffset] = hwcData[hwcIdx + 2]
            }
        }

        return chwData
    }

    // MARK: - Step 6: Zero-Padding

    /// Right-pads a CHW tensor from contentWidth to targetWidth with zeros.
    ///
    /// The input CHW data has shape [channels, height, contentWidth].
    /// The output has shape [channels, height, targetWidth] with zeros filling
    /// columns contentWidth..<targetWidth.
    private func padToTargetWidth(
        _ chwData: [Float],
        contentWidth: Int,
        targetWidth: Int,
        height: Int,
        channels: Int
    ) -> [Float] {
        if contentWidth == targetWidth {
            return chwData
        }

        let targetChannelSize = height * targetWidth
        let sourceChannelSize = height * contentWidth
        var padded = [Float](repeating: 0, count: channels * targetChannelSize)

        for c in 0..<channels {
            for y in 0..<height {
                let srcOffset = c * sourceChannelSize + y * contentWidth
                let dstOffset = c * targetChannelSize + y * targetWidth
                // Copy contentWidth floats per row, leaving the rest as zeros
                for x in 0..<contentWidth {
                    padded[dstOffset + x] = chwData[srcOffset + x]
                }
            }
        }

        return padded
    }
}
