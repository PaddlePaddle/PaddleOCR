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

/// Batched recognition input: N × C × H × W with a common width after padding.
struct RecBatchPreprocessResult {
    let tensorData: [Float]
    let tensorShape: [Int]
}

// MARK: - RecPreprocessor

/// Recognition resize + normalize + pad: dynamic width, fixed height, scale to [-1, 1],
/// CHW layout, right-pad to target width.
///
/// Recognition resize, per-channel normalization, HWC→CHW, and width padding.
/// `DecodeImage.img_mode` is applied via `InferenceConfig.decodeImageChannelOrder`. Resize uses OpenCV `INTER_LINEAR`.
///
/// Set environment variable `PADDLEOCR_REC_ALIGN_WIDTH_TO_8` to `1`, `true`, or `yes` to round the
/// canvas width up to a multiple of 8 (off by default).
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

        let pixelBytes = try extractHWCPixels(from: image, width: originalW, height: originalH, order: channelOrder)
        let (paddedData, canvasW, resizedW) = try lineResizeNormalizeCHW(
            pixelBytes: pixelBytes,
            originalW: originalW,
            originalH: originalH,
            alignCanvasTo8: Self.isRecognitionCanvasWidthAlignedTo8Enabled()
        )

        return RecPreprocessResult(
            tensorData: paddedData,
            tensorShape: [1, imgC, imgH, canvasW],
            originalSize: (width: originalW, height: originalH),
            resizedWidth: resizedW
        )
    }

    /// Builds a batched tensor: each line is resized and padded to its canvas width, then all rows
    /// are padded on the right to the maximum canvas width in the batch (shape `[N, C, H, W]`).
    func preprocessBatch(_ images: [CGImage]) throws -> RecBatchPreprocessResult {
        guard !images.isEmpty else {
            throw RecPreprocessorError.invalidImage
        }

        let align8 = Self.isRecognitionCanvasWidthAlignedTo8Enabled()
        var rowCHW: [[Float]] = []
        var canvasWidths: [Int] = []

        for image in images {
            let originalW = image.width
            let originalH = image.height
            guard originalW > 0, originalH > 0 else {
                throw RecPreprocessorError.invalidImage
            }
            let pixelBytes = try extractHWCPixels(from: image, width: originalW, height: originalH, order: channelOrder)
            let (chw, canvasW, _) = try lineResizeNormalizeCHW(
                pixelBytes: pixelBytes,
                originalW: originalW,
                originalH: originalH,
                alignCanvasTo8: align8
            )
            rowCHW.append(chw)
            canvasWidths.append(canvasW)
        }

        guard let maxCanvasW = canvasWidths.max() else {
            throw RecPreprocessorError.invalidImage
        }

        let n = rowCHW.count
        let channelSize = imgH * maxCanvasW
        var flat = [Float](repeating: 0, count: n * imgC * channelSize)

        for (b, row) in rowCHW.enumerated() {
            let w = canvasWidths[b]
            let paddedRow: [Float]
            if w == maxCanvasW {
                paddedRow = row
            } else {
                let contentW = w
                paddedRow = padToTargetWidth(
                    row,
                    contentWidth: contentW,
                    targetWidth: maxCanvasW,
                    height: imgH,
                    channels: imgC
                )
            }
            let batchOffset = b * imgC * channelSize
            for i in 0..<paddedRow.count {
                flat[batchOffset + i] = paddedRow[i]
            }
        }

        return RecBatchPreprocessResult(
            tensorData: flat,
            tensorShape: [n, imgC, imgH, maxCanvasW]
        )
    }

    /// When `PADDLEOCR_REC_ALIGN_WIDTH_TO_8` is set to a truthy value, the canvas width is rounded up to a multiple of 8.
    private static func isRecognitionCanvasWidthAlignedTo8Enabled() -> Bool {
        let v = ProcessInfo.processInfo.environment["PADDLEOCR_REC_ALIGN_WIDTH_TO_8"]?.lowercased()
        return v == "1" || v == "true" || v == "yes"
    }

    /// One text line: HWC uint8 → resize → normalize → CHW, padded to the computed canvas width.
    private func lineResizeNormalizeCHW(
        pixelBytes: [UInt8],
        originalW: Int,
        originalH: Int,
        alignCanvasTo8: Bool
    ) throws -> (chw: [Float], canvasW: Int, resizedW: Int) {
        let w = originalW
        let h = originalH
        let whRatio = Float(w) / Float(h)
        let defaultRatio = Float(imgW) / Float(imgH)
        let maxWhRatio = max(defaultRatio, whRatio)
        var canvasW = Int(Float(imgH) * maxWhRatio)
        let resizedW: Int

        if canvasW > maxImgW {
            canvasW = maxImgW
            resizedW = maxImgW
        } else {
            let ratio = Float(w) / Float(h)
            if Int(ceil(Float(imgH) * ratio)) > canvasW {
                resizedW = canvasW
            } else {
                resizedW = Int(ceil(Float(imgH) * ratio))
            }
        }

        if alignCanvasTo8 {
            canvasW = min(max(roundUpToMultipleOf(canvasW, 8), resizedW), maxImgW)
        }

        let resizedPixels = resizeWithOpenCV(pixelBytes, srcW: w, srcH: h, dstW: resizedW, dstH: imgH)
        let normalizedHWC = normalizePixels(resizedPixels, width: resizedW, height: imgH)
        let chwData = hwcToCHW(normalizedHWC, width: resizedW, height: imgH)
        let paddedData = padToTargetWidth(
            chwData,
            contentWidth: resizedW,
            targetWidth: canvasW,
            height: imgH,
            channels: imgC
        )
        return (paddedData, canvasW, resizedW)
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

    // MARK: - Resize (OpenCV)

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

    // MARK: - Normalize (recognition)

    /// Maps pixel values from [0, 255] to [-1, 1] via `pixel / 127.5 - 1.0`.
    private func normalizePixels(_ pixels: [UInt8], width: Int, height: Int) -> [Float] {
        let count = width * height * 3
        var result = [Float](repeating: 0, count: count)

        for i in 0..<count {
            result[i] = Float(pixels[i]) / 127.5 - 1.0
        }

        return result
    }

    // MARK: - HWC → CHW

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

    // MARK: - Zero-Padding

    /// Right-pads a CHW tensor from contentWidth to targetWidth with zeros.
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
                for x in 0..<contentWidth {
                    padded[dstOffset + x] = chwData[srcOffset + x]
                }
            }
        }

        return padded
    }
}

private func roundUpToMultipleOf(_ value: Int, _ multiple: Int) -> Int {
    guard multiple > 0 else { return value }
    return (value + multiple - 1) / multiple * multiple
}
