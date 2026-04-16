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

enum DetPreprocessorError: LocalizedError {
    case invalidImage
    case configMissing(String)
    case pixelExtractionFailed

    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Invalid input image: width or height is zero"
        case .configMissing(let detail):
            return "Required preprocessing config missing: \(detail)"
        case .pixelExtractionFailed:
            return "Failed to extract pixel data from image"
        }
    }
}

// MARK: - Preprocessing Result

/// The output of detection preprocessing: a float32 tensor and metadata needed for postprocessing.
struct PreprocessResult {
    /// Flat CHW array of shape [1, 3, resizedH, resizedW].
    let tensorData: [Float]
    /// Tensor dimensions: [1, 3, resizedH, resizedW].
    let tensorShape: [Int]
    /// Original image dimensions before any resizing.
    let originalSize: (width: Int, height: Int)
    /// Image dimensions after resize and padding to stride multiples.
    let resizedSize: (width: Int, height: Int)
    /// Vertical resize ratio: resizedH / originalH.
    let ratioH: Float
    /// Horizontal resize ratio: resizedW / originalW.
    let ratioW: Float
}

// MARK: - DetPreprocessor

/// Implements detection preprocessing: **DetResizeForTest** →
/// **NormalizeImage** (from the model config file) -> **ToCHWImage**.
///
/// Rasterization uses CoreGraphics; resize/pad use OpenCV (`INTER_LINEAR`, `copyMakeBorder`).
/// HWC channel order follows `DecodeImage.img_mode` (`InferenceConfig.decodeImageChannelOrder`).
struct DetPreprocessor {
    private enum ResizeRule {
        case limitSide(len: Int, limitType: String, maxSideLimit: Int)
        case longEdge(resizeLong: Int)
    }

    private let scale: Float
    private let mean: [Float]
    private let std: [Float]
    private let channelOrder: InferenceImageChannelOrder

    init(config: InferenceConfig) throws {
        var foundScale: Float?
        var foundMean: [Float]?
        var foundStd: [Float]?

        for op in config.preProcess.transformOps {
            switch op {
            case .decodeImage:
                break
            case .detResizeForTest(_):
                break
            case .normalizeImage(let s, let m, let st, _):
                foundScale = s
                foundMean = m
                foundStd = st
            case .toCHWImage, .recResizeImg, .unknown:
                break
            }
        }

        guard let scale = foundScale, let mean = foundMean, let std = foundStd else {
            throw DetPreprocessorError.configMissing("NormalizeImage with scale, mean, std")
        }
        guard mean.count == 3, std.count == 3 else {
            throw DetPreprocessorError.configMissing("NormalizeImage mean/std must have exactly 3 values")
        }

        self.scale = scale
        self.mean = mean
        self.std = std
        self.channelOrder = config.decodeImageChannelOrder
    }

    private static func resizeRule(from params: DetResizeParams) throws -> ResizeRule {
        .limitSide(
            len: params.limitSideLen,
            limitType: params.limitType,
            maxSideLimit: params.maxSideLimit
        )
    }

    /// Rasterizes ``CGImage`` to HWC uint8 (`BGR` or `RGB` per `DecodeImage.img_mode`).
    func preprocess(_ image: CGImage, detResize: DetResizeParams) throws -> PreprocessResult {
        let originalW = image.width
        let originalH = image.height

        guard originalW > 0, originalH > 0 else {
            throw DetPreprocessorError.invalidImage
        }

        let pixelBytes = try rasterizeImageIntoHWCBuffer(
            from: image, width: originalW, height: originalH, order: channelOrder
        )
        return try preprocessFromHWCPixelBuffer(
            pixelBytes: pixelBytes,
            originalWidth: originalW,
            originalHeight: originalH,
            detResize: detResize
        )
    }

    private func preprocessFromHWCPixelBuffer(
        pixelBytes initialPixels: [UInt8],
        originalWidth: Int,
        originalHeight: Int,
        detResize: DetResizeParams
    ) throws -> PreprocessResult {
        let resizeRule = try Self.resizeRule(from: detResize)
        var pixelBytes = initialPixels
        var currentW = originalWidth
        var currentH = originalHeight

        if currentH + currentW < 64 {
            let paddedH = max(32, currentH)
            let paddedW = max(32, currentW)
            pixelBytes = padImage(pixelBytes, fromW: currentW, fromH: currentH, toW: paddedW, toH: paddedH)
            currentW = paddedW
            currentH = paddedH
        }

        let (resizeW, resizeH, ratioW, ratioH): (Int, Int, Float, Float)
        switch resizeRule {
        case .limitSide(let len, let limitType, let maxSideLimit):
            (resizeW, resizeH, ratioW, ratioH) = computeResizeLimitSide(
                srcW: currentW,
                srcH: currentH,
                limitSideLen: len,
                limitType: limitType,
                maxSideLimit: maxSideLimit
            )
        case .longEdge(let resizeLong):
            (resizeW, resizeH, ratioW, ratioH) = computeResizeLongEdge(
                srcW: currentW,
                srcH: currentH,
                resizeLong: resizeLong
            )
        }

        let resizedPixels = resizeImageBuffer(
            pixelBytes, srcW: currentW, srcH: currentH, dstW: resizeW, dstH: resizeH
        )

        let normalizedHWC = normalizePixels(resizedPixels, width: resizeW, height: resizeH)
        let chwData = hwcToCHW(normalizedHWC, width: resizeW, height: resizeH)

        return PreprocessResult(
            tensorData: chwData,
            tensorShape: [1, 3, resizeH, resizeW],
            originalSize: (width: originalWidth, height: originalHeight),
            resizedSize: (width: resizeW, height: resizeH),
            ratioH: ratioH,
            ratioW: ratioW
        )
    }

    // MARK: - Rasterization

    /// Row-major HWC uint8: **BGR** or **RGB** per `DecodeImage.img_mode` in the model config file.
    private func rasterizeImageIntoHWCBuffer(
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
            throw DetPreprocessorError.pixelExtractionFailed
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

    private func padImage(_ pixels: [UInt8], fromW: Int, fromH: Int, toW: Int, toH: Int) -> [UInt8] {
        if fromW == toW && fromH == toH {
            return pixels
        }
        let out = PDBOpenCVImageBridge.padRGBU8(
            Data(pixels), width: fromW, height: fromH, padWidth: toW, padHeight: toH
        )
        let expected = toW * toH * 3
        guard out.count == expected else {
            return [UInt8](repeating: 0, count: expected)
        }
        return [UInt8](out)
    }

    // MARK: - Resize geometry

    /// Limit-side resize with multiple-of-32 dimensions, optional cap on the larger side.
    private func computeResizeLimitSide(
        srcW: Int,
        srcH: Int,
        limitSideLen: Int,
        limitType: String,
        maxSideLimit: Int
    ) -> (width: Int, height: Int, ratioW: Float, ratioH: Float) {
        let h = Float(srcH)
        let w = Float(srcW)
        let ratio: Float
        switch limitType {
        case "max":
            if max(srcH, srcW) > limitSideLen {
                if srcH > srcW {
                    ratio = Float(limitSideLen) / h
                } else {
                    ratio = Float(limitSideLen) / w
                }
            } else {
                ratio = 1.0
            }
        case "min":
            if min(srcH, srcW) < limitSideLen {
                if srcH < srcW {
                    ratio = Float(limitSideLen) / h
                } else {
                    ratio = Float(limitSideLen) / w
                }
            } else {
                ratio = 1.0
            }
        case "resize_long":
            ratio = Float(limitSideLen) / Float(max(srcH, srcW))
        default:
            ratio = 1.0
        }

        var resizeH = Int(Float(srcH) * ratio)
        var resizeW = Int(Float(srcW) * ratio)

        if max(resizeH, resizeW) > maxSideLimit {
            let shrink = Float(maxSideLimit) / Float(max(resizeH, resizeW))
            resizeH = Int(Float(resizeH) * shrink)
            resizeW = Int(Float(resizeW) * shrink)
        }

        resizeH = max(Int((Double(resizeH) / 32.0).rounded(.toNearestOrEven)) * 32, 32)
        resizeW = max(Int((Double(resizeW) / 32.0).rounded(.toNearestOrEven)) * 32, 32)

        if resizeH == srcH && resizeW == srcW {
            return (srcW, srcH, 1.0, 1.0)
        }

        let ratioH = Float(resizeH) / Float(srcH)
        let ratioW = Float(resizeW) / Float(srcW)
        return (resizeW, resizeH, ratioW, ratioH)
    }

    /// Long-edge resize with stride 128 on both sides.
    private func computeResizeLongEdge(
        srcW: Int,
        srcH: Int,
        resizeLong: Int
    ) -> (width: Int, height: Int, ratioW: Float, ratioH: Float) {
        let ratio: Float
        if srcH > srcW {
            ratio = Float(resizeLong) / Float(srcH)
        } else {
            ratio = Float(resizeLong) / Float(srcW)
        }

        var resizeH = Int(Float(srcH) * ratio)
        var resizeW = Int(Float(srcW) * ratio)

        let stride = 128
        resizeH = ((resizeH + stride - 1) / stride) * stride
        resizeW = ((resizeW + stride - 1) / stride) * stride

        let ratioH = Float(resizeH) / Float(srcH)
        let ratioW = Float(resizeW) / Float(srcW)
        return (resizeW, resizeH, ratioW, ratioH)
    }

    private func resizeImageBuffer(
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

    // MARK: - NormalizeImage

    private func normalizePixels(_ pixels: [UInt8], width: Int, height: Int) -> [Float] {
        let pixelCount = width * height
        var result = [Float](repeating: 0, count: pixelCount * 3)

        let scaleOverStd = [scale / std[0], scale / std[1], scale / std[2]]
        let meanOverStd = [mean[0] / std[0], mean[1] / std[1], mean[2] / std[2]]

        for i in 0..<pixelCount {
            let baseIdx = i * 3
            result[baseIdx] = Float(pixels[baseIdx]) * scaleOverStd[0] - meanOverStd[0]
            result[baseIdx + 1] = Float(pixels[baseIdx + 1]) * scaleOverStd[1] - meanOverStd[1]
            result[baseIdx + 2] = Float(pixels[baseIdx + 2]) * scaleOverStd[2] - meanOverStd[2]
        }

        return result
    }

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
}
