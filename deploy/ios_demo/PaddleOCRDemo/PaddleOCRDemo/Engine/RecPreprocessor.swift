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
/// All shape parameters are read from InferenceConfig at initialization time.
/// Image manipulation uses CoreGraphics -- no OpenCV.
///
/// Algorithm:
/// 1. Compute target width from aspect ratio (ceil to match reference `math.ceil`)
/// 2. Resize to (resized_w, imgH) using bilinear interpolation
/// 3. Normalize with recognition formula: pixel / 127.5 - 1.0 (maps [0,255] to [-1,1])
/// 4. HWC -> CHW transpose
/// 5. Right-pad with zeros to target width
struct RecPreprocessor {
    /// Number of channels (always 3 for RGB).
    private let imgC: Int
    /// Target height from config RecResizeImg.image_shape (e.g., 48).
    private let imgH: Int
    /// Default width from config RecResizeImg.image_shape (e.g., 320).
    private let imgW: Int
    /// Absolute maximum width cap to prevent excessive memory usage.
    private let maxImgW: Int = 3200

    /// Creates a RecPreprocessor by extracting image_shape from the RecResizeImg transform op.
    ///
    /// - Parameter config: A parsed InferenceConfig from the recognition model's config file.
    /// - Throws: `RecPreprocessorError.configMissing` if RecResizeImg with image_shape is absent.
    init(config: InferenceConfig) throws {
        var foundImageShape: [Int]?

        for op in config.preProcess.transformOps {
            switch op {
            case .recResizeImg(let imageShape):
                foundImageShape = imageShape
            default:
                break
            }
        }

        guard let imageShape = foundImageShape, imageShape.count >= 3 else {
            throw RecPreprocessorError.configMissing("RecResizeImg with image_shape")
        }

        self.imgC = imageShape[0]
        self.imgH = imageShape[1]
        self.imgW = imageShape[2]
    }

    /// Runs the OCRResizeNormImg pipeline on a CGImage.
    ///
    /// Pipeline: extract pixels -> compute dimensions -> resize -> normalize -> HWC-to-CHW -> pad
    ///
    /// - Parameter image: The input image (typically a cropped text region).
    /// - Returns: A `RecPreprocessResult` with the float32 CHW tensor and metadata.
    func preprocess(_ image: CGImage) throws -> RecPreprocessResult {
        let originalW = image.width
        let originalH = image.height

        guard originalW > 0, originalH > 0 else {
            throw RecPreprocessorError.invalidImage
        }

        // Step 0: Extract RGB pixel bytes from CGImage
        let pixelBytes = try extractRGBPixels(from: image, width: originalW, height: originalH)

        // Step 1: Compute target dimensions (same as reference `resize` + `resize_norm_img`)
        let whRatio = Float(originalW) / Float(originalH)
        let maxWhRatio = max(Float(imgW) / Float(imgH), whRatio)
        var targetW = Int(Float(imgH) * maxWhRatio)
        if targetW > maxImgW {
            targetW = maxImgW
        }

        // Step 2: Compute resized width (same as reference `resize_norm_img`)
        let ratio = Float(originalW) / Float(originalH)
        let resizedW: Int
        if targetW > maxImgW {
            resizedW = maxImgW
        } else if Int(ceil(Float(imgH) * ratio)) > targetW {
            resizedW = targetW
        } else {
            resizedW = Int(ceil(Float(imgH) * ratio))
        }

        // Step 3: Resize image to (resizedW, imgH) using bilinear interpolation
        let resizedPixels = resizeImage(pixelBytes, srcW: originalW, srcH: originalH, dstW: resizedW, dstH: imgH)

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

    // MARK: - Step 0: Pixel Extraction

    /// Extracts RGB pixel bytes from a CGImage, discarding the alpha channel.
    /// Output is a UInt8 array in row-major HWC order: [H * W * 3].
    private func extractRGBPixels(from image: CGImage, width: Int, height: Int) throws -> [UInt8] {
        let bytesPerPixel = 4 // RGBA
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

        // Convert RGBX (4 bytes per pixel) to RGB (3 bytes per pixel)
        let pixelCount = height * width
        var rgbData = [UInt8](repeating: 0, count: pixelCount * 3)
        for i in 0..<pixelCount {
            rgbData[i * 3]     = rgbaData[i * 4]     // R
            rgbData[i * 3 + 1] = rgbaData[i * 4 + 1] // G
            rgbData[i * 3 + 2] = rgbaData[i * 4 + 2] // B
        }

        return rgbData
    }

    // MARK: - Step 3: Resize

    /// Resizes an RGB pixel buffer using bilinear interpolation via CoreGraphics.
    /// This matches cv2.resize() with INTER_LINEAR (the default).
    private func resizeImage(
        _ pixels: [UInt8], srcW: Int, srcH: Int, dstW: Int, dstH: Int
    ) -> [UInt8] {
        if srcW == dstW && srcH == dstH {
            return pixels
        }

        // Convert RGB (3 bpp) to RGBA (4 bpp) for CGContext compatibility
        let srcPixelCount = srcW * srcH
        var rgbaData = [UInt8](repeating: 255, count: srcPixelCount * 4)
        for i in 0..<srcPixelCount {
            rgbaData[i * 4]     = pixels[i * 3]
            rgbaData[i * 4 + 1] = pixels[i * 3 + 1]
            rgbaData[i * 4 + 2] = pixels[i * 3 + 2]
            // Alpha stays 255
        }

        // Draw into a destination context at the target size
        let dstBytesPerRow = dstW * 4
        var dstRGBA = [UInt8](repeating: 0, count: dstH * dstBytesPerRow)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let srcContext = CGContext(
                  data: &rgbaData,
                  width: srcW,
                  height: srcH,
                  bitsPerComponent: 8,
                  bytesPerRow: srcW * 4,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
              ),
              let srcImage = srcContext.makeImage(),
              let dstContext = CGContext(
                  data: &dstRGBA,
                  width: dstW,
                  height: dstH,
                  bitsPerComponent: 8,
                  bytesPerRow: dstBytesPerRow,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
              ) else {
            // Fallback: return zero-filled buffer of correct size
            return [UInt8](repeating: 0, count: dstH * dstW * 3)
        }

        dstContext.interpolationQuality = .high
        dstContext.draw(srcImage, in: CGRect(x: 0, y: 0, width: dstW, height: dstH))

        // Convert back to RGB (3 bpp)
        let dstPixelCount = dstW * dstH
        var rgbResult = [UInt8](repeating: 0, count: dstPixelCount * 3)
        for i in 0..<dstPixelCount {
            rgbResult[i * 3]     = dstRGBA[i * 4]
            rgbResult[i * 3 + 1] = dstRGBA[i * 4 + 1]
            rgbResult[i * 3 + 2] = dstRGBA[i * 4 + 2]
        }

        return rgbResult
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

    // MARK: - Step 5: ToCHWImage

    /// Converts HWC layout [H][W][3] to CHW layout [3][H][W].
    /// Output is a flat Float array of length 3 * H * W.
    private func hwcToCHW(_ hwcData: [Float], width: Int, height: Int) -> [Float] {
        let channelSize = height * width
        var chwData = [Float](repeating: 0, count: 3 * channelSize)

        for y in 0..<height {
            for x in 0..<width {
                let hwcIdx = (y * width + x) * 3
                let pixelOffset = y * width + x
                chwData[0 * channelSize + pixelOffset] = hwcData[hwcIdx]     // R channel
                chwData[1 * channelSize + pixelOffset] = hwcData[hwcIdx + 1] // G channel
                chwData[2 * channelSize + pixelOffset] = hwcData[hwcIdx + 2] // B channel
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
