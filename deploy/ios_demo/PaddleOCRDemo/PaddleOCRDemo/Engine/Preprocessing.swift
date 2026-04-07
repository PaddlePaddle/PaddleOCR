import Accelerate
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

/// Implements the detection preprocessing pipeline from inference.yml:
/// DetResizeForTest -> NormalizeImage -> ToCHWImage.
///
/// All parameters are read from InferenceConfig at initialization time.
/// Image manipulation uses CoreGraphics and Accelerate -- no OpenCV.
struct DetPreprocessor {
    private let resizeLong: Int
    private let scale: Float
    private let mean: [Float]
    private let std: [Float]
    private let normalizeOrder: String

    /// Creates a DetPreprocessor by extracting transform parameters from the given config.
    ///
    /// - Parameter config: A parsed InferenceConfig from the detection model's inference.yml.
    /// - Throws: `DetPreprocessorError.configMissing` if required transform ops are absent.
    init(config: InferenceConfig) throws {
        var foundResizeLong: Int?
        var foundScale: Float?
        var foundMean: [Float]?
        var foundStd: [Float]?
        var foundOrder: String?

        for op in config.preProcess.transformOps {
            switch op {
            case .detResizeForTest(let rl):
                foundResizeLong = rl
            case .normalizeImage(let s, let m, let st, let o):
                foundScale = s
                foundMean = m
                foundStd = st
                foundOrder = o
            case .toCHWImage, .recResizeImg, .unknown:
                break
            }
        }

        guard let resizeLong = foundResizeLong else {
            throw DetPreprocessorError.configMissing("DetResizeForTest with resize_long")
        }
        guard let scale = foundScale, let mean = foundMean, let std = foundStd else {
            throw DetPreprocessorError.configMissing("NormalizeImage with scale, mean, std")
        }
        guard mean.count == 3, std.count == 3 else {
            throw DetPreprocessorError.configMissing("NormalizeImage mean/std must have exactly 3 values")
        }

        self.resizeLong = resizeLong
        self.scale = scale
        self.mean = mean
        self.std = std
        self.normalizeOrder = foundOrder ?? "hwc"
    }

    /// Runs the full detection preprocessing pipeline on a CGImage.
    ///
    /// Pipeline: image_padding (if tiny) -> DetResizeForTest -> NormalizeImage -> ToCHWImage
    ///
    /// - Parameter image: The input image as a CGImage.
    /// - Returns: A `PreprocessResult` with the float32 CHW tensor and resize metadata.
    func preprocess(_ image: CGImage) throws -> PreprocessResult {
        let originalW = image.width
        let originalH = image.height

        guard originalW > 0, originalH > 0 else {
            throw DetPreprocessorError.invalidImage
        }

        // Step 0: Extract RGB pixel bytes from CGImage
        var pixelBytes = try extractRGBPixels(from: image, width: originalW, height: originalH)
        var currentW = originalW
        var currentH = originalH

        // Step 1: Image padding for tiny images (h + w < 64)
        if currentH + currentW < 64 {
            let paddedH = max(32, currentH)
            let paddedW = max(32, currentW)
            pixelBytes = padImage(pixelBytes, fromW: currentW, fromH: currentH, toW: paddedW, toH: paddedH)
            currentW = paddedW
            currentH = paddedH
        }

        // Step 2: DetResizeForTest (resize_type=2, triggered by resize_long in config)
        let (resizeW, resizeH, ratioW, ratioH) = computeResizeDimensions(
            srcW: currentW, srcH: currentH, resizeLong: resizeLong
        )

        let resizedPixels = resizeImage(
            pixelBytes, srcW: currentW, srcH: currentH, dstW: resizeW, dstH: resizeH
        )

        // Step 3: NormalizeImage — (pixel * scale - mean) / std
        let normalizedHWC = normalizePixels(resizedPixels, width: resizeW, height: resizeH)

        // Step 4: ToCHWImage — HWC [H,W,3] -> CHW [3,H,W]
        let chwData = hwcToCHW(normalizedHWC, width: resizeW, height: resizeH)

        return PreprocessResult(
            tensorData: chwData,
            tensorShape: [1, 3, resizeH, resizeW],
            originalSize: (width: originalW, height: originalH),
            resizedSize: (width: resizeW, height: resizeH),
            ratioH: ratioH,
            ratioW: ratioW
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
            throw DetPreprocessorError.pixelExtractionFailed
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Convert RGBX (4 bytes per pixel) to RGB (3 bytes per pixel)
        var rgbData = [UInt8](repeating: 0, count: height * width * 3)
        for i in 0..<(height * width) {
            rgbData[i * 3]     = rgbaData[i * 4]     // R
            rgbData[i * 3 + 1] = rgbaData[i * 4 + 1] // G
            rgbData[i * 3 + 2] = rgbaData[i * 4 + 2] // B
        }

        return rgbData
    }

    // MARK: - Step 1: Image Padding

    /// Pads a small image to at least 32x32 with zero-fill, matching Python's image_padding.
    private func padImage(_ pixels: [UInt8], fromW: Int, fromH: Int, toW: Int, toH: Int) -> [UInt8] {
        var padded = [UInt8](repeating: 0, count: toH * toW * 3)
        for y in 0..<fromH {
            for x in 0..<fromW {
                let srcIdx = (y * fromW + x) * 3
                let dstIdx = (y * toW + x) * 3
                padded[dstIdx]     = pixels[srcIdx]
                padded[dstIdx + 1] = pixels[srcIdx + 1]
                padded[dstIdx + 2] = pixels[srcIdx + 2]
            }
        }
        return padded
    }

    // MARK: - Step 2: DetResizeForTest

    /// Computes the target resize dimensions matching Python's resize_image_type2.
    ///
    /// 1. Scale so the longest side equals `resizeLong`
    /// 2. Ceil both dimensions to the nearest multiple of 128
    private func computeResizeDimensions(
        srcW: Int, srcH: Int, resizeLong: Int
    ) -> (width: Int, height: Int, ratioW: Float, ratioH: Float) {
        let ratio: Float
        if srcH > srcW {
            ratio = Float(resizeLong) / Float(srcH)
        } else {
            ratio = Float(resizeLong) / Float(srcW)
        }

        var resizeH = Int(Float(srcH) * ratio)
        var resizeW = Int(Float(srcW) * ratio)

        let maxStride = 128
        resizeH = ((resizeH + maxStride - 1) / maxStride) * maxStride
        resizeW = ((resizeW + maxStride - 1) / maxStride) * maxStride

        let ratioH = Float(resizeH) / Float(srcH)
        let ratioW = Float(resizeW) / Float(srcW)

        return (resizeW, resizeH, ratioW, ratioH)
    }

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

        // Draw into a destination context at the target size — CGContext uses bilinear interpolation
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

    // MARK: - Step 3: NormalizeImage

    /// Applies NormalizeImage: `(pixel_float32 * scale - mean) / std` per channel.
    /// Input: UInt8 RGB pixels in HWC layout. Output: Float32 RGB in HWC layout.
    ///
    /// The normalization order from inference.yml is "hwc", meaning mean/std are applied
    /// in HWC layout (per-pixel, per-channel).
    private func normalizePixels(_ pixels: [UInt8], width: Int, height: Int) -> [Float] {
        let pixelCount = width * height
        var result = [Float](repeating: 0, count: pixelCount * 3)

        // Precompute per-channel normalization coefficients
        // Formula: normalized[c] = (pixel_uint8 * scale - mean[c]) / std[c]
        //        = pixel_uint8 * (scale / std[c]) - (mean[c] / std[c])
        let scaleOverStd = [scale / std[0], scale / std[1], scale / std[2]]
        let meanOverStd = [mean[0] / std[0], mean[1] / std[1], mean[2] / std[2]]

        for i in 0..<pixelCount {
            let baseIdx = i * 3
            result[baseIdx]     = Float(pixels[baseIdx])     * scaleOverStd[0] - meanOverStd[0]
            result[baseIdx + 1] = Float(pixels[baseIdx + 1]) * scaleOverStd[1] - meanOverStd[1]
            result[baseIdx + 2] = Float(pixels[baseIdx + 2]) * scaleOverStd[2] - meanOverStd[2]
        }

        return result
    }

    // MARK: - Step 4: ToCHWImage

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
}
