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

enum PerspectiveCropError: LocalizedError {
    case invalidPolygon(String)
    case zeroDimension
    case singularMatrix
    case imageCreationFailed

    var errorDescription: String? {
        switch self {
        case .invalidPolygon(let detail):
            return "Invalid polygon for perspective crop: \(detail)"
        case .zeroDimension:
            return "Computed crop dimension is zero"
        case .singularMatrix:
            return "Perspective matrix is singular (cannot be inverted)"
        case .imageCreationFailed:
            return "Failed to create output CGImage"
        }
    }
}

// MARK: - PerspectiveCrop

/// Crops a text region from the source image using a 4-point perspective transform,
/// then rotates 90 degrees CCW if the result is tall-narrow (height/width >= 1.5).
///
/// Mirrors the quad crop path used by the reference OCR inference stack (minimum-area
/// rectangle ordering, perspective warp with replicate borders, tall-narrow rotation).
///
/// The implementation is pure Swift using CoreGraphics -- no OpenCV dependency.
/// It performs:
/// 1. DLT (Direct Linear Transform) to compute the 3x3 homography matrix
/// 2. Backward-mapping warp with bilinear interpolation and BORDER_REPLICATE
/// 3. 90-degree CCW rotation for tall-narrow crops
struct PerspectiveCrop {

    /// Crop a text region from the source image using perspective transform,
    /// then rotate 90 degrees CCW if the result is tall-narrow (height/width >= 1.5).
    ///
    /// - Parameters:
    ///   - image: Source CGImage to crop from.
    ///   - polygon: 4 corner points as [[Int32]] in order: TL, TR, BR, BL.
    /// - Returns: Cropped (and potentially rotated) CGImage.
    static func crop(_ image: CGImage, polygon: [[Int32]]) throws -> CGImage {
        guard polygon.count == 4, polygon.allSatisfy({ $0.count == 2 }) else {
            throw PerspectiveCropError.invalidPolygon(
                "Expected 4 points with 2 coordinates each, got \(polygon.count) points"
            )
        }

        let srcPts = polygon.map { (Float($0[0]), Float($0[1])) }

        // Compute output dimensions from polygon edge lengths (max opposite edge lengths).
        let cropWidth = Int(max(
            distance(srcPts[0], srcPts[1]),
            distance(srcPts[2], srcPts[3])
        ))
        let cropHeight = Int(max(
            distance(srcPts[0], srcPts[3]),
            distance(srcPts[1], srcPts[2])
        ))

        guard cropWidth > 0, cropHeight > 0 else {
            throw PerspectiveCropError.zeroDimension
        }

        // Destination rectangle corners
        let dstPts: [(Float, Float)] = [
            (0, 0),
            (Float(cropWidth), 0),
            (Float(cropWidth), Float(cropHeight)),
            (0, Float(cropHeight)),
        ]

        // Compute perspective transform matrix (src -> dst), then invert for
        // backward mapping (dst -> src)
        let H = try computePerspectiveMatrix(src: srcPts, dst: dstPts)
        let Hinv = try invertMatrix3x3(H)

        // Extract source pixel data as RGB bytes
        let srcPixels = try extractRGBPixels(from: image)
        let srcW = image.width
        let srcH = image.height

        // Warp: for each output pixel, sample from source using bilinear interpolation
        var outputPixels = [UInt8](repeating: 0, count: cropWidth * cropHeight * 3)
        for v in 0..<cropHeight {
            for u in 0..<cropWidth {
                // Backward map: (u, v) in dst -> (x, y) in src
                let uf = Float(u) + 0.5 // pixel center
                let vf = Float(v) + 0.5
                let w = Hinv[6] * uf + Hinv[7] * vf + Hinv[8]
                let x = (Hinv[0] * uf + Hinv[1] * vf + Hinv[2]) / w
                let y = (Hinv[3] * uf + Hinv[4] * vf + Hinv[5]) / w

                // Bilinear interpolation with BORDER_REPLICATE (clamp)
                let (r, g, b) = bilinearSample(
                    srcPixels, width: srcW, height: srcH,
                    x: x - 0.5, y: y - 0.5
                )
                let idx = (v * cropWidth + u) * 3
                outputPixels[idx] = r
                outputPixels[idx + 1] = g
                outputPixels[idx + 2] = b
            }
        }

        // Create CGImage from output pixels
        var cropImage = try createCGImage(from: outputPixels, width: cropWidth, height: cropHeight)

        // Rotate 90 degrees CCW if tall-narrow (same rule as reference OpenCV `rot90`).
        if Float(cropHeight) / Float(cropWidth) >= 1.5 {
            cropImage = try rotateCCW90(cropImage)
        }

        return cropImage
    }
}

// MARK: - Geometry Helpers

private extension PerspectiveCrop {

    /// Euclidean distance between two 2D points.
    static func distance(_ a: (Float, Float), _ b: (Float, Float)) -> Float {
        let dx = b.0 - a.0
        let dy = b.1 - a.1
        return sqrtf(dx * dx + dy * dy)
    }
}

// MARK: - Perspective Matrix (DLT)

private extension PerspectiveCrop {

    /// Compute the 3x3 perspective transform matrix from 4 source points to 4 destination points
    /// using the Direct Linear Transform (DLT) algorithm.
    ///
    /// Matches `cv2.getPerspectiveTransform(src, dst)`.
    ///
    /// The homography H maps source points to destination points:
    ///   w * [u, v, 1]^T = H * [x, y, 1]^T
    ///
    /// This produces an 8x8 linear system (h[8] = 1.0 by normalization).
    /// Solved using Gaussian elimination with partial pivoting.
    ///
    /// - Parameters:
    ///   - src: 4 source points as (x, y) tuples.
    ///   - dst: 4 destination points as (x, y) tuples.
    /// - Returns: 9-element array representing the 3x3 matrix in row-major order.
    static func computePerspectiveMatrix(
        src: [(Float, Float)],
        dst: [(Float, Float)]
    ) throws -> [Float64] {
        // Build 8x8 system: A * h = b
        // For each point pair (x_i, y_i) -> (u_i, v_i):
        //   x_i*h0 + y_i*h1 + h2 - x_i*u_i*h6 - y_i*u_i*h7 = u_i
        //   x_i*h3 + y_i*h4 + h5 - x_i*v_i*h6 - y_i*v_i*h7 = v_i
        var A = [[Float64]](repeating: [Float64](repeating: 0, count: 8), count: 8)
        var b = [Float64](repeating: 0, count: 8)

        for i in 0..<4 {
            let x = Float64(src[i].0)
            let y = Float64(src[i].1)
            let u = Float64(dst[i].0)
            let v = Float64(dst[i].1)

            let row1 = i * 2
            A[row1][0] = x
            A[row1][1] = y
            A[row1][2] = 1
            A[row1][3] = 0
            A[row1][4] = 0
            A[row1][5] = 0
            A[row1][6] = -x * u
            A[row1][7] = -y * u
            b[row1] = u

            let row2 = i * 2 + 1
            A[row2][0] = 0
            A[row2][1] = 0
            A[row2][2] = 0
            A[row2][3] = x
            A[row2][4] = y
            A[row2][5] = 1
            A[row2][6] = -x * v
            A[row2][7] = -y * v
            b[row2] = v
        }

        // Gaussian elimination with partial pivoting
        let h = try solveLinearSystem(&A, &b, n: 8)

        // Return as 3x3 matrix: [h0, h1, h2, h3, h4, h5, h6, h7, 1.0]
        return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]
    }

    /// Solve an NxN linear system Ax = b using Gaussian elimination with partial pivoting.
    /// Modifies A and b in place, returns the solution vector.
    static func solveLinearSystem(
        _ A: inout [[Float64]],
        _ b: inout [Float64],
        n: Int
    ) throws -> [Float64] {
        // Forward elimination with partial pivoting
        for col in 0..<n {
            // Find pivot
            var maxVal: Float64 = 0
            var maxRow = col
            for row in col..<n {
                let val = abs(A[row][col])
                if val > maxVal {
                    maxVal = val
                    maxRow = row
                }
            }

            guard maxVal > 1e-12 else {
                throw PerspectiveCropError.singularMatrix
            }

            // Swap rows
            if maxRow != col {
                A.swapAt(col, maxRow)
                b.swapAt(col, maxRow)
            }

            // Eliminate below
            let pivot = A[col][col]
            for row in (col + 1)..<n {
                let factor = A[row][col] / pivot
                for k in col..<n {
                    A[row][k] -= factor * A[col][k]
                }
                b[row] -= factor * b[col]
            }
        }

        // Back substitution
        var x = [Float64](repeating: 0, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            var sum: Float64 = b[i]
            for j in (i + 1)..<n {
                sum -= A[i][j] * x[j]
            }
            x[i] = sum / A[i][i]
        }

        return x
    }
}

// MARK: - Matrix Inversion

private extension PerspectiveCrop {

    /// Analytical 3x3 matrix inversion using the cofactor/adjugate formula.
    ///
    /// The input is a 9-element array in row-major order:
    /// [m00, m01, m02, m10, m11, m12, m20, m21, m22]
    ///
    /// Uses Float64 for numerical precision.
    static func invertMatrix3x3(_ m: [Float64]) throws -> [Float] {
        let m00 = m[0], m01 = m[1], m02 = m[2]
        let m10 = m[3], m11 = m[4], m12 = m[5]
        let m20 = m[6], m21 = m[7], m22 = m[8]

        let det = m00 * (m11 * m22 - m12 * m21)
                - m01 * (m10 * m22 - m12 * m20)
                + m02 * (m10 * m21 - m11 * m20)

        guard abs(det) > 1e-12 else {
            throw PerspectiveCropError.singularMatrix
        }

        let invDet = 1.0 / det

        // Cofactor matrix transposed (adjugate), divided by determinant
        return [
            Float((m11 * m22 - m12 * m21) * invDet),
            Float((m02 * m21 - m01 * m22) * invDet),
            Float((m01 * m12 - m02 * m11) * invDet),
            Float((m12 * m20 - m10 * m22) * invDet),
            Float((m00 * m22 - m02 * m20) * invDet),
            Float((m02 * m10 - m00 * m12) * invDet),
            Float((m10 * m21 - m11 * m20) * invDet),
            Float((m01 * m20 - m00 * m21) * invDet),
            Float((m00 * m11 - m01 * m10) * invDet),
        ]
    }
}

// MARK: - Pixel Extraction & Image Creation

private extension PerspectiveCrop {

    /// Extract RGB pixel data from a CGImage as a flat [UInt8] array in row-major HWC order.
    ///
    /// Uses `CGColorSpace.sRGB` and `noneSkipLast` (RGBX 4-byte) format, then strips
    /// the padding byte to produce pure RGB (3 bytes per pixel).
    static func extractRGBPixels(from image: CGImage) throws -> [UInt8] {
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
            throw PerspectiveCropError.imageCreationFailed
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Convert RGBX -> RGB
        let pixelCount = width * height
        var rgbData = [UInt8](repeating: 0, count: pixelCount * 3)
        for i in 0..<pixelCount {
            rgbData[i * 3] = rgbaData[i * 4]
            rgbData[i * 3 + 1] = rgbaData[i * 4 + 1]
            rgbData[i * 3 + 2] = rgbaData[i * 4 + 2]
        }

        return rgbData
    }

    /// Create a CGImage from a raw RGB byte array.
    static func createCGImage(from pixels: [UInt8], width: Int, height: Int) throws -> CGImage {
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel

        // Convert RGB -> RGBX for CGContext compatibility
        let pixelCount = width * height
        var rgbaData = [UInt8](repeating: 255, count: pixelCount * bytesPerPixel)
        for i in 0..<pixelCount {
            rgbaData[i * 4] = pixels[i * 3]
            rgbaData[i * 4 + 1] = pixels[i * 3 + 1]
            rgbaData[i * 4 + 2] = pixels[i * 3 + 2]
            // Alpha byte stays 255
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
            throw PerspectiveCropError.imageCreationFailed
        }

        return cgImage
    }
}

// MARK: - Bilinear Sampling

private extension PerspectiveCrop {

    /// Bilinear interpolation sampling from an RGB pixel buffer with BORDER_REPLICATE (clamp).
    ///
    /// - Parameters:
    ///   - pixels: Source RGB pixel data in row-major HWC order.
    ///   - width: Source image width.
    ///   - height: Source image height.
    ///   - x: Fractional x coordinate in source space.
    ///   - y: Fractional y coordinate in source space.
    /// - Returns: Interpolated (R, G, B) pixel values.
    static func bilinearSample(
        _ pixels: [UInt8],
        width: Int, height: Int,
        x: Float, y: Float
    ) -> (UInt8, UInt8, UInt8) {
        // Clamp to valid range (BORDER_REPLICATE)
        let maxX = Float(width - 1)
        let maxY = Float(height - 1)
        let cx = min(max(x, 0), maxX)
        let cy = min(max(y, 0), maxY)

        let x0 = Int(cx)
        let y0 = Int(cy)
        let x1 = min(x0 + 1, width - 1)
        let y1 = min(y0 + 1, height - 1)

        let fx = cx - Float(x0)
        let fy = cy - Float(y0)
        let fx1 = 1.0 - fx
        let fy1 = 1.0 - fy

        // Four corner pixel indices (RGB, 3 bytes per pixel)
        let i00 = (y0 * width + x0) * 3
        let i10 = (y0 * width + x1) * 3
        let i01 = (y1 * width + x0) * 3
        let i11 = (y1 * width + x1) * 3

        // Weights for the four corners
        let w00 = fx1 * fy1
        let w10 = fx * fy1
        let w01 = fx1 * fy
        let w11 = fx * fy

        // Interpolate each channel
        let r = w00 * Float(pixels[i00]) + w10 * Float(pixels[i10])
              + w01 * Float(pixels[i01]) + w11 * Float(pixels[i11])
        let g = w00 * Float(pixels[i00 + 1]) + w10 * Float(pixels[i10 + 1])
              + w01 * Float(pixels[i01 + 1]) + w11 * Float(pixels[i11 + 1])
        let b = w00 * Float(pixels[i00 + 2]) + w10 * Float(pixels[i10 + 2])
              + w01 * Float(pixels[i01 + 2]) + w11 * Float(pixels[i11 + 2])

        return (
            UInt8(min(max(r + 0.5, 0), 255)),
            UInt8(min(max(g + 0.5, 0), 255)),
            UInt8(min(max(b + 0.5, 0), 255))
        )
    }
}

// MARK: - Rotation

private extension PerspectiveCrop {

    /// Rotate a CGImage 90 degrees counter-clockwise (one quarter-turn CCW).
    ///
    /// Creates a new CGContext with swapped dimensions, applies a rotation transform,
    /// and draws the original image.
    static func rotateCCW90(_ image: CGImage) throws -> CGImage {
        let originalWidth = image.width
        let originalHeight = image.height

        // After CCW 90-degree rotation: new width = old height, new height = old width
        let rotatedWidth = originalHeight
        let rotatedHeight = originalWidth

        let bytesPerPixel = 4
        let bytesPerRow = rotatedWidth * bytesPerPixel
        var pixelData = [UInt8](repeating: 0, count: rotatedHeight * bytesPerRow)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                  data: &pixelData,
                  width: rotatedWidth,
                  height: rotatedHeight,
                  bitsPerComponent: 8,
                  bytesPerRow: bytesPerRow,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
              ) else {
            throw PerspectiveCropError.imageCreationFailed
        }

        // Apply CCW 90-degree rotation:
        // Translate so the rotation pivot produces the correct output placement,
        // then rotate by -pi/2 (CCW).
        context.translateBy(x: 0, y: CGFloat(rotatedHeight))
        context.rotate(by: -.pi / 2)

        // Draw the original image into the rotated coordinate space
        context.draw(image, in: CGRect(x: 0, y: 0, width: originalWidth, height: originalHeight))

        guard let rotatedImage = context.makeImage() else {
            throw PerspectiveCropError.imageCreationFailed
        }

        return rotatedImage
    }
}
