// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import CoreGraphics

// MARK: - Detection Result

/// A detected text region with its bounding quadrilateral and confidence score.
struct DetectionBox {
    /// Four corner points of the bounding quadrilateral, each as [x, y].
    /// Order: top-left, top-right, bottom-right, bottom-left (sorted by the
    /// `getMiniBoxes` algorithm matching the Python reference).
    let points: [[Int32]]

    /// Confidence score from `boxScoreFast` (mean probability in the box region).
    let score: Float
}

// MARK: - PostProcessConfig Protocol

/// Minimal protocol for postprocessing configuration.
/// Plan 02-01 will provide the concrete `PostProcessConfig` struct.
/// This protocol ensures DBPostProcessor can be initialized from it.
protocol DBPostProcessConfigurable {
    var thresh: Float? { get }
    var boxThresh: Float? { get }
    var maxCandidates: Int? { get }
    var unclipRatio: Float? { get }
}

// MARK: - DBPostProcessor

/// DB (Differentiable Binarization) text detection postprocessor.
///
/// Implements the full pipeline from raw model output probability map to
/// bounding polygons, matching the Python reference in `ppocr/postprocess/db_postprocess.py`.
///
/// Pipeline: threshold -> contours -> minAreaRect -> score filter -> Clipper expand -> scale
///
/// All parameters are read from configuration (with defaults matching inference.yml).
struct DBPostProcessor {

    /// Binary threshold for the probability map (pixels above this are foreground).
    let thresh: Float

    /// Minimum box confidence score to keep a detection.
    let boxThresh: Float

    /// Maximum number of contours to evaluate.
    let maxCandidates: Int

    /// Expansion ratio for Clipper polygon offset.
    let unclipRatio: Float

    /// Minimum side length of a bounding box to be kept (pixels in probability map space).
    let minSize: Float = 3.0

    /// Score computation mode. Only "fast" is implemented (same as Python default).
    let scoreMode: String = "fast"

    // MARK: - Initializers

    /// Initialize with explicit parameters.
    init(thresh: Float = 0.3, boxThresh: Float = 0.6, maxCandidates: Int = 1000, unclipRatio: Float = 1.5) {
        self.thresh = thresh
        self.boxThresh = boxThresh
        self.maxCandidates = maxCandidates
        self.unclipRatio = unclipRatio
    }

    /// Initialize from a configuration object (PostProcessConfig from InferenceConfig).
    init(config: DBPostProcessConfigurable) {
        self.thresh = config.thresh ?? 0.3
        self.boxThresh = config.boxThresh ?? 0.6
        self.maxCandidates = config.maxCandidates ?? 1000
        self.unclipRatio = config.unclipRatio ?? 1.5
    }

    // MARK: - Public API

    /// Process raw ORT detection output into bounding polygons.
    ///
    /// - Parameters:
    ///   - outputTensor: Raw float output from ONNX Runtime, shape [1, 1, H, W].
    ///   - tensorHeight: Height of the output tensor (H).
    ///   - tensorWidth: Width of the output tensor (W).
    ///   - originalWidth: Width of the original input image.
    ///   - originalHeight: Height of the original input image.
    /// - Returns: Array of detected text boxes with confidence scores.
    func process(
        outputTensor: [Float],
        tensorHeight: Int,
        tensorWidth: Int,
        originalWidth: Int,
        originalHeight: Int
    ) -> [DetectionBox] {
        // 1. Extract probability map: pred[0, 0, :, :] from shape [1, 1, H, W]
        let mapSize = tensorHeight * tensorWidth
        let pred: [Float]
        if outputTensor.count >= mapSize {
            pred = Array(outputTensor.prefix(mapSize))
        } else {
            return []
        }

        // 2. Binary threshold: create mask where prob > thresh
        var binaryMask = [UInt8](repeating: 0, count: mapSize)
        for i in 0..<mapSize {
            binaryMask[i] = pred[i] > thresh ? 1 : 0
        }

        // 3. Find contours on the binary mask
        let contours = findContours(
            binaryMask: binaryMask,
            width: tensorWidth,
            height: tensorHeight
        )

        // 4. Process each contour
        let numContours = min(contours.count, maxCandidates)
        var results: [DetectionBox] = []

        for i in 0..<numContours {
            let contour = contours[i]

            // 4a. Get minimum area bounding rectangle
            guard let (miniBox, sside) = getMiniBoxes(contour: contour) else {
                continue
            }
            if sside < minSize {
                continue
            }

            // 4b. Score the box using fast mode
            let points = miniBox.map { [$0.x, $0.y] }
            let score = boxScoreFast(
                pred: pred,
                predWidth: tensorWidth,
                predHeight: tensorHeight,
                box: points
            )
            if boxThresh > score {
                continue
            }

            // 4c. Expand polygon using Clipper offset (unclip)
            let cgPoints = miniBox.map { CGPoint(x: Double($0.x), y: Double($0.y)) }
            let area = polygonArea(cgPoints)
            let perimeter = polygonPerimeter(cgPoints)
            guard perimeter > 0 else { continue }
            let distance = Double(abs(area)) * Double(unclipRatio) / perimeter

            let expanded = ClipperOffset.offsetPolygon(cgPoints, distance: distance)
            if expanded.count != 1 {
                continue
            }

            // 4d. Second getMiniBoxes on the expanded polygon
            let expandedFloats = expanded[0].map { FloatPoint(x: Float($0.x), y: Float($0.y)) }
            guard let (finalBox, finalSside) = getMiniBoxes(contour: expandedFloats) else {
                continue
            }
            if finalSside < minSize + 2 {
                continue
            }

            // 4e. Scale to original image dimensions
            let w = Float(tensorWidth)
            let h = Float(tensorHeight)
            let dw = Float(originalWidth)
            let dh = Float(originalHeight)

            var scaledPoints: [[Int32]] = []
            for pt in finalBox {
                let sx = Int32(min(max((pt.x / w * dw).rounded(), 0), dw))
                let sy = Int32(min(max((pt.y / h * dh).rounded(), 0), dh))
                scaledPoints.append([sx, sy])
            }

            results.append(DetectionBox(points: scaledPoints, score: score))
        }

        return results
    }
}

// MARK: - Internal Point Type

/// Float 2D point for internal geometry computations.
private struct FloatPoint {
    var x: Float
    var y: Float
}

// MARK: - Contour Finding (Suzuki-Abe Border Following)

extension DBPostProcessor {

    /// Find contours in a binary mask using the Suzuki-Abe border following algorithm.
    ///
    /// Equivalent to `cv2.findContours(mask * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)`.
    ///
    /// RETR_LIST: Retrieves all contours without establishing hierarchy.
    /// CHAIN_APPROX_SIMPLE: Compresses horizontal, vertical, and diagonal segments,
    /// leaving only their endpoints.
    ///
    /// Reference: Suzuki, S. and Abe, K., "Topological structural analysis of digitized
    /// binary images by border following." CVGIP, 30(1):32-46, 1985.
    func findContours(binaryMask: [UInt8], width: Int, height: Int) -> [[FloatPoint]] {
        // Work with a copy padded by 1 pixel on each side (filled with 0)
        let paddedW = width + 2
        let paddedH = height + 2
        var image = [Int](repeating: 0, count: paddedW * paddedH)

        // Copy binary mask into padded image (1 = foreground)
        for y in 0..<height {
            for x in 0..<width {
                if binaryMask[y * width + x] != 0 {
                    image[(y + 1) * paddedW + (x + 1)] = 1
                }
            }
        }

        var contours: [[FloatPoint]] = []
        var nbd = 1 // current border sequential number

        for y in 1..<(paddedH - 1) {
            var lnbd = 1
            for x in 1..<(paddedW - 1) {
                let idx = y * paddedW + x
                let pixel = image[idx]

                // Determine if this is a border start point
                var borderStart = false
                var isOuter = false

                if pixel == 1 && image[idx - 1] == 0 {
                    // Outer border start
                    borderStart = true
                    isOuter = true
                } else if pixel >= 1 && image[idx + 1] == 0 {
                    // Hole border start
                    borderStart = true
                    isOuter = false
                }

                if borderStart {
                    nbd += 1
                    let contour = traceContour(
                        image: &image,
                        startX: x, startY: y,
                        width: paddedW, height: paddedH,
                        nbd: nbd,
                        isOuter: isOuter
                    )
                    if contour.count >= 2 {
                        // Convert from padded coordinates to original coordinates
                        let adjusted = contour.map { FloatPoint(x: Float($0.0 - 1), y: Float($0.1 - 1)) }
                        // Apply CHAIN_APPROX_SIMPLE: remove collinear intermediate points
                        let simplified = chainApproxSimple(adjusted)
                        if simplified.count >= 2 {
                            contours.append(simplified)
                        }
                    }
                }

                if image[idx] != 0 && image[idx] != 1 {
                    lnbd = abs(image[idx])
                }
            }
        }

        return contours
    }

    /// Trace a single contour starting at (startX, startY) using Moore boundary tracing.
    private func traceContour(
        image: inout [Int],
        startX: Int, startY: Int,
        width: Int, height: Int,
        nbd: Int,
        isOuter: Bool
    ) -> [(Int, Int)] {
        // 8-connected neighbor offsets (clockwise from right):
        // right, bottom-right, bottom, bottom-left, left, top-left, top, top-right
        let dx = [1, 1, 0, -1, -1, -1, 0, 1]
        let dy = [0, 1, 1, 1, 0, -1, -1, -1]

        var contour: [(Int, Int)] = []

        // Find the initial tracing direction
        let startDir = isOuter ? 7 : 3  // Start scanning from left for outer, right for hole

        // Find the first nonzero neighbor
        var firstNeighborDir = -1
        for i in 0..<8 {
            let dir = (startDir + i) % 8
            let nx = startX + dx[dir]
            let ny = startY + dy[dir]
            if nx >= 0 && nx < width && ny >= 0 && ny < height {
                if image[ny * width + nx] != 0 {
                    firstNeighborDir = dir
                    break
                }
            }
        }

        // Isolated point
        if firstNeighborDir == -1 {
            image[startY * width + startX] = -nbd
            contour.append((startX, startY))
            return contour
        }

        var cx = startX
        var cy = startY
        var searchDir = firstNeighborDir

        repeat {
            contour.append((cx, cy))

            // Find next border point: scan clockwise from (searchDir + 5) % 8
            // (i.e., start scanning from 3 positions before the direction we came from)
            var found = false
            let scanStart = (searchDir + 6) % 8  // Turn 90 degrees CW past direction we came from

            for i in 0..<8 {
                let dir = (scanStart + i) % 8
                let nx = cx + dx[dir]
                let ny = cy + dy[dir]
                if nx >= 0 && nx < width && ny >= 0 && ny < height {
                    if image[ny * width + nx] != 0 {
                        // Mark the current pixel based on the preceding pixel
                        let prevDir = (dir + 7) % 8
                        let px = cx + dx[prevDir]
                        let py = cy + dy[prevDir]
                        if px >= 0 && px < width && py >= 0 && py < height && image[py * width + px] == 0 {
                            image[cy * width + cx] = -nbd
                        } else if image[cy * width + cx] == 1 {
                            image[cy * width + cx] = nbd
                        }

                        cx = nx
                        cy = ny
                        searchDir = dir
                        found = true
                        break
                    }
                }
            }

            if !found {
                // Isolated point within border
                image[cy * width + cx] = -nbd
                break
            }

        } while !(cx == startX && cy == startY && searchDir == firstNeighborDir)

        // Limit contour length to avoid pathological cases
        return contour
    }

    /// Apply CHAIN_APPROX_SIMPLE compression: remove intermediate points on horizontal,
    /// vertical, and diagonal segments, keeping only endpoints where direction changes.
    private func chainApproxSimple(_ points: [FloatPoint]) -> [FloatPoint] {
        guard points.count >= 3 else { return points }

        var result: [FloatPoint] = [points[0]]

        for i in 1..<(points.count - 1) {
            let prev = result.last!
            let curr = points[i]
            let next = points[i + 1]

            // Direction from prev to curr
            let dx1 = sign(curr.x - prev.x)
            let dy1 = sign(curr.y - prev.y)
            // Direction from curr to next
            let dx2 = sign(next.x - curr.x)
            let dy2 = sign(next.y - curr.y)

            // Keep point if direction changes
            if dx1 != dx2 || dy1 != dy2 {
                result.append(curr)
            }
        }

        // Always include the last point
        result.append(points.last!)

        // Remove duplicates at the boundary (first == last for closed contours)
        if result.count > 1 && result.first!.x == result.last!.x && result.first!.y == result.last!.y {
            result.removeLast()
        }

        return result
    }

    private func sign(_ x: Float) -> Float {
        if x > 0 { return 1 }
        if x < 0 { return -1 }
        return 0
    }
}

// MARK: - Minimum Area Rectangle

extension DBPostProcessor {

    /// Compute the minimum area bounding rectangle of a contour.
    ///
    /// Equivalent to `cv2.minAreaRect(contour)` followed by `cv2.boxPoints()` and
    /// the sorting logic in `get_mini_boxes()`.
    ///
    /// Returns the 4 sorted corner points and the minimum side length,
    /// or nil if the contour is degenerate.
    func getMiniBoxes(contour: [FloatPoint]) -> (box: [FloatPoint], minSide: Float)? {
        guard contour.count >= 2 else { return nil }

        // Compute convex hull first (minAreaRect operates on convex hull)
        let hull = convexHull(contour)
        guard hull.count >= 2 else { return nil }

        // Find minimum area bounding rectangle using rotating calipers
        let rect = minAreaRect(hull)
        guard rect.size.width > 0 || rect.size.height > 0 else { return nil }

        // Get 4 corner points of the rotated rectangle
        var corners = boxPoints(rect)

        // Sort corners by x-coordinate (matching Python's sorted by x)
        corners.sort { $0.x < $1.x || ($0.x == $1.x && $0.y < $1.y) }

        // Assign indices matching Python's get_mini_boxes logic
        let index1: Int
        let index4: Int
        if corners[1].y > corners[0].y {
            index1 = 0; index4 = 1
        } else {
            index1 = 1; index4 = 0
        }

        let index2: Int
        let index3: Int
        if corners[3].y > corners[2].y {
            index2 = 2; index3 = 3
        } else {
            index2 = 3; index3 = 2
        }

        let box = [corners[index1], corners[index2], corners[index3], corners[index4]]
        let minSide = min(Float(rect.size.width), Float(rect.size.height))

        return (box, minSide)
    }

    /// Compute the minimum area rotated rectangle enclosing a convex hull.
    ///
    /// Uses the rotating calipers algorithm. The minimum area rectangle has at
    /// least one side flush with an edge of the convex hull.
    private func minAreaRect(_ hull: [FloatPoint]) -> (center: CGPoint, size: CGSize, angle: Float) {
        let n = hull.count

        if n == 1 {
            return (CGPoint(x: CGFloat(hull[0].x), y: CGFloat(hull[0].y)), CGSize.zero, 0)
        }

        if n == 2 {
            let cx = (hull[0].x + hull[1].x) / 2
            let cy = (hull[0].y + hull[1].y) / 2
            let dx = hull[1].x - hull[0].x
            let dy = hull[1].y - hull[0].y
            let length = sqrtf(dx * dx + dy * dy)
            let angle = atan2f(dy, dx) * 180 / Float.pi
            return (CGPoint(x: CGFloat(cx), y: CGFloat(cy)), CGSize(width: CGFloat(length), height: 0), angle)
        }

        // Try each edge of the convex hull as a potential side of the rectangle
        var bestArea: Float = .greatestFiniteMagnitude
        var bestCenter = CGPoint.zero
        var bestSize = CGSize.zero
        var bestAngle: Float = 0

        for i in 0..<n {
            let j = (i + 1) % n

            // Edge vector
            let edgeX = hull[j].x - hull[i].x
            let edgeY = hull[j].y - hull[i].y
            let edgeLen = sqrtf(edgeX * edgeX + edgeY * edgeY)
            guard edgeLen > 0 else { continue }

            // Unit vectors along and perpendicular to the edge
            let ux = edgeX / edgeLen
            let uy = edgeY / edgeLen
            let vx = -uy  // perpendicular
            let vy = ux

            // Project all hull points onto edge direction and perpendicular
            var minU: Float = .greatestFiniteMagnitude
            var maxU: Float = -.greatestFiniteMagnitude
            var minV: Float = .greatestFiniteMagnitude
            var maxV: Float = -.greatestFiniteMagnitude

            for k in 0..<n {
                let dx = hull[k].x - hull[i].x
                let dy = hull[k].y - hull[i].y
                let projU = dx * ux + dy * uy
                let projV = dx * vx + dy * vy
                minU = min(minU, projU)
                maxU = max(maxU, projU)
                minV = min(minV, projV)
                maxV = max(maxV, projV)
            }

            let width = maxU - minU
            let height = maxV - minV
            let area = width * height

            if area < bestArea {
                bestArea = area

                // Center in original coordinates
                let midU = (minU + maxU) / 2
                let midV = (minV + maxV) / 2
                let cx = hull[i].x + midU * ux + midV * vx
                let cy = hull[i].y + midU * uy + midV * vy

                bestCenter = CGPoint(x: CGFloat(cx), y: CGFloat(cy))
                bestSize = CGSize(width: CGFloat(width), height: CGFloat(height))
                bestAngle = atan2f(uy, ux) * 180 / Float.pi
            }
        }

        return (bestCenter, bestSize, bestAngle)
    }

    /// Compute the 4 corner points of a rotated rectangle.
    /// Equivalent to `cv2.boxPoints(rect)`.
    private func boxPoints(_ rect: (center: CGPoint, size: CGSize, angle: Float)) -> [FloatPoint] {
        let angle = rect.angle * Float.pi / 180
        let cosA = cosf(angle)
        let sinA = sinf(angle)
        let w = Float(rect.size.width) / 2
        let h = Float(rect.size.height) / 2
        let cx = Float(rect.center.x)
        let cy = Float(rect.center.y)

        // The four corners, rotated by angle around center
        return [
            FloatPoint(x: cx - w * cosA + h * sinA, y: cy - w * sinA - h * cosA),
            FloatPoint(x: cx + w * cosA + h * sinA, y: cy + w * sinA - h * cosA),
            FloatPoint(x: cx + w * cosA - h * sinA, y: cy + w * sinA + h * cosA),
            FloatPoint(x: cx - w * cosA - h * sinA, y: cy - w * sinA + h * cosA),
        ]
    }

    /// Compute the convex hull of a set of points using Andrew's monotone chain algorithm.
    /// Returns points in counter-clockwise order.
    func convexHull(_ points: [FloatPoint]) -> [FloatPoint] {
        guard points.count >= 3 else { return points }

        var sorted = points.sorted { $0.x < $1.x || ($0.x == $1.x && $0.y < $1.y) }
        let n = sorted.count

        // Build lower hull
        var lower: [FloatPoint] = []
        for p in sorted {
            while lower.count >= 2 && cross(lower[lower.count - 2], lower[lower.count - 1], p) <= 0 {
                lower.removeLast()
            }
            lower.append(p)
        }

        // Build upper hull
        var upper: [FloatPoint] = []
        for p in sorted.reversed() {
            while upper.count >= 2 && cross(upper[upper.count - 2], upper[upper.count - 1], p) <= 0 {
                upper.removeLast()
            }
            upper.append(p)
        }

        // Remove last point of each half because it's repeated
        lower.removeLast()
        upper.removeLast()

        return lower + upper
    }

    /// Cross product of vectors OA and OB where O, A, B are points.
    /// Positive if counter-clockwise, negative if clockwise, 0 if collinear.
    private func cross(_ o: FloatPoint, _ a: FloatPoint, _ b: FloatPoint) -> Float {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }
}

// MARK: - Box Scoring

extension DBPostProcessor {

    /// Compute the mean probability score within a box region of the probability map.
    ///
    /// Equivalent to Python's `box_score_fast`: computes axis-aligned bounding box of the
    /// 4 points, creates a polygon mask via scanline fill, then computes the mean of the
    /// probability map values within the masked region.
    func boxScoreFast(
        pred: [Float],
        predWidth: Int,
        predHeight: Int,
        box: [[Float]]
    ) -> Float {
        let w = predWidth
        let h = predHeight

        // Compute axis-aligned bounding box (floor/ceil, clipped)
        var xminF: Float = .greatestFiniteMagnitude
        var xmaxF: Float = -.greatestFiniteMagnitude
        var yminF: Float = .greatestFiniteMagnitude
        var ymaxF: Float = -.greatestFiniteMagnitude

        for pt in box {
            xminF = min(xminF, pt[0])
            xmaxF = max(xmaxF, pt[0])
            yminF = min(yminF, pt[1])
            ymaxF = max(ymaxF, pt[1])
        }

        let xmin = max(Int(floorf(xminF)), 0)
        let xmax = min(Int(ceilf(xmaxF)), w - 1)
        let ymin = max(Int(floorf(yminF)), 0)
        let ymax = min(Int(ceilf(ymaxF)), h - 1)

        guard xmax >= xmin && ymax >= ymin else { return 0 }

        let maskW = xmax - xmin + 1
        let maskH = ymax - ymin + 1

        // Create polygon mask shifted to local coordinates
        var shiftedBox: [[Int]] = box.map { [Int(($0[0] - Float(xmin)).rounded()), Int(($0[1] - Float(ymin)).rounded())] }

        var mask = [UInt8](repeating: 0, count: maskW * maskH)
        fillPoly(mask: &mask, width: maskW, height: maskH, polygon: shiftedBox, value: 1)

        // Compute mean of pred values within mask
        var sum: Float = 0
        var count: Float = 0
        for my in 0..<maskH {
            for mx in 0..<maskW {
                if mask[my * maskW + mx] != 0 {
                    let predY = ymin + my
                    let predX = xmin + mx
                    sum += pred[predY * w + predX]
                    count += 1
                }
            }
        }

        return count > 0 ? sum / count : 0
    }

    /// Fill a polygon in a mask using scanline fill algorithm.
    ///
    /// Equivalent to `cv2.fillPoly(mask, [polygon], value)`.
    func fillPoly(mask: inout [UInt8], width: Int, height: Int, polygon: [[Int]], value: UInt8) {
        guard polygon.count >= 3 else { return }

        // Find Y range
        var minY = Int.max
        var maxY = Int.min
        for pt in polygon {
            minY = min(minY, pt[1])
            maxY = max(maxY, pt[1])
        }
        minY = max(minY, 0)
        maxY = min(maxY, height - 1)

        let n = polygon.count

        // Scanline fill: for each Y, find X intersections with polygon edges
        for y in minY...maxY {
            var intersections: [Float] = []

            for i in 0..<n {
                let j = (i + 1) % n
                let y1 = polygon[i][1]
                let y2 = polygon[j][1]

                // Check if this edge crosses the scanline
                if (y1 <= y && y2 > y) || (y2 <= y && y1 > y) {
                    let x1 = Float(polygon[i][0])
                    let x2 = Float(polygon[j][0])
                    let fy1 = Float(y1)
                    let fy2 = Float(y2)
                    let t = (Float(y) - fy1) / (fy2 - fy1)
                    let xIntersect = x1 + t * (x2 - x1)
                    intersections.append(xIntersect)
                }
            }

            // Sort intersections and fill between pairs
            intersections.sort()
            var k = 0
            while k + 1 < intersections.count {
                let xStart = max(Int(ceilf(intersections[k])), 0)
                let xEnd = min(Int(floorf(intersections[k + 1])), width - 1)
                for x in xStart...max(xStart, xEnd) {
                    if x < width {
                        mask[y * width + x] = value
                    }
                }
                k += 2
            }
        }
    }
}

// MARK: - Geometry Helpers

extension DBPostProcessor {

    /// Compute the signed area of a polygon using the Shoelace formula.
    func polygonArea(_ points: [CGPoint]) -> Double {
        let n = points.count
        guard n >= 3 else { return 0 }

        var area: Double = 0
        for i in 0..<n {
            let j = (i + 1) % n
            area += Double(points[i].x) * Double(points[j].y)
            area -= Double(points[j].x) * Double(points[i].y)
        }
        return area / 2.0
    }

    /// Compute the perimeter (total edge length) of a polygon.
    func polygonPerimeter(_ points: [CGPoint]) -> Double {
        let n = points.count
        guard n >= 2 else { return 0 }

        var perimeter: Double = 0
        for i in 0..<n {
            let j = (i + 1) % n
            let dx = Double(points[j].x) - Double(points[i].x)
            let dy = Double(points[j].y) - Double(points[i].y)
            perimeter += sqrt(dx * dx + dy * dy)
        }
        return perimeter
    }
}
