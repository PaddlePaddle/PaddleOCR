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

// MARK: - Types

/// Integer 2D point used internally by the Clipper offset algorithm.
/// Clipper operates on integer coordinates for robustness.
struct IntPoint: Equatable {
    let x: Int64
    let y: Int64
}

/// Floating-point 2D point for intermediate geometric calculations.
struct DoublePoint {
    var x: Double
    var y: Double
}

/// Join type for polygon offset. Determines how corners are handled.
enum JoinType {
    case jtSquare
    case jtRound
    case jtMiter
}

/// End type for polygon offset. Determines how path endpoints are handled.
enum EndType {
    case etClosedPolygon
    case etClosedLine
    case etOpenSquare
    case etOpenRound
    case etOpenButt
}

// MARK: - ClipperOffset

/// Pure Swift port of the Clipper library's polygon offset algorithm.
///
/// This implements the offset (inflation/deflation) operation from Angus Johnson's
/// Clipper library (version 6.x), which is the same algorithm used by the Python
/// pyclipper library. The primary use case is the `JT_ROUND + ET_CLOSEDPOLYGON`
/// combination required by DBPostProcess's `unclip()` method.
///
/// Algorithm: For each edge of the input polygon, compute the outward unit normal.
/// At each vertex, offset the point along the normal by `delta`. For round joins,
/// insert arc points to create a smooth rounded corner when the turn is convex.
class ClipperOffset {

    /// Controls how far a mitered join can extend before being squared off.
    /// Only affects `.jtMiter` joins.
    var miterLimit: Double = 2.0

    /// Controls the granularity of arc approximation for round joins.
    /// Smaller values produce smoother arcs with more points.
    var arcTolerance: Double = 0.25

    // Internal storage
    private var paths: [[IntPoint]] = []
    private var joinTypes: [JoinType] = []
    private var endTypes: [EndType] = []

    // Computed during execution
    private var normals: [DoublePoint] = []
    private var delta: Double = 0
    private var sinA: Double = 0
    private var cosA: Double = 0
    private var stepsPerRadian: Double = 0
    private var resultPaths: [[IntPoint]] = []
    private var currentPath: [IntPoint] = []

    private static let twoPi = Double.pi * 2.0
    private static let defaultArcTolerance = 0.25

    init(miterLimit: Double = 2.0, arcTolerance: Double = 0.25) {
        self.miterLimit = miterLimit
        self.arcTolerance = arcTolerance
    }

    // MARK: - Public API

    /// Add a path to be offset.
    ///
    /// - Parameters:
    ///   - path: Array of integer points defining the polygon or polyline.
    ///   - joinType: How to handle corners (`.jtRound` for DB postprocessing).
    ///   - endType: How to handle endpoints (`.etClosedPolygon` for DB postprocessing).
    func addPath(_ path: [IntPoint], joinType: JoinType, endType: EndType) {
        guard path.count >= 2 else { return }

        // Strip duplicate consecutive points and the closing point if it equals the first
        var cleaned: [IntPoint] = []
        for point in path {
            if cleaned.isEmpty || cleaned.last! != point {
                cleaned.append(point)
            }
        }
        if cleaned.count > 1 && cleaned.last! == cleaned.first! {
            cleaned.removeLast()
        }
        guard cleaned.count >= 2 else { return }

        paths.append(cleaned)
        joinTypes.append(joinType)
        endTypes.append(endType)
    }

    /// Compute the offset of all added paths by the given delta.
    ///
    /// - Parameter delta: The offset distance. Positive expands, negative shrinks.
    /// - Returns: Array of offset polygons, each as an array of `IntPoint`.
    func execute(delta: Double) -> [[IntPoint]] {
        resultPaths = []

        if delta == 0 {
            // No offset: return copies of the original paths
            return paths
        }

        self.delta = delta

        // Compute arc steps based on arc tolerance
        let absDelta = abs(delta)
        var arcTol = self.arcTolerance
        if arcTol <= 0 {
            arcTol = ClipperOffset.defaultArcTolerance
        } else if arcTol > absDelta * ClipperOffset.defaultArcTolerance {
            arcTol = absDelta * ClipperOffset.defaultArcTolerance
        }

        let steps = Double.pi / acos(1.0 - arcTol / absDelta)
        stepsPerRadian = steps / ClipperOffset.twoPi

        for i in 0..<paths.count {
            let path = paths[i]
            let endType = endTypes[i]
            let joinType = joinTypes[i]

            if endType == .etClosedPolygon || endType == .etClosedLine {
                offsetClosedPath(path, joinType: joinType, endType: endType)
            } else {
                offsetOpenPath(path, joinType: joinType, endType: endType)
            }
        }

        return resultPaths
    }

    /// Clear all stored paths for reuse.
    func clear() {
        paths.removeAll()
        joinTypes.removeAll()
        endTypes.removeAll()
    }

    // MARK: - Convenience API

    /// Convenience method that offsets a single closed polygon using JT_ROUND + ET_CLOSEDPOLYGON.
    ///
    /// This is the exact combination used by DBPostProcess's `unclip()` method.
    /// Accepts float coordinates (as `CGPoint`), converts to integer internally
    /// (matching pyclipper's default behavior of rounding to nearest integer),
    /// computes the offset, and returns the result as float coordinates.
    ///
    /// - Parameters:
    ///   - polygon: Array of points defining the closed polygon.
    ///   - distance: The expansion distance (positive to expand).
    /// - Returns: Array of offset polygons, each as an array of `CGPoint`.
    static func offsetPolygon(_ polygon: [CGPoint], distance: Double) -> [[CGPoint]] {
        let clipper = ClipperOffset()

        // Convert CGPoint to IntPoint (round to nearest integer, matching pyclipper)
        let intPath = polygon.map { IntPoint(x: Int64($0.x.rounded()), y: Int64($0.y.rounded())) }

        clipper.addPath(intPath, joinType: .jtRound, endType: .etClosedPolygon)
        let results = clipper.execute(delta: distance)

        // Convert back to CGPoint
        return results.map { path in
            path.map { CGPoint(x: Double($0.x), y: Double($0.y)) }
        }
    }

    // MARK: - Offset Computation (Closed Paths)

    private func offsetClosedPath(_ path: [IntPoint], joinType: JoinType, endType: EndType) {
        let count = path.count
        guard count >= 2 else { return }

        // Compute normals for each edge
        normals = computeNormals(for: path, closed: true)

        currentPath = []

        // Determine winding direction to decide if delta should be negated
        let area = polygonArea(path)
        // For positive delta (expand): if polygon is clockwise (negative area in screen coords),
        // we want the offset to go outward.
        // pyclipper convention: positive delta always expands regardless of winding.
        // We handle this by ensuring delta and winding agree.
        var effectiveDelta = self.delta
        if endType == .etClosedPolygon {
            if area < 0 && effectiveDelta > 0 {
                effectiveDelta = -effectiveDelta
            } else if area > 0 && effectiveDelta < 0 {
                effectiveDelta = -effectiveDelta
            }
        }
        let savedDelta = self.delta
        self.delta = effectiveDelta

        for j in 0..<count {
            offsetPoint(j, path: path, joinType: joinType, count: count)
        }

        if !currentPath.isEmpty {
            resultPaths.append(currentPath)
        }

        self.delta = savedDelta
    }

    private func offsetOpenPath(_ path: [IntPoint], joinType: JoinType, endType: EndType) {
        // Open path offset is not needed for DBPostProcess, but included for completeness
        let count = path.count
        guard count >= 2 else { return }

        normals = computeNormals(for: path, closed: false)
        currentPath = []

        // First point
        currentPath.append(IntPoint(
            x: Int64((Double(path[0].x) + delta * normals[0].x).rounded()),
            y: Int64((Double(path[0].y) + delta * normals[0].y).rounded())
        ))

        for j in 1..<(count - 1) {
            offsetPoint(j, path: path, joinType: joinType, count: count)
        }

        // Last point
        let last = count - 1
        currentPath.append(IntPoint(
            x: Int64((Double(path[last].x) + delta * normals[last - 1].x).rounded()),
            y: Int64((Double(path[last].y) + delta * normals[last - 1].y).rounded())
        ))

        if !currentPath.isEmpty {
            resultPaths.append(currentPath)
        }
    }

    // MARK: - Per-Vertex Offset

    private func offsetPoint(_ j: Int, path: [IntPoint], joinType: JoinType, count: Int) {
        let prevIdx = (j + count - 1) % count
        let nextIdx = (j + 1) % count

        // sinA = cross product of normals (determines convexity)
        sinA = normals[prevIdx].x * normals[j].y - normals[j].x * normals[prevIdx].y
        // cosA = dot product of normals
        cosA = normals[prevIdx].x * normals[j].x + normals[prevIdx].y * normals[j].y

        // If normals are nearly parallel (edges are collinear)
        if abs(sinA * delta) < 1.0 {
            if cosA > 0 {
                // Nearly identical normals - just add a single offset point
                currentPath.append(IntPoint(
                    x: Int64((Double(path[j].x) + delta * normals[j].x).rounded()),
                    y: Int64((Double(path[j].y) + delta * normals[j].y).rounded())
                ))
                return
            }
            // Else: 180-degree turn, handle below
        }

        // Determine if this vertex is convex or concave relative to the offset direction
        if sinA * delta < 0 {
            // Concave side: add two offset points (one for each adjacent edge normal)
            currentPath.append(IntPoint(
                x: Int64((Double(path[j].x) + delta * normals[prevIdx].x).rounded()),
                y: Int64((Double(path[j].y) + delta * normals[prevIdx].y).rounded())
            ))
            currentPath.append(IntPoint(
                x: Int64((Double(path[j].x) + delta * normals[j].x).rounded()),
                y: Int64((Double(path[j].y) + delta * normals[j].y).rounded())
            ))
        } else {
            // Convex side: apply the selected join type
            switch joinType {
            case .jtRound:
                doRound(j, path: path)
            case .jtSquare:
                doSquare(j, path: path)
            case .jtMiter:
                let r = 1.0 + cosA
                if r >= miterLimit {
                    doMiter(j, path: path, r: r)
                } else {
                    doSquare(j, path: path)
                }
            }
        }
    }

    // MARK: - Join Types

    private func doRound(_ j: Int, path: [IntPoint]) {
        let prevIdx = (j + path.count - 1) % path.count

        // Angle between the two edge normals
        let angle = atan2(sinA, cosA)
        var steps = max(Int((stepsPerRadian * abs(angle)).rounded()), 1)

        let px = Double(path[j].x)
        let py = Double(path[j].y)

        // Start from the previous edge's normal
        var startX = normals[prevIdx].x
        var startY = normals[prevIdx].y

        // Rotation per step
        let stepAngle = angle / Double(steps)
        let stepSin = sin(stepAngle)
        let stepCos = cos(stepAngle)

        for _ in 0...steps {
            currentPath.append(IntPoint(
                x: Int64((px + delta * startX).rounded()),
                y: Int64((py + delta * startY).rounded())
            ))
            // Rotate the normal by stepAngle
            let newX = startX * stepCos - startY * stepSin
            let newY = startX * stepSin + startY * stepCos
            startX = newX
            startY = newY
        }
    }

    private func doSquare(_ j: Int, path: [IntPoint]) {
        let prevIdx = (j + path.count - 1) % path.count
        let px = Double(path[j].x)
        let py = Double(path[j].y)

        currentPath.append(IntPoint(
            x: Int64((px + delta * normals[prevIdx].x).rounded()),
            y: Int64((py + delta * normals[prevIdx].y).rounded())
        ))
        currentPath.append(IntPoint(
            x: Int64((px + delta * normals[j].x).rounded()),
            y: Int64((py + delta * normals[j].y).rounded())
        ))
    }

    private func doMiter(_ j: Int, path: [IntPoint], r: Double) {
        let prevIdx = (j + path.count - 1) % path.count
        let q = delta / r
        let px = Double(path[j].x)
        let py = Double(path[j].y)

        currentPath.append(IntPoint(
            x: Int64((px + (normals[prevIdx].x + normals[j].x) * q).rounded()),
            y: Int64((py + (normals[prevIdx].y + normals[j].y) * q).rounded())
        ))
    }

    // MARK: - Geometry Helpers

    /// Compute outward unit normals for each edge of the path.
    /// For a closed path, edge i goes from path[i] to path[(i+1) % count].
    /// For an open path, edge i goes from path[i] to path[i+1].
    private func computeNormals(for path: [IntPoint], closed: Bool) -> [DoublePoint] {
        let count = path.count
        var result = [DoublePoint](repeating: DoublePoint(x: 0, y: 0), count: count)

        let edgeCount = closed ? count : count - 1
        for i in 0..<edgeCount {
            let next = (i + 1) % count
            let dx = Double(path[next].x - path[i].x)
            let dy = Double(path[next].y - path[i].y)
            let len = sqrt(dx * dx + dy * dy)
            if len > 0 {
                // Outward normal: perpendicular to edge direction, pointing left
                result[i] = DoublePoint(x: dy / len, y: -dx / len)
            } else {
                result[i] = DoublePoint(x: 0, y: 0)
            }
        }

        // For open paths, the last normal is the same as the second-to-last
        if !closed && count >= 2 {
            result[count - 1] = result[count - 2]
        }

        return result
    }

    /// Compute the signed area of an integer polygon using the Shoelace formula.
    /// Positive area = counter-clockwise, negative = clockwise (in standard math coords).
    private func polygonArea(_ path: [IntPoint]) -> Double {
        let count = path.count
        guard count >= 3 else { return 0 }

        var area: Double = 0
        for i in 0..<count {
            let j = (i + 1) % count
            area += Double(path[i].x) * Double(path[j].y)
            area -= Double(path[j].x) * Double(path[i].y)
        }
        return area / 2.0
    }
}
