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

import Foundation

// MARK: - BoxSorter

/// Sorts detection boxes in natural reading order: top-to-bottom, then
/// left-to-right within the same text line.
///
/// Two-pass: sort by top-left (y then x), then swap same-line neighbors when
/// the y-gap is below `yThreshold` but x is out of order.
struct BoxSorter {

    /// Y-coordinate threshold for same-line detection. Boxes whose top-left y
    /// coordinates differ by less than this value are considered on the same line
    /// and sorted left-to-right. Default: 10 pixels.
    static let yThreshold: Int32 = 10

    /// Sort detection boxes in reading order: top-to-bottom, then left-to-right
    /// within the same line (defined by yThreshold).
    ///
    /// Algorithm:
    /// 1. Sort all boxes by top-left corner: first by y, then by x.
    /// 2. Insertion-sort pass: for each box, walk backward and swap with the
    ///    previous box if they are on the same line (y-difference < 10) and
    ///    the current box's x is smaller.
    ///
    /// - Parameter boxes: Unordered detection boxes from DBPostProcessor.
    /// - Returns: Boxes sorted in reading order.
    static func sortInReadingOrder(_ boxes: [DetectionBox]) -> [DetectionBox] {
        guard boxes.count > 1 else { return boxes }

        // Step 1: Sort by top-left point -- first by y, then by x
        var sorted = boxes.sorted { a, b in
            let ay = a.points[0][1], ax = a.points[0][0]
            let by = b.points[0][1], bx = b.points[0][0]
            if ay != by { return ay < by }
            return ax < bx
        }

        // Step 2: Backward insertion sort for same-line boxes
        for i in 0..<(sorted.count - 1) {
            var j = i
            while j >= 0 {
                let nextY = sorted[j + 1].points[0][1]
                let currY = sorted[j].points[0][1]
                let nextX = sorted[j + 1].points[0][0]
                let currX = sorted[j].points[0][0]
                if abs(nextY - currY) < yThreshold && nextX < currX {
                    sorted.swapAt(j, j + 1)
                    j -= 1
                } else {
                    break
                }
            }
        }

        return sorted
    }
}
