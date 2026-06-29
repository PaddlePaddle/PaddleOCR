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

package com.paddle.ocr.postprocess

import org.opencv.core.Point

internal object QuadGeometry {
    fun orderMinAreaRectPoints(points: Array<Point>): List<Point> {
        val sorted = points.sortedBy { it.x }
        val topLeft: Point
        val bottomLeft: Point
        if (sorted[1].y > sorted[0].y) {
            topLeft = sorted[0]
            bottomLeft = sorted[1]
        } else {
            topLeft = sorted[1]
            bottomLeft = sorted[0]
        }

        val topRight: Point
        val bottomRight: Point
        if (sorted[3].y > sorted[2].y) {
            topRight = sorted[2]
            bottomRight = sorted[3]
        } else {
            topRight = sorted[3]
            bottomRight = sorted[2]
        }
        return listOf(topLeft, topRight, bottomRight, bottomLeft)
    }
}
