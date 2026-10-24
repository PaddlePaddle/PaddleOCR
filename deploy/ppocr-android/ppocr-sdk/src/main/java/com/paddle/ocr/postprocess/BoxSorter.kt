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

import com.paddle.ocr.model.OCRBox

object BoxSorter {
    private const val ROW_THRESHOLD_Y = 10f

    fun sortInReadingOrder(boxes: List<OCRBox>): List<OCRBox> {
        if (boxes.size <= 1) return boxes

        val list = boxes.toMutableList()
        list.sortWith(compareBy({ it.points[0].y }, { it.points[0].x }))

        // Align with PaddleX SortQuadBoxes: bubble left-to-right inside a 10px row band.
        for (i in 0 until list.size - 1) {
            var j = i
            while (j >= 0) {
                val next = list[j + 1]
                val curr = list[j]
                if (kotlin.math.abs(next.points[0].y - curr.points[0].y) < ROW_THRESHOLD_Y &&
                    next.points[0].x < curr.points[0].x
                ) {
                    list[j] = next
                    list[j + 1] = curr
                    j--
                } else {
                    break
                }
            }
        }
        return list
    }
}
