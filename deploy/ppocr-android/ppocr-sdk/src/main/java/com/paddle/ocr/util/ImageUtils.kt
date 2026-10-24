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

package com.paddle.ocr.util

import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

object ImageUtils {

    fun resizeToMultipleOf32(
        src: Mat,
        limitSideLen: Int,
        limitType: String,
        maxSideLimit: Int,
    ): Mat {
        val h = src.rows()
        val w = src.cols()
        var ratio = when (limitType.lowercase()) {
            "max" -> if (maxOf(h, w) > limitSideLen) limitSideLen.toDouble() / maxOf(h, w) else 1.0
            "min" -> if (minOf(h, w) < limitSideLen) limitSideLen.toDouble() / minOf(h, w) else 1.0
            "resize_long" -> limitSideLen.toDouble() / maxOf(h, w)
            else -> throw IllegalArgumentException("Unsupported det limit type: $limitType")
        }

        var newH = (h * ratio).toInt()
        var newW = (w * ratio).toInt()
        if (maxOf(newH, newW) > maxSideLimit) {
            ratio = maxSideLimit.toDouble() / maxOf(newH, newW)
            newH = (newH * ratio).toInt()
            newW = (newW * ratio).toInt()
        }

        newH = maxOf(MathUtils.roundHalfToEven(newH / 32.0) * 32, 32)
        newW = maxOf(MathUtils.roundHalfToEven(newW / 32.0) * 32, 32)
        val dst = Mat()
        Imgproc.resize(src, dst, Size(newW.toDouble(), newH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        return dst
    }
}
