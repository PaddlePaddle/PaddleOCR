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

package com.paddle.ocr.preprocess

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.ceil

data class RecPreprocessResult(
    val tensorData: FloatArray,
    val shape: LongArray,
)

object RecPreprocessor {
    private const val FIXED_HEIGHT = 48
    private const val MAX_IMG_W = 3200

    fun preprocessBatch(crops: List<Mat>): RecPreprocessResult {
        // Convert BGR to RGB and resize to fixed height while preserving aspect ratio
        val resizedMats = mutableListOf<Mat>()
        for (crop in crops) {
            // Convert BGR to RGB (model expects RGB input)
            val rgb = Mat()
            Imgproc.cvtColor(crop, rgb, Imgproc.COLOR_BGR2RGB)
            val h = rgb.rows()
            val w = rgb.cols()
            val aspectRatio = if (h > 0) w.toDouble() / h else 1.0
            val newW = ceil(FIXED_HEIGHT * aspectRatio).toInt().coerceAtMost(MAX_IMG_W)
            val dst = Mat()
            Imgproc.resize(rgb, dst, Size(newW.toDouble(), FIXED_HEIGHT.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            rgb.release()
            resizedMats.add(dst)
        }

        // Convert to float and normalize: (x / 255 - 0.5) / 0.5 = x / 127.5 - 1
        val floatMats = mutableListOf<Mat>()
        for (mat in resizedMats) {
            val floatMat = Mat(mat.rows(), mat.cols(), CvType.CV_32FC3)
            mat.convertTo(floatMat, CvType.CV_32F)
            // Use Scalar with all 3 channels set — single-value Scalar only sets val[0]!
            Core.divide(floatMat, org.opencv.core.Scalar(127.5, 127.5, 127.5), floatMat)
            Core.subtract(floatMat, org.opencv.core.Scalar(1.0, 1.0, 1.0), floatMat)

            floatMats.add(floatMat)
            mat.release()  // Release resized mat
        }
        resizedMats.clear()

        val maxW = floatMats.maxOf { it.cols() }
        val n = floatMats.size

        // Pad to max width
        val paddedMats = mutableListOf<Mat>()
        for (mat in floatMats) {
            if (mat.cols() == maxW) {
                paddedMats.add(mat)
            } else {
                val padded = Mat(FIXED_HEIGHT, maxW, CvType.CV_32FC3, org.opencv.core.Scalar(0.0))
                val roi = padded.submat(0, FIXED_HEIGHT, 0, mat.cols())
                mat.copyTo(roi)
                roi.release()
                mat.release()
                paddedMats.add(padded)
            }
        }
        floatMats.clear()

        // Build tensor data
        val channelSize = FIXED_HEIGHT * maxW
        val tensorData = FloatArray(n * 3 * channelSize)
        for (b in 0 until n) {
            val mat = paddedMats[b]
            val channels = mutableListOf<Mat>()
            Core.split(mat, channels)
            for (c in 0..2) {
                val buf = FloatArray(channelSize)
                channels[c].get(0, 0, buf)
                System.arraycopy(buf, 0, tensorData, (b * 3 + c) * channelSize, channelSize)
                channels[c].release()
            }
            mat.release()
        }
        paddedMats.clear()

        return RecPreprocessResult(
            tensorData = tensorData,
            shape = longArrayOf(n.toLong(), 3, FIXED_HEIGHT.toLong(), maxW.toLong()),
        )
    }
}
