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

import android.graphics.Bitmap
import com.paddle.ocr.util.BitmapUtils
import com.paddle.ocr.util.ImageUtils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

data class DetPreprocessResult(
    val tensorData: FloatArray,
    val shape: LongArray,
    val originalH: Int,
    val originalW: Int,
)

object DetPreprocessor {
    private val mean = doubleArrayOf(0.485, 0.456, 0.406)
    private val std = doubleArrayOf(0.229, 0.224, 0.225)
    private const val scale = 1.0 / 255.0

    fun preprocess(
        bitmap: Bitmap,
        limitSideLen: Int,
        limitType: String,
        maxSideLimit: Int,
        imgMode: String,
    ): DetPreprocessResult {
        val src = BitmapUtils.bitmapToBGRMat(bitmap)
        return try {
            preprocess(src, limitSideLen, limitType, maxSideLimit, imgMode)
        } finally {
            src.release()
        }
    }

    fun preprocess(
        src: Mat,
        limitSideLen: Int,
        limitType: String,
        maxSideLimit: Int,
        imgMode: String,
    ): DetPreprocessResult {
        val originalH = src.rows()
        val originalW = src.cols()
        val input = if (imgMode.uppercase() == "RGB") {
            Mat().also { Imgproc.cvtColor(src, it, Imgproc.COLOR_BGR2RGB) }
        } else {
            src
        }

        val resized = ImageUtils.resizeToMultipleOf32(input, limitSideLen, limitType, maxSideLimit)
        if (input !== src) input.release()

        val h = resized.rows()
        val w = resized.cols()
        val floatMat = Mat(h, w, CvType.CV_32FC3)
        resized.convertTo(floatMat, CvType.CV_32F)

        val channels = mutableListOf<Mat>()
        Core.split(floatMat, channels)
        for (c in 0..2) {
            Core.multiply(channels[c], Scalar(scale), channels[c])
            Core.subtract(channels[c], Scalar(mean[c]), channels[c])
            Core.divide(channels[c], Scalar(std[c]), channels[c])
        }
        resized.release()
        floatMat.release()

        val channelSize = h * w
        val tensorData = FloatArray(3 * channelSize)
        for (c in 0..2) {
            val chan = channels[c]
            val buf = FloatArray(channelSize)
            chan.get(0, 0, buf)
            System.arraycopy(buf, 0, tensorData, c * channelSize, channelSize)
            chan.release()
        }

        return DetPreprocessResult(
            tensorData = tensorData,
            shape = longArrayOf(1, 3, h.toLong(), w.toLong()),
            originalH = originalH,
            originalW = originalW,
        )
    }
}
