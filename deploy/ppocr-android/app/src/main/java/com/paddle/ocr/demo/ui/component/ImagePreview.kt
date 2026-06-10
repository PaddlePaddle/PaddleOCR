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

package com.paddle.ocr.demo.ui.component

import android.graphics.Bitmap
import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.ContentScale
import com.paddle.ocr.model.OCRResult
import kotlin.math.max

private val BOX_COLORS = listOf(
    Color(0xFFE53935),
    Color(0xFF1E88E5),
    Color(0xFF43A047),
    Color(0xFFFB8C00),
    Color(0xFF8E24AA),
    Color(0xFF00ACC1),
    Color(0xFFD81B60),
    Color(0xFF6D4C41),
)

@Composable
fun ImagePreview(
    bitmap: Bitmap,
    results: List<OCRResult>,
    modifier: Modifier = Modifier,
) {
    val imageBitmap: ImageBitmap = bitmap.asImageBitmap()
    val aspect = bitmap.width.toFloat() / bitmap.height.toFloat()

    // Paints for index badges
    val numberPaint = remember {
        Paint().apply {
            color = android.graphics.Color.WHITE
            textSize = 36f
            isAntiAlias = true
            isFakeBoldText = true
        }
    }
    val badgePaint = remember {
        Paint().apply {
            color = 0xCC000000.toInt()  // semi-transparent black
            isAntiAlias = true
            style = Paint.Style.FILL
        }
    }

    Box(modifier = modifier) {
        Image(
            bitmap = imageBitmap,
            contentDescription = "OCR Input",
            contentScale = ContentScale.Fit,
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(aspect),
        )
        if (results.isNotEmpty()) {
            Canvas(modifier = Modifier.matchParentSize()) {
                // Canvas matches Image exactly (both have same aspectRatio),
                // and ContentScale.Fit fills the Image composable with the bitmap
                // (same aspect ratio → no letterboxing), so coordinates map 1:1.
                val scaleX = size.width / bitmap.width
                val scaleY = size.height / bitmap.height

                results.forEachIndexed { index, result ->
                    val color = BOX_COLORS[index % BOX_COLORS.size]
                    val path = Path().apply {
                        result.box.points.forEachIndexed { i, pt ->
                            val x = pt.x * scaleX
                            val y = pt.y * scaleY
                            if (i == 0) moveTo(x, y) else lineTo(x, y)
                        }
                        close()
                    }
                    // Box fill
                    drawPath(path, color = color.copy(alpha = 0.15f))
                    // Box outline
                    drawPath(path, color = color, style = Stroke(width = 3f))

                    // Number badge: rounded rect background + white text
                    val topLeft = result.box.points.minByOrNull { it.y }?.let { pt ->
                        Offset(pt.x * scaleX, pt.y * scaleY)
                    } ?: Offset.Zero

                    val label = "${index + 1}"
                    val textWidth = numberPaint.measureText(label)
                    val badgeH = 44f
                    val badgeW = textWidth + 20f
                    val badgeX = topLeft.x
                    val badgeY = max(0f, topLeft.y - badgeH)

                    // Draw badge background
                    drawContext.canvas.nativeCanvas.drawRoundRect(
                        badgeX, badgeY, badgeX + badgeW, badgeY + badgeH,
                        8f, 8f, badgePaint,
                    )
                    // Draw number centered in badge
                    drawContext.canvas.nativeCanvas.drawText(
                        label,
                        badgeX + (badgeW - textWidth) / 2f,
                        badgeY + badgeH / 2f + 12f,  // center vertically
                        numberPaint,
                    )
                }
            }
        }
    }
}
