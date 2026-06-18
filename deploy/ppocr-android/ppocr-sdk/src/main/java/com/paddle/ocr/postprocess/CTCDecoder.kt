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

object CTCDecoder {
    private const val BLANK_IDX = 0

    fun decode(output: FloatArray, shape: LongArray, characterList: List<String>): List<Pair<String, Float>> {
        val batchSize = shape[0].toInt()
        val timeSteps = shape[1].toInt()
        val numClasses = shape[2].toInt()

        val results = mutableListOf<Pair<String, Float>>()
        for (b in 0 until batchSize) {
            val baseOffset = b * timeSteps * numClasses

            val indices = IntArray(timeSteps)
            val probs = FloatArray(timeSteps)
            for (t in 0 until timeSteps) {
                val offset = baseOffset + t * numClasses
                var maxIdx = 0
                var maxVal = output[offset]
                for (c in 1 until numClasses) {
                    val v = output[offset + c]
                    if (v > maxVal) {
                        maxVal = v
                        maxIdx = c
                    }
                }
                indices[t] = maxIdx
                probs[t] = maxVal
            }

            val keptProbs = mutableListOf<Float>()
            val sb = StringBuilder()
            var prevIdx = -1
            for (t in 0 until timeSteps) {
                val idx = indices[t]
                if (idx != BLANK_IDX && idx != prevIdx) {
                    val charIdx = idx - 1
                    if (charIdx >= 0 && charIdx < characterList.size) {
                        sb.append(characterList[charIdx])
                        keptProbs.add(probs[t])
                    }
                }
                prevIdx = idx
            }

            val confidence = if (keptProbs.isNotEmpty()) keptProbs.average().toFloat() else 0f
            results.add(Pair(sb.toString(), confidence))
        }
        return results
    }
}
