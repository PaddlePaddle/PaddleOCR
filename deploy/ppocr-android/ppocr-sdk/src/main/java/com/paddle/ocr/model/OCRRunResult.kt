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

package com.paddle.ocr.model

data class OCRRunResult(
    val results: List<OCRResult>,
    val detectionTimeMs: Long,
    val recognitionTimeMs: Long,
    val totalTimeMs: Long,
    val lineCount: Int,
    // Detailed per-stage timing
    val detPreprocessMs: Long = 0,
    val detInferenceMs: Long = 0,
    val detPostprocessMs: Long = 0,
    val recPreprocessMs: Long = 0,
    val recInferenceMs: Long = 0,
    val recPostprocessMs: Long = 0,
    val pipelineOverheadMs: Long = 0,
    val coldLoadTimeMs: Long = 0,
    // Input tensor shapes
    val detInputShape: List<Int> = emptyList(),
    val recInputShapes: List<List<Int>> = emptyList(),
    // Per-line timing (only populated when recBatchSize == 1)
    val perLineRecMs: List<Long> = emptyList(),
)
