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

package com.paddle.ocr

data class PaddleOCRConfig(
    val detImgMode: String = "BGR",
    val detLimitSideLen: Int = 64,
    val detLimitType: String = "min",
    val detMaxSideLimit: Int = 4000,
    val detThresh: Float = 0.3f,
    val detBoxThresh: Float = 0.6f,
    val detUnclipRatio: Float = 1.5f,
    val detMaxCandidates: Int = 3000,
    val detUseDilation: Boolean = false,
    val detScoreMode: String = "fast",
    val detBoxType: String = "quad",
    val recScoreThresh: Float = 0.0f,
    val recBatchSize: Int = 1,
)
