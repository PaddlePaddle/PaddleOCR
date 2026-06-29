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

package com.paddle.ocr.demo

import android.app.Application
import android.util.Log
import com.paddle.ocr.PaddleOCR
import com.paddle.ocr.PaddleOCRConfig
import com.paddle.ocr.util.OpenCVUtils
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class OCRApplication : Application() {

    sealed class ModelState {
        data object Loading : ModelState()
        data class Ready(val ocr: PaddleOCR) : ModelState()
        data class Error(val message: String) : ModelState()
    }

    private val applicationScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val _modelState = MutableStateFlow<ModelState>(ModelState.Loading)

    val modelState: StateFlow<ModelState> = _modelState.asStateFlow()

    val ocr: PaddleOCR?
        get() = (_modelState.value as? ModelState.Ready)?.ocr

    val isModelLoaded: Boolean
        get() = _modelState.value is ModelState.Ready

    override fun onCreate() {
        super.onCreate()
        instance = this
        loadModels()
    }

    fun retryLoadModels() {
        if (_modelState.value is ModelState.Loading) return
        loadModels()
    }

    private fun loadModels() {
        _modelState.value = ModelState.Loading
        applicationScope.launch {
            try {
                if (!OpenCVUtils.init(this@OCRApplication)) {
                    throw IllegalStateException("Failed to initialize OpenCV native library")
                }
                val loadedOcr = PaddleOCR.create(
                    context = this@OCRApplication,
                    config = PaddleOCRConfig(
                        recScoreThresh = 0.0f,
                        recBatchSize = 1,
                    ),
                )
                _modelState.value = ModelState.Ready(loadedOcr)
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                val message = modelLoadErrorMessage(t)
                Log.e(TAG, message, t)
                _modelState.value = ModelState.Error(message)
            }
        }
    }

    private fun modelLoadErrorMessage(error: Throwable): String {
        val details = listOfNotNull(error.message, error.cause?.message)
            .distinct()
            .joinToString(": ")
        return if (details.isBlank()) {
            "Failed to load OCR models"
        } else {
            "Failed to load OCR models: $details"
        }
    }

    override fun onTerminate() {
        super.onTerminate()
        applicationScope.launch {
            ocr?.release()
        }
    }

    companion object {
        private const val TAG = "OCRApplication"

        lateinit var instance: OCRApplication
            private set
    }
}
