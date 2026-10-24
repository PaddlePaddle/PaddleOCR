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

package com.paddle.ocr.demo.ui.viewmodel

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.paddle.ocr.demo.OCRApplication
import com.paddle.ocr.model.OCRResult
import com.paddle.ocr.model.OCRRunResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class OCRViewModel : ViewModel() {

    sealed class UIState {
        data object Loading : UIState()
        data object Ready : UIState()
        data class Processing(val bitmap: Bitmap) : UIState()
        data class Result(val bitmap: Bitmap, val result: OCRRunResult) : UIState()
        data class Error(val message: String) : UIState()
    }

    data class TimingInfo(
        val detectionMs: Long,
        val recognitionMs: Long,
        val totalMs: Long,
    )

    private val _uiState = MutableStateFlow<UIState>(
        uiStateForModelState(OCRApplication.instance.modelState.value)
    )
    val uiState: StateFlow<UIState> = _uiState.asStateFlow()

    private val _timing = MutableStateFlow<TimingInfo?>(null)
    val timing: StateFlow<TimingInfo?> = _timing.asStateFlow()

    private fun uiStateForModelState(modelState: OCRApplication.ModelState): UIState {
        return when (modelState) {
            is OCRApplication.ModelState.Loading -> UIState.Loading
            is OCRApplication.ModelState.Ready -> UIState.Ready
            is OCRApplication.ModelState.Error -> UIState.Error(modelState.message)
        }
    }

    init {
        viewModelScope.launch {
            OCRApplication.instance.modelState.collect { modelState ->
                when (modelState) {
                    is OCRApplication.ModelState.Loading -> _uiState.value = UIState.Loading
                    is OCRApplication.ModelState.Ready -> {
                        if (_uiState.value is UIState.Loading || _uiState.value is UIState.Error) {
                            _uiState.value = UIState.Ready
                        }
                    }
                    is OCRApplication.ModelState.Error -> _uiState.value = UIState.Error(modelState.message)
                }
            }
        }
    }

    fun onImageSelected(uri: Uri) {
        viewModelScope.launch {
            try {
                val bytes = withContext(Dispatchers.IO) {
                    OCRApplication.instance.contentResolver.openInputStream(uri)?.use { it.readBytes() }
                }
                if (bytes != null) {
                    val bitmap = withContext(Dispatchers.IO) {
                        decodeSampledBitmap(bytes, maxWidth = 2048, maxHeight = 2048)
                    }
                    if (bitmap != null) {
                        processImageBytes(bytes, bitmap)
                    } else {
                        _uiState.value = UIState.Error("Failed to load image")
                    }
                } else {
                    _uiState.value = UIState.Error("Failed to load image")
                }
            } catch (e: Exception) {
                _uiState.value = UIState.Error(e.message ?: "Unknown error")
            }
        }
    }

    fun onSampleImageClicked(resId: Int) {
        viewModelScope.launch {
            val bytes = withContext(Dispatchers.IO) {
                OCRApplication.instance.resources.openRawResource(resId).use { it.readBytes() }
            }
            val bitmap = withContext(Dispatchers.IO) {
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            }
            if (bitmap != null) {
                processImageBytes(bytes, bitmap)
            } else {
                _uiState.value = UIState.Error("Failed to load image")
            }
        }
    }

    private suspend fun processImageBytes(bytes: ByteArray, bitmap: Bitmap) {
        _uiState.value = UIState.Processing(bitmap)

        try {
            val ocr = OCRApplication.instance.ocr
                ?: throw IllegalStateException("OCR engine not initialized")

            val result = ocr.recognize(bytes)
            _uiState.value = UIState.Result(bitmap, result)
            _timing.value = TimingInfo(
                detectionMs = result.detectionTimeMs,
                recognitionMs = result.recognitionTimeMs,
                totalMs = result.totalTimeMs,
            )
        } catch (e: Exception) {
            _uiState.value = UIState.Error(e.message ?: "OCR failed")
        }
    }

    fun retry() {
        val app = OCRApplication.instance
        if (app.isModelLoaded) {
            _uiState.value = UIState.Ready
        } else {
            app.retryLoadModels()
        }
    }

    fun copyAllResults(results: List<OCRResult>) {
        val text = results.joinToString("\n") { it.text }
        val clipboard = OCRApplication.instance.getSystemService(android.content.Context.CLIPBOARD_SERVICE)
                as android.content.ClipboardManager
        clipboard.setPrimaryClip(android.content.ClipData.newPlainText("OCR Results", text))
    }

    private fun decodeSampledBitmap(
        bytes: ByteArray,
        maxWidth: Int,
        maxHeight: Int,
    ): Bitmap? {
        // First pass: decode bounds only
        val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
        val (origW, origH) = opts.outWidth to opts.outHeight
        if (origW <= 0 || origH <= 0) return null

        // Calculate sample size (power of 2)
        var sampleSize = 1
        while (sampleSize * 2 <= maxOf(origW / maxWidth, origH / maxHeight)) {
            sampleSize *= 2
        }

        // Second pass: decode with sampling
        return BitmapFactory.Options().apply {
            inSampleSize = sampleSize
        }.let { BitmapFactory.decodeByteArray(bytes, 0, bytes.size, it) }
    }
}
