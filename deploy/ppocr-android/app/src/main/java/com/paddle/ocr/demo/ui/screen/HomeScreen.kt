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

package com.paddle.ocr.demo.ui.screen

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.paddle.ocr.demo.ui.component.*
import com.paddle.ocr.demo.ui.viewmodel.OCRViewModel

@Composable
fun HomeScreen(viewModel: OCRViewModel = viewModel()) {
    val state by viewModel.uiState.collectAsState()
    val timing by viewModel.timing.collectAsState()

    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri -> uri?.let { viewModel.onImageSelected(it) } }

    Column(modifier = Modifier.fillMaxSize()) {
        when (val s = state) {
            is OCRViewModel.UIState.Loading -> {
                LoadingOverlay("Loading OCR models...")
            }
            is OCRViewModel.UIState.Ready -> {
                Column(modifier = Modifier.verticalScroll(rememberScrollState())) {
                    Spacer(modifier = Modifier.height(32.dp))
                    ImagePicker(
                        onGalleryClick = { galleryLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                        onSampleClick = { viewModel.onSampleImageClicked(it) },
                        sampleImages = emptyList(),
                    )
                }
            }
            is OCRViewModel.UIState.Processing -> {
                Column(modifier = Modifier.verticalScroll(rememberScrollState())) {
                    Spacer(modifier = Modifier.height(16.dp))
                    ImagePreview(bitmap = s.bitmap, results = emptyList())
                    Spacer(modifier = Modifier.height(16.dp))
                    LoadingOverlay("Processing...")
                }
            }
            is OCRViewModel.UIState.Result -> {
                Column(modifier = Modifier.verticalScroll(rememberScrollState())) {
                    ImagePreview(
                        bitmap = s.bitmap,
                        results = s.result.results,
                        modifier = Modifier.fillMaxWidth(),
                    )
                    if (timing != null) {
                        TimingBar(
                            detectionMs = timing!!.detectionMs,
                            recognitionMs = timing!!.recognitionMs,
                            totalMs = timing!!.totalMs,
                        )
                    }
                    Spacer(modifier = Modifier.height(4.dp))
                    ResultList(
                        results = s.result.results,
                        onCopyAll = { viewModel.copyAllResults(s.result.results) },
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    ImagePicker(
                        onGalleryClick = { galleryLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                        onSampleClick = { viewModel.onSampleImageClicked(it) },
                        sampleImages = emptyList(),
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                }
            }
            is OCRViewModel.UIState.Error -> {
                LoadingOverlay("Error occurred")
                ErrorDialog(
                    message = s.message,
                    onRetry = { viewModel.retry() },
                    onDismiss = { viewModel.retry() },
                )
            }
        }
    }
}
