/*
 * Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

namespace fastertransformer {

template <typename T>
void topK_softMax_update(
    const T* log_probs,
    const T* bias,  // NOTE: bias is float in V3.1
    bool* finished,
    bool* alive_finished,
    int* sequence_length,
    int* word_ids,
    int* parent_ids,  // for update cache, only include alive beams
    int* output_word_ids,
    int* output_parent_ids,  // for gather tree, include both alive and finish
                             // beams
    float* output_cum_log_probs,  // NOTE: cum_log_probs is T in V3.1
    void* temp_storage,
    const int step,
    DecodingBeamsearchArguments args,
    cudaStream_t stream);

template <typename T>
void topK_update_kernelLauncher(
    void* workspace,
    size_t& workspace_size,
    const T* log_probs,
    bool* finished,
    bool* alive_finished,
    int* sequence_length,
    int* word_ids,
    int* parent_ids,  // for update cache, only include alive beams
    int* output_word_ids,
    int* output_parent_ids,  // for gather tree, include both alive and finish
                             // beams
    float* output_cum_log_probs,  // NOTE: cum_log_probs is T in V3.1
    const int step,
    DecodingBeamsearchArguments args,
    cudaStream_t stream);

}  // namespace fastertransformer
