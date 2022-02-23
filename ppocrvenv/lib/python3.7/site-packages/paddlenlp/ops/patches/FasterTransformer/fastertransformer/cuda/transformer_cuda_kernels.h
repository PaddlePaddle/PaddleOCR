/*
 * Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
void embeddings_kernel_launcher(T* from_tensor,
                                const T* embedding_table,
                                const T* position_encoding_table,
                                const T* type_table,
                                const int* memory_sequence_length,
                                const int* type_id,
                                const int* word_ids,
                                const int step,
                                const int batch_size,
                                const int hidden_units,
                                const bool pos_bias,
                                cudaStream_t stream);

template <typename T>
void init_cache_kernel_launcher(const float* cache_k,
                                const float* cache_v,
                                const int* memory_sequence_length,
                                T* k_tgt,
                                T* v_tgt,
                                int n_head,
                                int size_per_head,
                                int mem_len,
                                int batch_size,
                                int beam_size,
                                cudaStream_t stream);

template <typename T>
void init_logits_mask_Launcher(T* logits_mask,
                               int vocab_size,
                               int start_id,
                               int unk_id,
                               int mask_id,
                               cudaStream_t stream);

template <typename T>
void apply_penalties_Launcher(int step,
                              T* log_probs,
                              const bool* finished,
                              int* current_ids,
                              int* previous_ids,
                              int* parent_ids,
                              int batch_size,
                              int beam_width,
                              int vocab_size,
                              int end_id,
                              float temperature,
                              float len_penalty,
                              float repeat_penalty,
                              cudaStream_t stream,
                              const T* logits_mask);

template <typename T>
void update_KV_cache_kernelLauncher(T** key_cache,
                                    T** value_cache,
                                    const int* beam_ids,
                                    const int batch_size,
                                    const int beam_width,
                                    const int hidden_dim,
                                    const int step,
                                    const int start_len,
                                    const int cache_size,
                                    const int decoder_layers,
                                    cudaStream_t stream);
}
