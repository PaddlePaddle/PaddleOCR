/*
 * Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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
void transpose_cache_batch_major_kernelLauncher(T* k_dst,
                                                T* v_dst,
                                                const float* k_src,
                                                const float* v_src,
                                                const int* memory_seq_len,
                                                const int local_batch_size,
                                                const int memory_max_seq_len,
                                                const int cache_max_seq_len,
                                                const int size_per_head,
                                                const int local_head_num,
                                                cudaStream_t stream);

template <typename T>
void self_attention_dispatch(const int* memory_sequence_length,
                             T* key_buf,
                             T* value_buf,
                             T* query_buf,
                             const T* self_Q_bias,
                             T* key_cache,
                             const T* self_K_bias,
                             T* value_cache,
                             const T* self_V_bias,
                             T* context_buf,
                             int batch_size,
                             int head_num,
                             int size_per_head,
                             const int step,
                             const int memory_max_seq_len,
                             cudaStream_t stream);
}
