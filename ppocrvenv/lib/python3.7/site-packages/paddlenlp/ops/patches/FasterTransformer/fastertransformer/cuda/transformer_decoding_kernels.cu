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
__global__ void embeddings_kernels(T* from_tensor,
                                   const T* embedding_table,
                                   const T* position_encoding,
                                   const T* type_table,
                                   const int* memory_sequence_length,
                                   const int* type_id,
                                   const int* word_ids,
                                   const int step,
                                   const int batch_size,
                                   const int hidden_units,
                                   const bool pos_bias) {
  // 1. lookup from embedding table
  // 2. add the position encoding
  // 3. add the token type embedding
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < batch_size * hidden_units;
       index += blockDim.x * gridDim.x) {
    const int row_index = index / hidden_units;
    const int col_index = index % hidden_units;
    int pos = (pos_bias) ? (step - 1 + memory_sequence_length[row_index])
                         : (step - 1);
    from_tensor[index] =
        embedding_table[word_ids[row_index] * hidden_units + col_index] +
        position_encoding[pos * hidden_units + col_index] +
        type_table[type_id[row_index] * hidden_units + col_index];
  }
}

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
                                cudaStream_t stream) {
  dim3 grid(min(batch_size, 65536));
  dim3 block(min(hidden_units, 1024));

  embeddings_kernels<T><<<grid, block, 0, stream>>>(from_tensor,
                                                    embedding_table,
                                                    position_encoding_table,
                                                    type_table,
                                                    memory_sequence_length,
                                                    type_id,
                                                    word_ids,
                                                    step,
                                                    batch_size,
                                                    hidden_units,
                                                    pos_bias);
}


template <typename T>
__global__ void initial_cache_kernel(const float* cache_k,
                                     const float* cache_v,
                                     const int* memory_sequence_length,
                                     T* k_tgt,
                                     T* v_tgt,
                                     int n_head,
                                     int size_per_head,
                                     int mem_len,
                                     int batch_size,
                                     int beam_size = 1) {
  int tid = threadIdx.x;
  int bid = blockIdx.x / (beam_size * n_head);
  int beam_id = blockIdx.x % (n_head * beam_size) / n_head;
  int head_id = blockIdx.x % n_head;

  int offset = batch_size * beam_size * n_head * size_per_head;

  for (int ite = 0; ite < mem_len; ++ite) {
    int tgt_id = bid * beam_size * n_head * size_per_head +
                 beam_id * n_head * size_per_head + head_id * size_per_head +
                 tid;
    int src_id = bid * n_head * mem_len * size_per_head +
                 head_id * mem_len * size_per_head + ite * size_per_head + tid;
    k_tgt[ite * offset + tgt_id] = static_cast<T>(cache_k[src_id]);
    v_tgt[ite * offset + tgt_id] = static_cast<T>(cache_v[src_id]);
  }
}

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
                                cudaStream_t stream) {
  initial_cache_kernel<
      T><<<batch_size * beam_size * n_head, size_per_head, 0, stream>>>(
      cache_k,
      cache_v,
      memory_sequence_length,
      k_tgt,
      v_tgt,
      n_head,
      size_per_head,
      mem_len,
      batch_size,
      beam_size);
}

template <typename T>
__global__ void init_logits_mask_kernel(T* logits_mask,
                                        int vocab_size,
                                        int start_id = -1,
                                        int unk_id = -1,
                                        int mask_id = -1) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (bid * blockDim.x + tid == start_id) {
    logits_mask[bid * blockDim.x + tid] = static_cast<T>(-1e20f);
  } else if (bid * blockDim.x + tid == unk_id) {
    logits_mask[bid * blockDim.x + tid] = static_cast<T>(-1e20f);
  } else if (bid * blockDim.x + tid == mask_id) {
    logits_mask[bid * blockDim.x + tid] = static_cast<T>(-1e20f);
  } else if (bid * blockDim.x + tid < vocab_size) {
    logits_mask[bid * blockDim.x + tid] = static_cast<T>(0.0f);
  }
}

template <typename T>
void init_logits_mask_Launcher(T* logits_mask,
                               int vocab_size,
                               int start_id,
                               int unk_id,
                               int mask_id,
                               cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((vocab_size + block.x - 1) / block.x);

  init_logits_mask_kernel<T><<<grid, block, 0, stream>>>(
      logits_mask, vocab_size, start_id, unk_id, mask_id);
}

template <typename T>
__global__ void apply_penalties_kernel(int step,
                                       int vocab_size,
                                       int beam_width,
                                       T* log_probs,
                                       const bool* finished,
                                       int* current_ids,
                                       int* previous_ids,
                                       int* parent_ids,
                                       int end_id,
                                       float inv_temp,
                                       float len_penalty,
                                       float repeat_penalty,
                                       const T* logits_mask = nullptr) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int bbid = blockIdx.y;   // batch_size * beam_size: index
  int bbsize = gridDim.y;  // batch_size * beam_size
  int batchid = bbid / beam_width;
  // int beamid = bbid % beam_width;

  bool finish = (finished != nullptr) ? finished[bbid] : false;

  if (!finish) {
    // temperature
    if (inv_temp != 1.0) {
      for (int i = tid + bid * blockDim.x; i < vocab_size;
           i += blockDim.x * gridDim.x) {
        log_probs[i + bbid * vocab_size] *= inv_temp;
      }
    }

    if (tid == 0 && bid == 0) {
      // apply repetition penalty (this can apply the penalty multiple times to
      // a
      // repeated word).
      if (repeat_penalty != 1.0) {
        int prev_id = current_ids[bbid];
        if (log_probs[prev_id + bbid * vocab_size] > T(0)) {
          log_probs[prev_id + bbid * vocab_size] =
              float(log_probs[prev_id + bbid * vocab_size]) / repeat_penalty;
        } else {
          log_probs[prev_id + bbid * vocab_size] =
              float(log_probs[prev_id + bbid * vocab_size]) * repeat_penalty;
        }

        if (step > 1) {
          int parent_beamid = parent_ids[bbsize * (step - 2) + bbid];
          for (int i = step - 2; i > 0; --i) {
            prev_id =
                previous_ids[i * bbsize + batchid * beam_width + parent_beamid];
            if (log_probs[prev_id + bbid * vocab_size] > T(0)) {
              log_probs[prev_id + bbid * vocab_size] =
                  float(log_probs[prev_id + bbid * vocab_size]) /
                  repeat_penalty;
            } else {
              log_probs[prev_id + bbid * vocab_size] =
                  float(log_probs[prev_id + bbid * vocab_size]) *
                  repeat_penalty;
            }
            // if (i > 0) parent_beamid =
            // parent_ids[bbsize*(i-1)+parent_beamid];
            parent_beamid = parent_ids[bbsize * (i - 1) + parent_beamid];
          }
        }
        prev_id = previous_ids[batchid * beam_width];
        if (log_probs[prev_id + bbid * vocab_size] > T(0)) {
          log_probs[prev_id + bbid * vocab_size] =
              float(log_probs[prev_id + bbid * vocab_size]) / repeat_penalty;
        } else {
          log_probs[prev_id + bbid * vocab_size] =
              float(log_probs[prev_id + bbid * vocab_size]) * repeat_penalty;
        }
      }

      // apply length penalty
      // NOTE: The length penalty has different implementation. May be update.
      if (len_penalty != 1.0) {
        if (log_probs[end_id + bbid * vocab_size] > T(0)) {
          log_probs[end_id + bbid * vocab_size] =
              float(log_probs[end_id + bbid * vocab_size]) / len_penalty;
        } else {
          log_probs[end_id + bbid * vocab_size] =
              float(log_probs[end_id + bbid * vocab_size]) * len_penalty;
        }
      }
    }

    if (logits_mask) {
      for (int i = tid + bid * blockDim.x; i < vocab_size;
           i += blockDim.x * gridDim.x) {
        log_probs[i + bbid * vocab_size] += logits_mask[i];
      }
    }
  }
}

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
                              const T* logits_mask = nullptr) {
  dim3 block(256);
  dim3 grid((vocab_size + block.x - 1) / block.x, beam_width * batch_size);

  apply_penalties_kernel<T><<<grid, block, 0, stream>>>(step,
                                                        vocab_size,
                                                        beam_width,
                                                        log_probs,
                                                        finished,
                                                        current_ids,
                                                        previous_ids,
                                                        parent_ids,
                                                        end_id,
                                                        1.f / temperature,
                                                        len_penalty,
                                                        repeat_penalty,
                                                        logits_mask);
}

template void embeddings_kernel_launcher(float* from_tensor,
                                         const float* embedding_table,
                                         const float* position_encoding_table,
                                         const float* sent_table,
                                         const int* memory_sequence_length,
                                         const int* type_id,
                                         const int* word_ids,
                                         const int step,
                                         const int batch_size,
                                         const int hidden_units,
                                         const bool pos_bias,
                                         cudaStream_t stream);

template void embeddings_kernel_launcher(half* from_tensor,
                                         const half* embedding_table,
                                         const half* position_encoding_table,
                                         const half* sent_table,
                                         const int* memory_sequence_length,
                                         const int* type_id,
                                         const int* word_ids,
                                         const int step,
                                         const int batch_size,
                                         const int hidden_units,
                                         const bool pos_bias,
                                         cudaStream_t stream);

template void init_cache_kernel_launcher(const float* cache_k,
                                         const float* cache_v,
                                         const int* memory_sequence_length,
                                         float* k_tgt,
                                         float* v_tgt,
                                         int n_head,
                                         int size_per_head,
                                         int mem_len,
                                         int batch_size,
                                         int beam_size,
                                         cudaStream_t stream);

template void init_cache_kernel_launcher(const float* cache_k,
                                         const float* cache_v,
                                         const int* memory_sequence_length,
                                         half* k_tgt,
                                         half* v_tgt,
                                         int n_head,
                                         int size_per_head,
                                         int mem_len,
                                         int batch_size,
                                         int beam_size,
                                         cudaStream_t stream);

template void init_logits_mask_Launcher(float* logits_mask,
                                        int vocab_size,
                                        int start_id,
                                        int unk_id,
                                        int mask_id,
                                        cudaStream_t stream);

template void init_logits_mask_Launcher(half* logits_mask,
                                        int vocab_size,
                                        int start_id,
                                        int unk_id,
                                        int mask_id,
                                        cudaStream_t stream);

template void apply_penalties_Launcher(int step,
                                       float* log_probs,
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
                                       const float* logits_mask);

template void apply_penalties_Launcher(int step,
                                       half* log_probs,
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
                                       const half* logits_mask);
}
