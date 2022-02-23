#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <iterator>
#include <random>
#include <sstream>
#include <vector>

#include "fastertransformer/cuda/cub/cub.cuh"
#include "fusion_gpt_op.h"
#include "pd_traits.h"


template <paddle::DataType D>
std::vector<paddle::Tensor> gpt2_kernel(
    const paddle::Tensor& input,
    const paddle::Tensor& attn_mask,
    const paddle::Tensor& start_length,
    const paddle::Tensor& word_emb,
    const std::vector<paddle::Tensor>& self_ln_weight,
    const std::vector<paddle::Tensor>& self_ln_bias,
    const std::vector<paddle::Tensor>& self_q_weight,
    const std::vector<paddle::Tensor>& self_q_bias,
    const std::vector<paddle::Tensor>& self_k_weight,
    const std::vector<paddle::Tensor>& self_k_bias,
    const std::vector<paddle::Tensor>& self_v_weight,
    const std::vector<paddle::Tensor>& self_v_bias,
    const std::vector<paddle::Tensor>& self_out_weight,
    const std::vector<paddle::Tensor>& self_out_bias,
    const std::vector<paddle::Tensor>& ffn_ln_weight,
    const std::vector<paddle::Tensor>& ffn_ln_bias,
    const std::vector<paddle::Tensor>& ffn_inter_weight,
    const std::vector<paddle::Tensor>& ffn_inter_bias,
    const std::vector<paddle::Tensor>& ffn_out_weight,
    const std::vector<paddle::Tensor>& ffn_out_bias,
    const paddle::Tensor& decoder_ln_weight,
    const paddle::Tensor& decoder_ln_bias,
    const paddle::Tensor& positional_embedding_weight,
    const paddle::Tensor& emb_weight,
    paddle::Tensor& output_ids,
    const int& topk,
    const float& topp,
    const int& max_len,
    const int& n_head,
    const int& size_per_head,
    const int& num_layer,
    const int& bos_id,
    const int& eos_id,
    const float& temperature,
    cublasHandle_t cublas_handle_,
    cublasLtHandle_t cublaslt_handle_,
    cudaStream_t stream) {
  auto input_dims = input.shape();
  int batch_size_ = input_dims[0];
  int start_len = input_dims[1];
  const int vocab_size = word_emb.shape()[0];

  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t_;

  DecodingInitParam<DataType_> decoding_params;
  decoding_params.cublas_handle = cublas_handle_;
  decoding_params.cublaslt_handle = cublaslt_handle_;

  decoding_params.output_ids = output_ids.mutable_data<int>(word_emb.place());

  typedef DecoderTransformerTraits<traits_::OpType> DecodingTraits_;
  decoding_params.stream = stream;
  fastertransformer::Allocator<AllocatorType::PD> allocator_(stream);

  const int hidden_unit = size_per_head * n_head;

  TensorParallelParam tensor_parallel_param;
  LayerParallelParam layer_parallel_param;
  // TODO: multi-cards supports.
  // ncclComm_t tensor_para_nccl_comm, layer_para_nccl_comm;

  tensor_parallel_param.rank = 0;
  tensor_parallel_param.world_size = 1;
  // TODO: multi-cards supports.
  // tensor_parallel_param.nccl_comm = tensor_para_nccl_comm;
  tensor_parallel_param.local_head_num_ = n_head;
  tensor_parallel_param.local_hidden_units_ = hidden_unit;

  layer_parallel_param.rank = 0;
  layer_parallel_param.world_size = 1;
  // TODO: multi-cards supports.
  // layer_parallel_param.nccl_comm = layer_para_nccl_comm;
  layer_parallel_param.layers_per_group = num_layer;
  layer_parallel_param.local_batch_size = batch_size_;

  DecodingGpt<DecodingTraits_::OpType>* gpt_decoding;

  decoding_params.request_batch_size = batch_size_;
  decoding_params.max_input_len = start_len;
  decoding_params.request_input_len = start_len;
  decoding_params.request_output_len = max_len - start_len;

  decoding_params.d_start_ids = const_cast<int *>(input.data<int>());
  decoding_params.d_attn_mask =
      reinterpret_cast<DataType_*>(const_cast<data_t_ *>(attn_mask.data<data_t_>()));
  decoding_params.d_start_lengths = start_length.data<int>();

  gpt_decoding =
      new DecodingGpt<DecodingTraits_::OpType>(allocator_,
                                               batch_size_,
                                               max_len,
                                               n_head,
                                               size_per_head,
                                               vocab_size,
                                               num_layer,
                                               bos_id,
                                               eos_id,
                                               topk,
                                               topp,
                                               temperature,
                                               1, /*tensor_para_size*/
                                               1, /*layer_para_size*/
                                               true /*is_fuse_QKV*/);

  gpt_decoding->set_tensor_parallel_param(tensor_parallel_param);
  gpt_decoding->set_layer_parallel_param(layer_parallel_param);

  DecoderInitParam<DataType_>* params =
      new DecoderInitParam<DataType_>[num_layer];

  for (int i = 0; i < num_layer; ++i) {
    if (layer_parallel_param.is_valid(i) == false) {
      continue;
    }

    params[i].stream = stream;
    params[i].cublas_handle = cublas_handle_;
    params[i].cublaslt_handle = cublaslt_handle_;

    params[i].request_batch_size = batch_size_;
    params[i].request_max_mem_seq_len = start_len;

    params[i].self_layernorm.gamma =
        reinterpret_cast<const DataType_*>(self_ln_weight[i].data<data_t_>());
    params[i].self_layernorm.beta =
        reinterpret_cast<const DataType_*>(self_ln_bias[i].data<data_t_>());

    params[i].self_attention.query_weight.kernel =
        reinterpret_cast<const DataType_*>(self_q_weight[i].data<data_t_>());
    params[i].self_attention.query_weight.bias =
        reinterpret_cast<const DataType_*>(self_q_bias[i].data<data_t_>());
    params[i].self_attention.key_weight.kernel =
        reinterpret_cast<const DataType_*>(self_k_weight[i].data<data_t_>());
    params[i].self_attention.key_weight.bias =
        reinterpret_cast<const DataType_*>(self_k_bias[i].data<data_t_>());
    params[i].self_attention.value_weight.kernel =
        reinterpret_cast<const DataType_*>(self_v_weight[i].data<data_t_>());
    params[i].self_attention.value_weight.bias =
        reinterpret_cast<const DataType_*>(self_v_bias[i].data<data_t_>());

    params[i].self_attention.attention_output_weight.kernel =
        reinterpret_cast<const DataType_*>(self_out_weight[i].data<data_t_>());
    params[i].self_attention.attention_output_weight.bias =
        reinterpret_cast<const DataType_*>(self_out_bias[i].data<data_t_>());

    params[i].ffn_layernorm.gamma =
        reinterpret_cast<const DataType_*>(ffn_ln_weight[i].data<data_t_>());
    params[i].ffn_layernorm.beta =
        reinterpret_cast<const DataType_*>(ffn_ln_bias[i].data<data_t_>());

    params[i].ffn.intermediate_weight.kernel =
        reinterpret_cast<const DataType_*>(ffn_inter_weight[i].data<data_t_>());
    params[i].ffn.intermediate_weight.bias =
        reinterpret_cast<const DataType_*>(ffn_inter_bias[i].data<data_t_>());
    params[i].ffn.output_weight.kernel =
        reinterpret_cast<const DataType_*>(ffn_out_weight[i].data<data_t_>());
    params[i].ffn.output_weight.bias =
        reinterpret_cast<const DataType_*>(ffn_out_bias[i].data<data_t_>());
  }

  decoding_params.layernorm.gamma =
      reinterpret_cast<const DataType_*>(decoder_ln_weight.data<data_t_>());
  decoding_params.layernorm.beta =
      reinterpret_cast<const DataType_*>(decoder_ln_bias.data<data_t_>());
  decoding_params.embedding_table =
      reinterpret_cast<const DataType_*>(word_emb.data<data_t_>());
  decoding_params.embedding_kernel =
      reinterpret_cast<const DataType_*>(emb_weight.data<data_t_>());
  decoding_params.position_encoding_table = reinterpret_cast<const DataType_*>(
      positional_embedding_weight.data<data_t_>());

  gpt_decoding->forward_context(params, decoding_params);
  gpt_decoding->forward(params, decoding_params);

  delete gpt_decoding;
  delete[] params;

  return {output_ids};
}

std::vector<paddle::Tensor> GPT2CUDAForward(
    const paddle::Tensor& input,
    const paddle::Tensor& attn_mask,
    const paddle::Tensor& start_length,
    const paddle::Tensor& word_embedding,
    const std::vector<paddle::Tensor>& self_ln_weight,
    const std::vector<paddle::Tensor>& self_ln_bias,
    const std::vector<paddle::Tensor>& self_q_weight,
    const std::vector<paddle::Tensor>& self_q_bias,
    const std::vector<paddle::Tensor>& self_k_weight,
    const std::vector<paddle::Tensor>& self_k_bias,
    const std::vector<paddle::Tensor>& self_v_weight,
    const std::vector<paddle::Tensor>& self_v_bias,
    const std::vector<paddle::Tensor>& self_out_weight,
    const std::vector<paddle::Tensor>& self_out_bias,
    const std::vector<paddle::Tensor>& ffn_ln_weight,
    const std::vector<paddle::Tensor>& ffn_ln_bias,
    const std::vector<paddle::Tensor>& ffn_inter_weight,
    const std::vector<paddle::Tensor>& ffn_inter_bias,
    const std::vector<paddle::Tensor>& ffn_out_weight,
    const std::vector<paddle::Tensor>& ffn_out_bias,
    const paddle::Tensor& decoder_ln_weight,
    const paddle::Tensor& decoder_ln_bias,
    const paddle::Tensor& positional_embedding_weight,
    const paddle::Tensor& emb_weight,
    paddle::Tensor& output_ids,
    const int& topk,
    const float& topp,
    const int& max_len,
    const int& n_head,
    const int& size_per_head,
    const int& num_layer,
    const int& bos_id,
    const int& eos_id,
    const float& temperature,
    const bool& use_fp16 = false) {
  auto stream = word_embedding.stream();
  cublasHandle_t cublas_handle_;
  cublasCreate(&cublas_handle_);
  cublasLtHandle_t cublaslt_handle_;
  cublasLtCreate(&cublaslt_handle_);
  cublasSetStream(cublas_handle_, stream);

  std::vector<paddle::Tensor> ret;

  if (use_fp16) {
    ret = gpt2_kernel<paddle::DataType::FLOAT16>(input,
                                                 attn_mask,
                                                 start_length,
                                                 word_embedding,
                                                 self_ln_weight,
                                                 self_ln_bias,
                                                 self_q_weight,
                                                 self_q_bias,
                                                 self_k_weight,
                                                 self_k_bias,
                                                 self_v_weight,
                                                 self_v_bias,
                                                 self_out_weight,
                                                 self_out_bias,
                                                 ffn_ln_weight,
                                                 ffn_ln_bias,
                                                 ffn_inter_weight,
                                                 ffn_inter_bias,
                                                 ffn_out_weight,
                                                 ffn_out_bias,
                                                 decoder_ln_weight,
                                                 decoder_ln_bias,
                                                 positional_embedding_weight,
                                                 emb_weight,
                                                 output_ids,
                                                 topk,
                                                 topp,
                                                 max_len,
                                                 n_head,
                                                 size_per_head,
                                                 num_layer,
                                                 bos_id,
                                                 eos_id,
                                                 temperature,
                                                 cublas_handle_,
                                                 cublaslt_handle_,
                                                 stream);
  } else {
    ret = gpt2_kernel<paddle::DataType::FLOAT32>(input,
                                                 attn_mask,
                                                 start_length,
                                                 word_embedding,
                                                 self_ln_weight,
                                                 self_ln_bias,
                                                 self_q_weight,
                                                 self_q_bias,
                                                 self_k_weight,
                                                 self_k_bias,
                                                 self_v_weight,
                                                 self_v_bias,
                                                 self_out_weight,
                                                 self_out_bias,
                                                 ffn_ln_weight,
                                                 ffn_ln_bias,
                                                 ffn_inter_weight,
                                                 ffn_inter_bias,
                                                 ffn_out_weight,
                                                 ffn_out_bias,
                                                 decoder_ln_weight,
                                                 decoder_ln_bias,
                                                 positional_embedding_weight,
                                                 emb_weight,
                                                 output_ids,
                                                 topk,
                                                 topp,
                                                 max_len,
                                                 n_head,
                                                 size_per_head,
                                                 num_layer,
                                                 bos_id,
                                                 eos_id,
                                                 temperature,
                                                 cublas_handle_,
                                                 cublaslt_handle_,
                                                 stream);
  }

  cublasDestroy(cublas_handle_);
  cublasLtDestroy(cublaslt_handle_);
  return ret;
}
