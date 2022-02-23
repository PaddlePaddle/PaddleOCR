#include <string>
#include <vector>

#include "fusion_gpt_op.h"
#include "pd_traits.h"


std::vector<paddle::Tensor> GPT2Forward(
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
  int batch_size = input.shape()[0];
  int start_len = input.shape()[1];
  int total_len = max_len + start_len;
  std::vector<int64_t> output_dims({total_len, batch_size});
  auto output_ids = paddle::Tensor(input.place(), output_dims);

  if (word_embedding.place() == paddle::PlaceType::kGPU) {
    return GPT2CUDAForward(input,
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
                           total_len,
                           n_head,
                           size_per_head,
                           num_layer,
                           bos_id,
                           eos_id,
                           temperature,
                           use_fp16);
  } else {
    PD_THROW("Not implemented place. Only GPU is supported. ");
  }
}

std::vector<std::vector<int64_t>> GPT2InferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& attn_mask_shape,
    const std::vector<int64_t>& start_length,
    const std::vector<int64_t>& word_embedding_shape,
    const std::vector<std::vector<int64_t>>& self_ln_weight_shapes,
    const std::vector<std::vector<int64_t>>& self_ln_bias_shapes,
    const std::vector<std::vector<int64_t>>& self_q_weight_shapes,
    const std::vector<std::vector<int64_t>>& self_q_bias_shapes,
    const std::vector<std::vector<int64_t>>& self_k_weight_shapes,
    const std::vector<std::vector<int64_t>>& self_k_bias_shapes,
    const std::vector<std::vector<int64_t>>& self_v_weight_shapes,
    const std::vector<std::vector<int64_t>>& self_v_bias_shapes,
    const std::vector<std::vector<int64_t>>& self_out_weight_shapes,
    const std::vector<std::vector<int64_t>>& self_out_bias_shapes,
    const std::vector<std::vector<int64_t>>& ffn_ln_weight_shapes,
    const std::vector<std::vector<int64_t>>& ffn_ln_bias_shapes,
    const std::vector<std::vector<int64_t>>& ffn_inter_weight_shapes,
    const std::vector<std::vector<int64_t>>& ffn_inter_bias_shapes,
    const std::vector<std::vector<int64_t>>& ffn_out_weight_shapes,
    const std::vector<std::vector<int64_t>>& ffn_out_bias_shapes,
    const std::vector<int64_t>& decoder_ln_weight_shape,
    const std::vector<int64_t>& decoder_ln_bias_shape,
    const std::vector<int64_t>& positional_embedding_weight_shape,
    const std::vector<int64_t>& emb_weight_shape,
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
  int64_t batch_size = input_shape[0];
  int64_t start_len = input_shape[1];
  std::vector<int64_t> output_dims({max_len + start_len, batch_size});
  return {output_dims};
}

std::vector<paddle::DataType> GPT2InferDtype(
    const paddle::DataType& input_dtype,
    const paddle::DataType& attn_mask_dtype,
    const paddle::DataType& start_length_dtype,
    const paddle::DataType& word_embedding_dtype,
    const std::vector<paddle::DataType>& self_ln_weight_dtype,
    const std::vector<paddle::DataType>& self_ln_bias_dtype,
    const std::vector<paddle::DataType>& self_q_weight_dtype,
    const std::vector<paddle::DataType>& self_q_bias_dtype,
    const std::vector<paddle::DataType>& self_k_weight_dtype,
    const std::vector<paddle::DataType>& self_k_bias_dtype,
    const std::vector<paddle::DataType>& self_v_weight_dtype,
    const std::vector<paddle::DataType>& self_v_bias_dtype,
    const std::vector<paddle::DataType>& self_out_weight_dtype,
    const std::vector<paddle::DataType>& self_out_bias_dtype,
    const std::vector<paddle::DataType>& ffn_ln_weight_dtype,
    const std::vector<paddle::DataType>& ffn_ln_bias_dtype,
    const std::vector<paddle::DataType>& ffn_inter_weight_dtype,
    const std::vector<paddle::DataType>& ffn_inter_bias_dtype,
    const std::vector<paddle::DataType>& ffn_out_weight_dtype,
    const std::vector<paddle::DataType>& ffn_out_bias_dtype,
    const paddle::DataType& decoder_ln_weight_dtype,
    const paddle::DataType& decoder_ln_bias_dtype,
    const paddle::DataType& positional_embedding_weight_dtype,
    const paddle::DataType& emb_weight_dtype) {
  return {paddle::DataType::INT32};
}

PD_BUILD_OP(fusion_gpt)
    .Inputs({"Input",
             "AttentionMask",
             "StartLength",
             "WordEmbedding",
             paddle::Vec("SelfLayernormWeight"),
             paddle::Vec("SelfLayernormBias"),
             paddle::Vec("SelfQueryWeight"),
             paddle::Vec("SelfQueryBias"),
             paddle::Vec("SelfKeyWeight"),
             paddle::Vec("SelfKeyBias"),
             paddle::Vec("SelfValueWeight"),
             paddle::Vec("SelfValueBias"),
             paddle::Vec("SelfOutWeight"),
             paddle::Vec("SelfOutBias"),
             paddle::Vec("FFNLayernormWeight"),
             paddle::Vec("FFNLayernormBias"),
             paddle::Vec("FFNInterWeight"),
             paddle::Vec("FFNInterBias"),
             paddle::Vec("FFNOutWeight"),
             paddle::Vec("FFNOutBias"),
             "DecoderLayernormWeight",
             "DecoderLayernormBias",
             "PositionEncEmb",
             "EmbWeight"})
    .Outputs({"OutputIds"})
    .Attrs({"topk: int",
            "topp: float",
            "max_len: int",
            "n_head: int",
            "size_per_head: int",
            "num_layer: int",
            "bos_id: int",
            "eos_id: int",
            "temperature: float",
            "use_fp16: bool"})
    .SetKernelFn(PD_KERNEL(GPT2Forward))
    .SetInferShapeFn(PD_INFER_SHAPE(GPT2InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GPT2InferDtype));
