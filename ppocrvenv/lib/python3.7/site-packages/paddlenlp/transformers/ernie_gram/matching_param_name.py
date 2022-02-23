import os
import pickle
import paddle
import numpy as np


def match_embedding_param(convert_parameter_name_dict):
    convert_parameter_name_dict[
        "word_emb.weight"] = "embeddings.word_embeddings.weight"
    convert_parameter_name_dict[
        "pos_emb.weight"] = "embeddings.position_embeddings.weight"
    convert_parameter_name_dict[
        "sent_emb.weight"] = "embeddings.token_type_embeddings.weight"
    convert_parameter_name_dict["ln.weight"] = "embeddings.layer_norm.weight"
    convert_parameter_name_dict["ln.bias"] = "embeddings.layer_norm.bias"
    convert_parameter_name_dict[
        "rel_pos_bias_emb.weight"] = "embeddings.rel_pos_embedding.weight"
    return convert_parameter_name_dict


def match_encoder_param(convert_parameter_name_dict, layer_num=4):
    # Firstly, converts the multihead_attention to the parameter.
    proj_names = ["q", "k", "v", "o"]
    param_names = ["weight", "bias"]
    nlp_format = "encoder.layers.{}.self_attn.{}_proj.{}"
    ernie_format = "encoder_stack.block.{}.attn.{}.{}"
    for i in range(0, layer_num):
        for proj_name in proj_names:
            for param_name in param_names:
                if proj_name == "o":
                    nlp_name = nlp_format.format(i, 'out', param_name)
                else:
                    nlp_name = nlp_format.format(i, proj_name, param_name)
                ernie_name = ernie_format.format(i, proj_name, param_name)
                convert_parameter_name_dict[ernie_name] = nlp_name

    # Secondly, converts the encoder ffn parameter.  
    nlp_format = "encoder.layers.{}.linear{}.{}"
    ernie_format = "encoder_stack.block.{}.ffn.{}.{}"
    nlp_param_names = ["1", "2"]
    ernie_param_names = ["i", "o"]
    param_names = ["weight", "bias"]
    for i in range(0, layer_num):
        for nlp_name, ernie_name in zip(nlp_param_names, ernie_param_names):
            for param_name in param_names:
                nlp_format_name = nlp_format.format(i, nlp_name, param_name)
                ernie_format_name = ernie_format.format(i, ernie_name,
                                                        param_name)
                convert_parameter_name_dict[ernie_format_name] = nlp_format_name

    # Thirdly, converts the multi_head layer_norm parameter.
    nlp_format = "encoder.layers.{}.norm{}.{}"
    ernie_format = "encoder_stack.block.{}.ln{}.{}"
    proj_names = ["1", "2"]
    param_names = ["weight", "bias"]
    for i in range(0, layer_num):
        for proj_name in proj_names:
            for param_name in param_names:
                nlp_format_name = nlp_format.format(i, proj_name, param_name)
                ernie_format_name = ernie_format.format(i, proj_name,
                                                        param_name)
                convert_parameter_name_dict[ernie_format_name] = nlp_format_name

    return convert_parameter_name_dict


def match_pooler_parameter(convert_parameter_name_dict):
    convert_parameter_name_dict["pooler.weight"] = "pooler.dense.weight"
    convert_parameter_name_dict["pooler.bias"] = "pooler.dense.bias"
    return convert_parameter_name_dict


def match_mlm_parameter(convert_parameter_name_dict):
    # convert_parameter_name_dict["cls.predictions.decoder_weight"] = "word_embedding"
    convert_parameter_name_dict[
        "cls.predictions.decoder_bias"] = "mask_lm_out_fc.b_0"
    convert_parameter_name_dict[
        "cls.predictions.transform.weight"] = "mask_lm_trans_fc.w_0"
    convert_parameter_name_dict[
        "cls.predictions.transform.bias"] = "mask_lm_trans_fc.b_0"
    convert_parameter_name_dict[
        "cls.predictions.layer_norm.weight"] = "mask_lm_trans_layer_norm_scale"
    convert_parameter_name_dict[
        "cls.predictions.layer_norm.bias"] = "mask_lm_trans_layer_norm_bias"
    return convert_parameter_name_dict


def write_vocab(vocab_file):
    with open(
            vocab_file, 'r', encoding='utf8') as f, open(
                "ernie-gram-zh/new_vocab.txt", 'w', encoding='utf8') as nf:
        for line in f:
            word, word_id = line.strip().split("\t")
            nf.write(word + "\n")


if __name__ == "__main__":
    convert_parameter_name_dict = {}

    convert_parameter_name_dict = match_embedding_param(
        convert_parameter_name_dict)
    convert_parameter_name_dict = match_encoder_param(
        convert_parameter_name_dict, layer_num=12)
    convert_parameter_name_dict = match_pooler_parameter(
        convert_parameter_name_dict)
    ernie_state_dict = paddle.load('./ernie-gram-zh/saved_weights.pdparams')
    nlp_state_dict = {}
    for name, value in ernie_state_dict.items():
        nlp_name = convert_parameter_name_dict[name]
        nlp_state_dict['ernie_gram.' + nlp_name] = value

    paddle.save(nlp_state_dict, './ernie-gram-zh/ernie_gram_zh.pdparams')

    for ernie_name, nlp_name in convert_parameter_name_dict.items():
        print(ernie_name, "          ", nlp_name)
