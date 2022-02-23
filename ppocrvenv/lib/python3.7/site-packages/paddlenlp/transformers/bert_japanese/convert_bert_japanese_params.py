import paddle
import torch
import numpy as np
from paddle.utils.download import get_path_from_url

model_names = [
    "bert-base-japanese", "bert-base-japanese-whole-word-masking",
    "bert-base-japanese-char", "bert-base-japanese-char-whole-word-masking"
]

for model_name in model_names:
    torch_model_url = "https://huggingface.co/cl-tohoku/%s/resolve/main/pytorch_model.bin" % model_name
    torch_model_path = get_path_from_url(torch_model_url, '../bert')
    torch_state_dict = torch.load(torch_model_path)

    paddle_model_path = "%s.pdparams" % model_name
    paddle_state_dict = {}

    # State_dict's keys mapping: from torch to paddle
    keys_dict = {
        # about embeddings
        "embeddings.LayerNorm.gamma": "embeddings.layer_norm.weight",
        "embeddings.LayerNorm.beta": "embeddings.layer_norm.bias",

        # about encoder layer
        'encoder.layer': 'encoder.layers',
        'attention.self.query': 'self_attn.q_proj',
        'attention.self.key': 'self_attn.k_proj',
        'attention.self.value': 'self_attn.v_proj',
        'attention.output.dense': 'self_attn.out_proj',
        'attention.output.LayerNorm.gamma': 'norm1.weight',
        'attention.output.LayerNorm.beta': 'norm1.bias',
        'intermediate.dense': 'linear1',
        'output.dense': 'linear2',
        'output.LayerNorm.gamma': 'norm2.weight',
        'output.LayerNorm.beta': 'norm2.bias',

        # about cls predictions
        'cls.predictions.transform.dense': 'cls.predictions.transform',
        'cls.predictions.decoder.weight': 'cls.predictions.decoder_weight',
        'cls.predictions.transform.LayerNorm.gamma':
        'cls.predictions.layer_norm.weight',
        'cls.predictions.transform.LayerNorm.beta':
        'cls.predictions.layer_norm.bias',
        'cls.predictions.bias': 'cls.predictions.decoder_bias'
    }

    for torch_key in torch_state_dict:
        paddle_key = torch_key
        for k in keys_dict:
            if k in paddle_key:
                paddle_key = paddle_key.replace(k, keys_dict[k])

        if ('linear' in paddle_key) or ('proj' in paddle_key) or (
                'vocab' in paddle_key and 'weight' in paddle_key) or (
                    "dense.weight" in paddle_key) or (
                        'transform.weight' in paddle_key) or (
                            'seq_relationship.weight' in paddle_key):
            paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[
                torch_key].cpu().numpy().transpose())
        else:
            paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[
                torch_key].cpu().numpy())

        print("torch: ", torch_key, "\t", torch_state_dict[torch_key].shape)
        print("paddle: ", paddle_key, "\t", paddle_state_dict[paddle_key].shape,
              "\n")

    paddle.save(paddle_state_dict, paddle_model_path)
