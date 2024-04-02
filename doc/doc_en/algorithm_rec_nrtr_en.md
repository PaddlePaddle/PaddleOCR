# NRTR

- [1. Introduction](#1)
- [2. Environment](#2)
- [3. Model Training / Evaluation / Prediction](#3)
    - [3.1 Training](#3-1)
    - [3.2 Evaluation](#3-2)
    - [3.3 Prediction](#3-3)
- [4. Inference and Deployment](#4)
    - [4.1 Python Inference](#4-1)
    - [4.2 C++ Inference](#4-2)
    - [4.3 Serving](#4-3)
    - [4.4 More](#4-4)
- [5. FAQ](#5)
- [6. Release Note](#6)

<a name="1"></a>
## 1. Introduction

Paper:
> [NRTR: A No-Recurrence Sequence-to-Sequence Model For Scene Text Recognition](https://arxiv.org/abs/1806.00926)
> Fenfen Sheng and Zhineng Chen and Bo Xu
> ICDAR, 2019

Using MJSynth and SynthText two text recognition datasets for training, and evaluating on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE datasets, the algorithm reproduction effect is as follows:

|Model|Backbone|config|Acc|Download link|
| --- | --- | --- | --- | --- |
|NRTR|MTB|[rec_mtb_nrtr.yml](../../configs/rec/rec_mtb_nrtr.yml)|84.21%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mtb_nrtr_train.tar)|

<a name="2"></a>
## 2. Environment
Please refer to ["Environment Preparation"](./environment_en.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](./clone_en.md) to clone the project code.


<a name="3"></a>
## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Tutorial](./recognition_en.md). PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**.

Training:

Specifically, after the data preparation is completed, the training can be started. The training command is as follows:

```
#Single GPU training (long training period, not recommended)
python3 tools/train.py -c configs/rec/rec_mtb_nrtr.yml

#Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_mtb_nrtr.yml
```

Evaluation:

```
# GPU evaluation
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_mtb_nrtr.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

Prediction:

```
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_mtb_nrtr.yml -o Global.infer_img='./doc/imgs_words_en/word_10.png' Global.pretrained_model=./rec_mtb_nrtr_train/best_accuracy
```

<a name="4"></a>
## 4. Inference and Deployment

<a name="4-1"></a>
### 4.1 Python Inference
First, the model saved during the NRTR text recognition training process is converted into an inference model. ( [Model download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mtb_nrtr_train.tar)) ), you can use the following command to convert:

```
python3 tools/export_model.py -c configs/rec/rec_mtb_nrtr.yml -o Global.pretrained_model=./rec_mtb_nrtr_train/best_accuracy  Global.save_inference_dir=./inference/rec_mtb_nrtr
```

**Note:**
- If you are training the model on your own dataset and have modified the dictionary file, please pay attention to modify the `character_dict_path` in the configuration file to the modified dictionary file.
- If you modified the input size during training, please modify the `infer_shape` corresponding to NRTR in the `tools/export_model.py` file.

After the conversion is successful, there are three files in the directory:
```
/inference/rec_mtb_nrtr/
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```


For NRTR text recognition model inference, the following commands can be executed:

```
python3 tools/infer/predict_rec.py --image_dir='./doc/imgs_words_en/word_10.png' --rec_model_dir='./inference/rec_mtb_nrtr/' --rec_algorithm='NRTR' --rec_image_shape='1,32,100' --rec_char_dict_path='./ppocr/utils/EN_symbol_dict.txt'
```

![](../imgs_words_en/word_10.png)

After executing the command, the prediction result (recognized text and score) of the image above is printed to the screen, an example is as follows:
The result is as follows:
```shell
Predicts of ./doc/imgs_words_en/word_10.png:('pain', 0.9465042352676392)
```

<a name="4-2"></a>
### 4.2 C++ Inference

Not supported

<a name="4-3"></a>
### 4.3 Serving

Not supported

<a name="4-4"></a>
### 4.4 More

Not supported

<a name="5"></a>
## 5. FAQ

1. In the `NRTR` paper, Beam search is used to decode characters, but the speed is slow. Beam search is not used by default here, and greedy search is used to decode characters.

<a name="6"></a>
## 6. Release Note

1. The release/2.6 version updates the NRTR code structure. The new version of NRTR can load the model parameters of the old version (release/2.5 and before), and you may use the following code to convert the old version model parameters to the new version model parameters:

```python

    params = paddle.load('path/' + '.pdparams') # the old version parameters
    state_dict = model.state_dict() # the new version model parameters
    new_state_dict = {}

    for k1, v1 in state_dict.items():

        k = k1
        if 'encoder' in k and 'self_attn' in k and 'qkv' in k and 'weight' in k:

            k_para = k[:13] + 'layers.' + k[13:]
            q = params[k_para.replace('qkv', 'conv1')].transpose((1, 0, 2, 3))
            k = params[k_para.replace('qkv', 'conv2')].transpose((1, 0, 2, 3))
            v = params[k_para.replace('qkv', 'conv3')].transpose((1, 0, 2, 3))

            new_state_dict[k1] = np.concatenate([q[:, :, 0, 0], k[:, :, 0, 0], v[:, :, 0, 0]], -1)

        elif 'encoder' in k and 'self_attn' in k and 'qkv' in k and 'bias' in k:

            k_para = k[:13] + 'layers.' + k[13:]
            q = params[k_para.replace('qkv', 'conv1')]
            k = params[k_para.replace('qkv', 'conv2')]
            v = params[k_para.replace('qkv', 'conv3')]

            new_state_dict[k1] = np.concatenate([q, k, v], -1)

        elif 'encoder' in k and 'self_attn' in k and 'out_proj' in k:

            k_para = k[:13] + 'layers.' + k[13:]
            new_state_dict[k1] = params[k_para]

        elif 'encoder' in k and 'norm3' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            new_state_dict[k1] = params[k_para.replace('norm3', 'norm2')]

        elif 'encoder' in k and 'norm1' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            new_state_dict[k1] = params[k_para]


        elif 'decoder' in k and 'self_attn' in k and 'qkv' in k and 'weight' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            q = params[k_para.replace('qkv', 'conv1')].transpose((1, 0, 2, 3))
            k = params[k_para.replace('qkv', 'conv2')].transpose((1, 0, 2, 3))
            v = params[k_para.replace('qkv', 'conv3')].transpose((1, 0, 2, 3))
            new_state_dict[k1] = np.concatenate([q[:, :, 0, 0], k[:, :, 0, 0], v[:, :, 0, 0]], -1)

        elif 'decoder' in k and 'self_attn' in k and 'qkv' in k and 'bias' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            q = params[k_para.replace('qkv', 'conv1')]
            k = params[k_para.replace('qkv', 'conv2')]
            v = params[k_para.replace('qkv', 'conv3')]
            new_state_dict[k1] = np.concatenate([q, k, v], -1)

        elif 'decoder' in k and 'self_attn' in k and 'out_proj' in k:

            k_para = k[:13] + 'layers.' + k[13:]
            new_state_dict[k1] = params[k_para]

        elif 'decoder' in k and 'cross_attn' in k and 'q' in k and 'weight' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            k_para = k_para.replace('cross_attn', 'multihead_attn')
            q = params[k_para.replace('q', 'conv1')].transpose((1, 0, 2, 3))
            new_state_dict[k1] = q[:, :, 0, 0]

        elif 'decoder' in k and 'cross_attn' in k and 'q' in k and 'bias' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            k_para = k_para.replace('cross_attn', 'multihead_attn')
            q = params[k_para.replace('q', 'conv1')]
            new_state_dict[k1] = q

        elif 'decoder' in k and 'cross_attn' in k and 'kv' in k and 'weight' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            k_para = k_para.replace('cross_attn', 'multihead_attn')
            k = params[k_para.replace('kv', 'conv2')].transpose((1, 0, 2, 3))
            v = params[k_para.replace('kv', 'conv3')].transpose((1, 0, 2, 3))
            new_state_dict[k1] = np.concatenate([k[:, :, 0, 0], v[:, :, 0, 0]], -1)

        elif 'decoder' in k and 'cross_attn' in k and 'kv' in k and 'bias' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            k_para = k_para.replace('cross_attn', 'multihead_attn')
            k = params[k_para.replace('kv', 'conv2')]
            v = params[k_para.replace('kv', 'conv3')]
            new_state_dict[k1] = np.concatenate([k, v], -1)

        elif 'decoder' in k and 'cross_attn' in k and 'out_proj' in k:

            k_para = k[:13] + 'layers.' + k[13:]
            k_para = k_para.replace('cross_attn', 'multihead_attn')
            new_state_dict[k1] = params[k_para]
        elif 'decoder' in k and 'norm' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            new_state_dict[k1] = params[k_para]
        elif 'mlp' in k and 'weight' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            k_para = k_para.replace('fc', 'conv')
            k_para = k_para.replace('mlp.', '')
            w = params[k_para].transpose((1, 0, 2, 3))
            new_state_dict[k1] = w[:, :, 0, 0]
        elif 'mlp' in k and 'bias' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            k_para = k_para.replace('fc', 'conv')
            k_para = k_para.replace('mlp.', '')
            w = params[k_para]
            new_state_dict[k1] = w

        else:
            new_state_dict[k1] = params[k1]

        if list(new_state_dict[k1].shape) != list(v1.shape):
            print(k1)


    for k, v1 in state_dict.items():
        if k not in new_state_dict.keys():
            print(1, k)
        elif list(new_state_dict[k].shape) != list(v1.shape):
            print(2, k)



    model.set_state_dict(new_state_dict)
    paddle.save(model.state_dict(), 'nrtrnew_from_old_params.pdparams')

```

2. The new version has a clean code structure and improved inference speed compared with the old version.

## Citation

```bibtex
@article{Sheng2019NRTR,
  title     = {NRTR: A No-Recurrence Sequence-to-Sequence Model For Scene Text Recognition},
  author    = {Fenfen Sheng and Zhineng Chen and Bo Xu},
  booktitle = {ICDAR},
  year      = {2019},
  url       = {http://arxiv.org/abs/1806.00926},
  pages     = {781-786}
}
```
