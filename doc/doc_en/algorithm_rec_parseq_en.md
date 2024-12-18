# PasreQ

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

<a name="1"></a>
## 1. Introduction

Paper:
> [Scene Text Recognition with Permuted Autoregressive Sequence Models](https://arxiv.org/abs/2207.06966)
> Darwin Bautista, Rowel Atienza
> ECCV, 2021

Using real datasets (real) and synthetic datsets (synth) for training respectively，and evaluating on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE datasets. 
- The real datasets include COCO-Text, RCTW17, Uber-Text, ArT, LSVT, MLT19, ReCTS, TextOCR and OpenVINO datasets.
- The synthesis datasets include MJSynth and SynthText datasets.

the algorithm reproduction effect is as follows:

|Training Dataset|Model|Backbone|config|Acc|Download link|
| --- | --- | --- | --- | --- | --- |
|Synth|ParseQ|VIT|[rec_vit_parseq.yml](../../configs/rec/rec_vit_parseq.yml)|91.24%|[train model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/parseq/rec_vit_parseq_synth.tgz)|
|Real|ParseQ|VIT|[rec_vit_parseq.yml](../../configs/rec/rec_vit_parseq.yml)|94.74%|[train model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/parseq/rec_vit_parseq_real.tgz)|
|||||||

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
python3 tools/train.py -c configs/rec/rec_vit_parseq.yml

#Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_vit_parseq.yml
```

Evaluation:

```
# GPU evaluation
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_vit_parseq.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

Prediction:

```
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_vit_parseq.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
```

<a name="4"></a>
## 4. Inference and Deployment

<a name="4-1"></a>
### 4.1 Python Inference
First, the model saved during the SAR text recognition training process is converted into an inference model. ( [Model download link](https://paddleocr.bj.bcebos.com/dygraph_v2.1/parseq/rec_vit_parseq_real.tgz) ), you can use the following command to convert:

```
python3 tools/export_model.py -c configs/rec/rec_vit_parseq.yml -o Global.pretrained_model=./rec_vit_parseq_real/best_accuracy Global.save_inference_dir=./inference/rec_parseq
```

For SAR text recognition model inference, the following commands can be executed:

```
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words/en/word_1.png" --rec_model_dir="./inference/rec_parseq/" --rec_image_shape="3, 32, 128" --rec_algorithm="ParseQ" --rec_char_dict_path="ppocr/utils/dict/parseq_dict.txt" --max_text_length=25 --use_space_char=False
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


## Citation

```bibtex
@InProceedings{bautista2022parseq,
  title={Scene Text Recognition with Permuted Autoregressive Sequence Models},
  author={Bautista, Darwin and Atienza, Rowel},
  booktitle={European Conference on Computer Vision},
  pages={178--196},
  month={10},
  year={2022},
  publisher={Springer Nature Switzerland},
  address={Cham},
  doi={10.1007/978-3-031-19815-1_11},
  url={https://doi.org/10.1007/978-3-031-19815-1_11}
}
```
