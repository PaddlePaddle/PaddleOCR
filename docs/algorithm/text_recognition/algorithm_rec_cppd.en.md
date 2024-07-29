---
comments: true
---

# CPPD

## 1. Introduction

Paper:
> [Context Perception Parallel Decoder for Scene Text Recognition](https://arxiv.org/abs/2307.12270)
> Yongkun Du and Zhineng Chen and Caiyan Jia and Xiaoting Yin and Chenxia Li and Yuning Du and Yu-Gang Jiang

Scene text recognition models based on deep learning typically follow an Encoder-Decoder structure, where the decoder can be categorized into two types: (1) CTC and (2) Attention-based. Currently, most state-of-the-art (SOTA) models use an Attention-based decoder, which can be further divided into AR and PD types. In general, AR decoders achieve higher recognition accuracy than PD, while PD decoders are faster than AR. CPPD, with carefully designed CO and CC modules, achieves a balance between the accuracy of AR and the speed of PD.

The accuracy (%) and model files of CPPD on the public dataset of scene text recognition are as follows:：

* English dataset from [PARSeq](https://github.com/baudm/parseq).

|    Model      |IC13<br/>857 |  SVT  |IIIT5k<br/>3000 |IC15<br/>1811| SVTP  |CUTE80 | Avg |      Download       |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|:-----:|:-------:|
| CPPD Tiny  | 97.1  | 94.4 |   96.6   | 86.6  | 88.5 | 90.3 | 92.25 | [en](https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_tiny_en_train.tar) |
| CPPD Base | 98.2  | 95.5 |   97.6   | 87.9  | 90.0 | 92.7 | 93.80 | [en](https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_en_train.tar)|
| CPPD Base 48*160  | 97.5  | 95.5 |   97.7   | 87.7  | 92.4 | 93.7 | 94.10 | [en](https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_48_160_en_train.tar) |

* Trained on Synth dataset(MJ+ST), Test on Union14M-L benchmark from [U14m](https://github.com/Mountchicken/Union14M/).

|    Model      |Curve |  Multi-<br/>Oriented  |Artistic |Contextless| Salient  | Multi-<br/>word | General | Avg |     Download       |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|:-----:|:-------:|:-------:|
| CPPD Tiny  | 52.4  | 12.3 |   48.2   | 54.4  | 61.5 | 53.4 | 61.4 | 49.10 | Same as the table above. |
| CPPD Base | 65.5  | 18.6 |   56.0   | 61.9  | 71.0 | 57.5 | 65.8 | 56.63 | Same as the table above. |
| CPPD Base 48*160  | 71.9  | 22.1 |   60.5   | 67.9  | 78.3 | 63.9 | 67.1 | 61.69 | Same as the table above. |

* Trained on Union14M-L training dataset.

|    Model      |IC13<br/>857 |  SVT  |IIIT5k<br/>3000 |IC15<br/>1811| SVTP  |CUTE80 | Avg |      Download       |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|:-----:|:-------:|
| CPPD Base 32*128  | 98.7  | 98.5 |   99.4   | 91.7  | 96.7 | 99.7 | 97.44 | [en](https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_u14m_train.tar) |

|    Model      |Curve |  Multi-<br/>Oriented  |Artistic |Contextless| Salient  | Multi-<br/>word | General | Avg |     Download       |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|:-----:|:-------:|:-------:|
| CPPD Base 32*128  | 87.5  | 70.7 |   78.2   | 82.9  | 85.5 | 85.4 | 84.3 | 82.08 | Same as the table above. |

* Chinese dataset from [Chinese Benckmark](https://github.com/FudanVI/benchmarking-chinese-text-recognition).

|    Model      | Scene | Web | Document | Handwriting | Avg |      Download       |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|
| CPPD Base  | 74.4  | 76.1 |   98.6   | 55.3  | 76.10 | [ch](https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_ch_train.tar)  |
| CPPD Base + STN | 78.4  | 79.3 |   98.9   | 57.6  | 78.55 | [ch](https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_stn_ch_train.tar) |

## 2. Environment

Please refer to ["Environment Preparation"](../../ppocr/environment.en.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](../../ppocr/blog/clone.en.md)to clone the project code.

### Dataset Preparation

[English dataset download](https://github.com/baudm/parseq)
[Union14M-Benchmark download](https://github.com/Mountchicken/Union14M)
[Chinese dataset download](https://github.com/fudanvi/benchmarking-chinese-text-recognition#download)

## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Tutorial](../../ppocr/model_train/recognition.en.md). PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**.

### Training

Specifically, after the data preparation is completed, the training can be started. The training command is as follows:

```bash linenums="1"
# Single GPU training (long training period, not recommended)
python3 tools/train.py -c configs/rec/rec_svtrnet_cppd_base_en.yml

# Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_svtrnet_cppd_base_en.yml
```

### Evaluation

You can download the model files and configuration files provided by `CPPD`: [download link](https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_en_train.tar), take `CPPD-B` as an example, using the following command to evaluate:

```bash linenums="1"
# Download the tar archive containing the model files and configuration files of CPPD-B and extract it
wget https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_en_train.tar && tar xf rec_svtr_cppd_base_en_train.tar
# GPU evaluation
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c ./rec_svtr_cppd_base_en_train/rec_svtrnet_cppd_base_en.yml -o Global.pretrained_model=./rec_svtr_cppd_base_en_train/best_model
```

### Prediction

```bash linenums="1"
python3 tools/infer_rec.py -c ./rec_svtr_cppd_base_en_train/rec_svtrnet_cppd_base_en.yml -o Global.infer_img='./doc/imgs_words_en/word_10.png' Global.pretrained_model=./rec_svtr_cppd_base_en_train/best_model
```

## 4. Inference and Deployment

### 4.1 Python Inference

First, the model saved during the CPPD text recognition training process is converted into an inference model. ( [Model download link](https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_en_train.tar) ), you can use the following command to convert:

```bash linenums="1"
# export model
# en
python3 tools/export_model.py -c configs/rec/rec_svtrnet_cppd_base_en.yml -o Global.pretrained_model=./rec_svtr_cppd_base_en_train/best_model.pdparams Global.save_inference_dir=./rec_svtr_cppd_base_en_infer
# ch
python3 tools/export_model.py -c configs/rec/rec_svtrnet_cppd_base_ch.yml -o Global.pretrained_model=./rec_svtr_cppd_base_ch_train/best_model.pdparams Global.save_inference_dir=./rec_svtr_cppd_base_ch_infer

# speed test
# docker image https://hub.docker.com/r/paddlepaddle/paddle/tags/: sudo docker pull paddlepaddle/paddle:2.4.2-gpu-cuda11.2-cudnn8.2-trt8.0
# install auto_log: pip install https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
# en
python3 tools/infer/predict_rec.py --image_dir='../iiik' --rec_model_dir='./rec_svtr_cppd_base_en_infer/' --rec_algorithm='CPPD' --rec_image_shape='3,32,100' --rec_char_dict_path='./ppocr/utils/ic15_dict.txt' --warmup=True --benchmark=True --rec_batch_num=1 --use_tensorrt=True
# ch
python3 tools/infer/predict_rec.py --image_dir='../iiik' --rec_model_dir='./rec_svtr_cppd_base_ch_infer/' --rec_algorithm='CPPDPadding' --rec_image_shape='3,32,256' --warmup=True --benchmark=True --rec_batch_num=1 --use_tensorrt=True
# stn_ch
python3 tools/infer/predict_rec.py --image_dir='../iiik' --rec_model_dir='./rec_svtr_cppd_base_stn_ch_infer/' --rec_algorithm='CPPD' --rec_image_shape='3,64,256' --warmup=True --benchmark=True --rec_batch_num=1 --use_tensorrt=True
```

**Note:** If you are training the model on your own dataset and have modified the dictionary file, please pay attention to modify the `character_dict_path` in the configuration file to the modified dictionary file.

After the conversion is successful, there are three files in the directory:

```text linenums="1"
/inference/rec_svtr_cppd_base_en_infer/
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```

### 4.2 C++ Inference

Not supported

### 4.3 Serving

Not supported

### 4.4 More

Not supported

## Citation

```bibtex
@article{Du2023CPPD,
  title     = {Context Perception Parallel Decoder for Scene Text Recognition},
  author    = {Du, Yongkun and Chen, Zhineng and Jia, Caiyan and Yin, Xiaoting and Li, Chenxia and Du, Yuning and Jiang, Yu-Gang},
  booktitle = {Arxiv},
  year      = {2023},
  url       = {https://arxiv.org/abs/2307.12270}
}
```
