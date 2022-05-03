# SVTR

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
> [SVTR: Scene Text Recognition with a Single Visual Model](https://arxiv.org/abs/2205.00159)
> Yongkun Du and Zhineng Chen and Caiyan Jia Xiaoting Yin and Tianlun Zheng and Chenxia Li and Yuning Du and Yu-Gang Jiang
> IJCAI, 2022

<a name="model"></a>
The accuracy (%) and model files of SVTR on the public dataset of scene text recognition are as follows:
* Chinese dataset from [Chinese Benckmark](https://arxiv.org/abs/2112.15093) , and the Chinese training evaluation strategy of SVTR follows the paper.

|   Model    |IC13<br/>857 |  SVT  |IIIT5k<br/>3000 |IC15<br/>1811| SVTP  |CUTE80 | Avg_6 |IC15<br/>2077 |IC13<br/>1015 |IC03<br/>867|IC03<br/>860|Avg_10 | Chinese<br/>scene_test|                                                                                            Download link                                                                                            |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|:-----:|:-------:|:-------:|:-----:|:-----:|:---------------------------------------------:|:-----:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| SVTR Tiny  | 96.85  | 91.34 |   94.53   | 83.99  | 85.43 | 89.24 | 90.87 |  80.55  |  95.37  | 95.27 | 95.70 | 90.13 | 67.90 | [English](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_tiny_none_ctc_en_train.tar)  / [Chinese](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_tiny_none_ctc_ch_train.tar)  |
| SVTR Small | 95.92  | 93.04 |   95.03   | 84.70  | 87.91 | 92.01 | 91.63 |  82.72  |  94.88  | 96.08 | 96.28 | 91.02 | 69.00 | [English](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_small_none_ctc_en_train.tar) / [Chinese](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_small_none_ctc_ch_train.tar) |
| SVTR Base  | 97.08  | 91.50 |   96.03   | 85.20  | 89.92 | 91.67 | 92.33 |  83.73  |  95.66  | 95.62 | 95.81 | 91.61 | 71.40 |                          [English](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_base_none_ctc_en_train.tar)  /                                              -                          |
| SVTR Large | 97.20  | 91.65 |   96.30   | 86.58  | 88.37 | 95.14 | 92.82 |  84.54  |  96.35  | 96.54 | 96.74 | 92.24 | 72.10 | [English](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_large_none_ctc_en_train.tar) / [Chinese](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_large_none_ctc_ch_train.tar) |

<a name="2"></a>
## 2. Environment
Please refer to ["Environment Preparation"](./environment.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](./clone.md) to clone the project code.

#### Dataset Preparation

[English dataset download](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here)
[Chinese dataset download](https://github.com/fudanvi/benchmarking-chinese-text-recognition#download)

<a name="3"></a>
## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Tutorial](./recognition.md). PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**.

Training:

Specifically, after the data preparation is completed, the training can be started. The training command is as follows:

```
#Single GPU training (long training period, not recommended)
python3 tools/train.py -c configs/rec/rec_svtrnet.yml

#Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_svtrnet.yml
```

Evaluation:

You can download the model files and configuration files provided by `SVTR`: [download link](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_tiny_none_ctc_en_train.tar), take `SVTR-T` as an example, using the following command to evaluate:

```
# GPU evaluation
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c ./rec_svtr_tiny_none_ctc_en_train/rec_svtr_tiny_6local_6global_stn_en.yml -o Global.pretrained_model=./rec_svtr_tiny_none_ctc_en_train/best_accuracy
```

Prediction:

```
python3 tools/infer_rec.py -c ./rec_svtr_tiny_none_ctc_en_train/rec_svtr_tiny_6local_6global_stn_en.yml -o Global.infer_img='./doc/imgs_words_en/word_10.png' Global.pretrained_model=./rec_svtr_tiny_none_ctc_en_train/best_accuracy
```

<a name="4"></a>
## 4. Inference and Deployment

<a name="4-1"></a>
### 4.1 Python Inference
First, the model saved during the SVTR text recognition training process is converted into an inference model. ( [Model download link](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_tiny_none_ctc_en_train.tar) ), you can use the following command to convert:

```
python3 tools/export_model.py -c configs/rec/rec_svtrnet.yml -o Global.pretrained_model=./rec_svtr_tiny_none_ctc_en_train/best_accuracy  Global.save_inference_dir=./inference/rec_svtr_tiny_stn_en
```

**Note:**
- If you are training the model on your own dataset and have modified the dictionary file, please pay attention to modify the `character_dict_path` in the configuration file to the modified dictionary file.
- If you modified the input size during training, please modify the `infer_shape` corresponding to SVTR in the `tools/export_model.py` file.

After the conversion is successful, there are three files in the directory:
```
/inference/rec_svtr_tiny_stn_en/
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```


For SVTR text recognition model inference, the following commands can be executed:

```
python3 tools/infer/predict_rec.py --image_dir='./doc/imgs_words_en/word_10.png' --rec_model_dir='./inference/rec_svtr_tiny_stn_en/' --rec_algorithm='SVTR' --rec_image_shape='3,64,256' --rec_char_dict_path='./ppocr/utils/ic15_dict.txt'
```

![](../imgs_words_en/word_10.png)

After executing the command, the prediction result (recognized text and score) of the image above is printed to the screen, an example is as follows:
The result is as follows:
```shell
Predicts of ./doc/imgs_words_en/word_10.png:('pain', 0.9999998807907104)
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

1. Since most of the operators used by `SVTR` are matrix multiplication, in the GPU environment, the speed has an advantage, but in the environment where mkldnn is enabled on the CPU, `SVTR` has no advantage over the optimized convolutional network.

## Citation

```bibtex
@article{Du2022SVTR,
  title     = {SVTR: Scene Text Recognition with a Single Visual Model},
  author    = {Du, Yongkun and Chen, Zhineng and Jia, Caiyan and Yin, Xiaoting and Zheng, Tianlun and Li, Chenxia and Du, Yuning and Jiang, Yu-Gang},
  booktitle = {IJCAI},
  year      = {2022},
  url       = {https://arxiv.org/abs/2205.00159}
}
```
