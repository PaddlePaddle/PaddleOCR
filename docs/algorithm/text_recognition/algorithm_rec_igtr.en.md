---
comments: true
---

# IGTR

## 1. Introduction

Paper:
> [Instruction-Guided Scene Text Recognition](https://arxiv.org/abs/2401.17851),
> Yongkun Du, Zhineng Chen, Yuchen Su, Caiyan Jia, Yu-Gang Jiang,
> TPAMI 2025, 
> Source Repository: [OpenOCR](https://github.com/Topdu/OpenOCR)

Multi-modal models have shown appealing performance in visual recognition tasks, as free-form text-guided training evokes the ability to understand fine-grained visual content. However, current models cannot be trivially applied to scene text recognition (STR) due to the compositional difference between natural and text images. We propose a novel instruction-guided scene text recognition (IGTR) paradigm that formulates STR as an instruction learning problem and understands text images by predicting character attributes, e.g., character frequency, position, etc. IGTR first devises $\left \langle condition,question,answer \right \rangle$ instruction triplets, providing rich and diverse descriptions of character attributes. To effectively learn these attributes through question-answering, IGTR develops a lightweight instruction encoder, a cross-modal feature fusion module and a multi-task answer head, which guides nuanced text image understanding. Furthermore, IGTR realizes different recognition pipelines simply by using different instructions, enabling a character-understanding-based text reasoning paradigm that differs from current methods considerably. Experiments on English and Chinese benchmarks show that IGTR outperforms existing models by significant margins, while maintaining a small model size and fast inference speed. Moreover, by adjusting the sampling of instructions, IGTR offers an elegant way to tackle the recognition of rarely appearing and morphologically similar characters, which were previous challenges.

The accuracy (%) and model files of IGTR on the public dataset of scene text recognition are as follows:：

- Trained on Synth dataset(MJ+ST), test on Common Benchmarks, training and test datasets both from [PARSeq](https://github.com/baudm/parseq).

|  Model  | IC13<br/>857 | SVT  | IIIT5k<br/>3000 | IC15<br/>1811 | SVTP | CUTE80 |  Avg  |                                        Config&Model&Log                                         |
| :-----: | :----------: | :--: | :-------------: | :-----------: | :--: | :----: | :---: | :---------------------------------------------------------------------------------------------: |
| IGTR-PD |     97.6     | 95.2 |      97.6       |     88.4      | 91.6 |  95.5  | 94.30 | TODO |
| IGTR-AR |     98.6     | 95.7 |      98.2       |     88.4      | 92.4 |  95.5  | 94.78 |                                            as above                                             |

- Test on Union14M-Benchmark, from [Union14M](https://github.com/Mountchicken/Union14M/).

|  Model  | Curve | Multi-<br/>Oriented | Artistic | Contextless | Salient | Multi-<br/>word | General |  Avg  |    Config&Model&Log     |
| :-----: | :---: | :-----------------: | :------: | :---------: | :-----: | :-------------: | :-----: | :---: | :---------------------: |
| IGTR-PD | 76.9  |        30.6         |   59.1   |    63.3     |  77.8   |      62.5       |  66.7   | 62.40 | Same as the above table |
| IGTR-AR | 78.4  |        31.9         |   61.3   |    66.5     |  80.2   |      69.3       |  67.9   | 65.07 |        as above         |

- Trained on Union14M-L-LMDB-Filtered training dataset.

|    Model     | IC13<br/>857 | SVT  | IIIT5k<br/>3000 | IC15<br/>1811 | SVTP | CUTE80 |  Avg  |                                        Config&Model&Log                                         |
| :----------: | :----------: | :--: | :-------------: | :-----------: | :--: | :----: | :---: | :---------------------------------------------------------------------------------------------: |
|   IGTR-PD    |     97.7     | 97.7 |      98.3       |     89.8      | 93.7 |  97.9  | 95.86 | [PaddleOCR Model](https://paddleocr.bj.bcebos.com/igtr/rec_svtr_igtr_train.tar) |
|   IGTR-AR    |     98.1     | 98.4 |      98.7       |     90.5      | 94.9 |  98.3  | 96.48 |                                            as above                                             |
| IGTR-PD-60ep |     97.9     | 98.3 |      99.2       |     90.8      | 93.7 |  97.6  | 96.24 | TODO|
| IGTR-AR-60ep |     98.4     | 98.1 |      99.3       |     91.5      | 94.3 |  97.6  | 96.54 |                                            as above                                             |
|  IGTR-PD-PT  |     98.6     | 98.0 |      99.1       |     91.7      | 96.8 |  99.0  | 97.20 | TODO |
|  IGTR-AR-PT  |     98.8     | 98.3 |      99.2       |     92.0      | 96.8 |  99.0  | 97.34 |                                            as above                                             |

|    Model     | Curve | Multi-<br/>Oriented | Artistic | Contextless | Salient | Multi-<br/>word | General |  Avg  |    Config&Model&Log     |
| :----------: | :---: | :-----------------: | :------: | :---------: | :-----: | :-------------: | :-----: | :---: | :---------------------: |
|   IGTR-PD    | 88.1  |        89.9         |   74.2   |    80.3     |  82.8   |      79.2       |  83.0   | 82.51 | Same as the above table |
|   IGTR-AR    | 90.4  |        91.2         |   77.0   |    82.4     |  84.7   |      84.0       |  84.4   | 84.86 |        as above         |
| IGTR-PD-60ep | 90.0  |        92.1         |   77.5   |    82.8     |  86.0   |      83.0       |  84.8   | 85.18 | Same as the above table |
| IGTR-AR-60ep | 91.0  |        93.0         |   78.7   |    84.6     |  87.3   |      84.8       |  85.6   | 86.43 |        as above         |
|  IGTR-PD-PT  | 92.4  |        92.1         |   80.7   |    83.6     |  87.7   |      86.9       |  85.0   | 86.92 | Same as the above table |
|  IGTR-AR-PT  | 93.0  |        92.9         |   81.3   |    83.4     |  88.6   |      88.7       |  85.6   | 87.65 |        as above         |

- Trained and test on Chinese dataset, from [Chinese Benckmark](https://github.com/FudanVI/benchmarking-chinese-text-recognition).

|    Model    | Scene | Web  | Document | Handwriting |  Avg  |                                        Config&Model&Log                                         |
| :---------: | :---: | :--: | :------: | :---------: | :---: | :---------------------------------------------------------------------------------------------: |
|   IGTR-PD   | 73.1  | 74.8 |   98.6   |    52.5     | 74.75 |                                                                                                 |
|   IGTR-AR   | 75.1  | 76.4 |   98.7   |    55.3     | 76.37 |                                                                                                 |
| IGTR-PD-TS  | 73.5  | 75.9 |   98.7   |    54.5     | 75.65 | TODO |
| IGTR-AR-TS  | 75.6  | 77.0 |   98.8   |    57.3     | 77.17 |                                            as above                                             |
| IGTR-PD-Aug | 79.5  | 80.0 |   99.4   |    58.9     | 79.45 | TODO |
| IGTR-AR-Aug | 82.0  | 81.7 |   99.5   |    63.8     | 81.74 |                                            as above                                             |

Download all Configs, Models, and Logs from [OpenOCR](https://github.com/Topdu/OpenOCR/blob/main/configs/rec/igtr/readme.md), and then convert to paddleocr model file.

## 2. Environment

Please refer to ["Environment Preparation"](../../ppocr/environment.en.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](../../ppocr/blog/clone.en.md)to clone the project code.

### Dataset Preparation

- [English dataset download](https://github.com/baudm/parseq)

- [Union14M-L-LMDB-Filtered download](https://github.com/Topdu/OpenOCR/blob/main/docs/svtrv2.md#downloading-datasets)

- [Chinese dataset download](https://github.com/fudanvi/benchmarking-chinese-text-recognition#download)

## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Tutorial](../../ppocr/model_train/recognition.en.md). PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**.

### Training

Specifically, after the data preparation is completed, the training can be started. The training command is as follows:

```bash linenums="1"
# Single GPU training (long training period, not recommended)
python3 tools/train.py -c configs/rec/rec_svtrnet_igtr.yml

# Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_svtrnet_igtr.yml
```

### Evaluation

You can download the model files and configuration files provided by `IGTR`: [download link](https://paddleocr.bj.bcebos.com/igtr/rec_svtr_igtr_train.tar), using the following command to evaluate:

```bash linenums="1"
# Download the tar archive containing the model files and configuration files of IGTR-B and extract it
wget https://paddleocr.bj.bcebos.com/igtr/rec_svtr_igtr_train.tar && tar xf rec_svtr_igtr_train.tar
# GPU evaluation
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_svtrnet_igtr.yml -o Global.pretrained_model=./rec_svtr_igtr_train/best_model
```

### Prediction

```bash linenums="1"
python3 tools/infer_rec.py -c configs/rec/rec_svtrnet_igtr.yml -o Global.infer_img='./doc/imgs_words/word_10.png' Global.pretrained_model=./rec_svtr_igtr_train/best_model
```

## 4. Inference and Deployment

### 4.1 Python Inference

Coming soon.

### 4.2 C++ Inference

Not supported

### 4.3 Serving

Not supported

### 4.4 More

Not supported

## Citation

```bibtex
@article{Du2025IGTR,
  title     = {Instruction-Guided Scene Text Recognition},
  author    = {Du, Yongkun and Chen, Zhineng and Su, Yuchen and Jia, Caiyan and Jiang, Yu-Gang},
  journal   = {IEEE Trans. Pattern Anal. Mach. Intell.},
  year      = {2025},
  url       = {https://arxiv.org/abs/2401.17851}
}
```
