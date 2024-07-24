# LaTeX-OCR

- [1. Introduction](#1)
- [2. Environment](#2)
- [3. Model Training / Evaluation / Prediction](#3)
    - [3.1 Pickle File Generation](#3-1)
    - [3.2 Training](#3-2)
    - [3.3 Evaluation](#3-3)
    - [3.4 Prediction](#3-4)
- [4. Inference and Deployment](#4)
    - [4.1 Python Inference](#4-1)
    - [4.2 C++ Inference](#4-2)
    - [4.3 Serving](#4-3)
    - [4.4 More](#4-4)
- [5. FAQ](#5)

<a name="1"></a>
## 1. Introduction

Original Project:
> [https://github.com/lukas-blecher/LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR)


Using LaTeX-OCR printed mathematical expression recognition datasets for training, and evaluating on its test sets, the algorithm reproduction effect is as follows:

| Model       | Backbone |config| BLEU score  | normed edit distance  |  ExpRate  |Download link|
|-----------|----------| ---- |:-----------:|:---------------------:|:---------:| ----- |
| LaTeX-OCR | Hybrid ViT |[rec_latex_ocr.yml](../../configs/rec/rec_latex_ocr.yml)|   0.8821    |        0.0823         |  40.01%   |[trained model](https://paddleocr.bj.bcebos.com/contribution/rec_latex_ocr_train.tar)|

<a name="2"></a>
## 2. Environment
Please refer to ["Environment Preparation"](./environment_en.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](./clone_en.md) to clone the project code.

Furthermore, additional dependencies need to be installed:
```shell
pip install "tokenizers==0.19.1" "imagesize"
```

<a name="3"></a>
## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Tutorial](./recognition_en.md). PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**.

Pickle File Generation:

Download formulae.zip and math.txt in [Google Drive](https://drive.google.com/drive/folders/13CA4vAmOmD_I_dSbvLp-Lf0s6KiaNfuO), and then use the following command to generate the pickle file.

```shell
# Create a LaTeX-OCR dataset directory
mkdir -p train_data/LaTeXOCR
# Unzip formulae.zip and copy math.txt
unzip -d train_data/LaTeXOCR path/formulae.zip
cp path/math.txt train_data/LaTeXOCR
# Convert the original .txt file to a .pkl file to group images of different scales
# Training set conversion
python ppocr/utils/formula_utils/math_txt2pkl.py --image_dir=train_data/LaTeXOCR/train --mathtxt_path=train_data/LaTeXOCR/math.txt --output_dir=train_data/LaTeXOCR/
# Validation set conversion
python ppocr/utils/formula_utils/math_txt2pkl.py --image_dir=train_data/LaTeXOCR/val --mathtxt_path=train_data/LaTeXOCR/math.txt --output_dir=train_data/LaTeXOCR/
# Test set conversion
python ppocr/utils/formula_utils/math_txt2pkl.py --image_dir=train_data/LaTeXOCR/test --mathtxt_path=train_data/LaTeXOCR/math.txt --output_dir=train_data/LaTeXOCR/
```


Training:

Specifically, after the data preparation is completed, the training can be started. The training command is as follows:

```
#Single GPU training (Default training method)
python3 tools/train.py -c configs/rec/rec_latex_ocr.yml

#Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_latex_ocr.yml
```

Evaluation:

```
# GPU evaluation
# Validation set evaluation
python3 tools/eval.py -c configs/rec/rec_latex_ocr.yml -o Global.pretrained_model=./rec_latex_ocr_train/best_accuracy.pdparams Metric.cal_blue_score=True
# Test set evaluation
python3 tools/eval.py -c configs/rec/rec_latex_ocr.yml -o Global.pretrained_model=./rec_latex_ocr_train/best_accuracy.pdparams Metric.cal_blue_score=True Eval.dataset.data=./train_data/LaTeXOCR/latexocr_test.pkl
```

Prediction:

```
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_latex_ocr.yml  -o  Architecture.Backbone.is_predict=True Architecture.Backbone.is_export=True Architecture.Head.is_export=True Global.infer_img='./doc/datasets/pme_demo/0000013.png' Global.pretrained_model=./rec_latex_ocr_train/best_accuracy.pdparams
```

<a name="4"></a>
## 4. Inference and Deployment

<a name="4-1"></a>
### 4.1 Python Inference
First, the model saved during the LaTeX-OCR printed mathematical expression recognition training process is converted into an inference model. you can use the following command to convert:

```
python3 tools/export_model.py -c configs/rec/rec_latex_ocr.yml -o Global.pretrained_model=./rec_latex_ocr_train/best_accuracy.pdparams Global.save_inference_dir=./inference/rec_latex_ocr_infer/ Architecture.Backbone.is_predict=True Architecture.Backbone.is_export=True Architecture.Head.is_export=True

# The default output max length of the model is 512.
```

For LaTeX-OCR printed mathematical expression recognition model inference, the following commands can be executed:

```
python3 tools/infer/predict_rec.py --image_dir='./doc/datasets/pme_demo/0000295.png' --rec_algorithm="LaTeXOCR" --rec_batch_num=1 --rec_model_dir="./inference/rec_latex_ocr_infer/"  --rec_char_dict_path="./ppocr/utils/dict/latex_ocr_tokenizer.json"
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


```
