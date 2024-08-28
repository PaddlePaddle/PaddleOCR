# LaTeX-OCR

## 1. Introduction

Original Project:
> [https://github.com/lukas-blecher/LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR)


Using LaTeX-OCR printed mathematical expression recognition datasets for training, and evaluating on its test sets, the algorithm reproduction effect is as follows:

| Model       | Backbone |config| BLEU score  | normed edit distance  |  ExpRate  |Download link|
|-----------|----------| ---- |:-----------:|:---------------------:|:---------:| ----- |
| LaTeX-OCR | Hybrid ViT |[rec_latex_ocr.yml](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/rec_latex_ocr.yml)|   0.8821    |        0.0823         |  40.01%   |[trained model](https://paddleocr.bj.bcebos.com/contribution/rec_latex_ocr_train.tar)|

## 2. Environment
Please refer to ["Environment Preparation"](../../ppocr/environment.en.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](../../ppocr/blog/clone.en.md) to clone the project code.

Furthermore, additional dependencies need to be installed:
```shell
pip install -r docs/algorithm/formula_recognition/requirements.txt
```

## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Tutorial](../../ppocr/model_train/recognition.en.md). PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**.

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
python3 tools/eval.py -c configs/rec/rec_latex_ocr.yml -o Global.pretrained_model=./rec_latex_ocr_train/best_accuracy.pdparams
# Test set evaluation
python3 tools/eval.py -c configs/rec/rec_latex_ocr.yml -o Global.pretrained_model=./rec_latex_ocr_train/best_accuracy.pdparams Eval.dataset.data_dir=./train_data/LaTeXOCR/test Eval.dataset.data=./train_data/LaTeXOCR/latexocr_test.pkl
```

Prediction:

```
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_latex_ocr.yml  -o  Global.infer_img='./docs/datasets/images/pme_demo/0000013.png' Global.pretrained_model=./rec_latex_ocr_train/best_accuracy.pdparams
```

## 4. Inference and Deployment

### 4.1 Python Inference
First, the model saved during the LaTeX-OCR printed mathematical expression recognition training process is converted into an inference model. you can use the following command to convert:

```
python3 tools/export_model.py -c configs/rec/rec_latex_ocr.yml -o Global.pretrained_model=./rec_latex_ocr_train/best_accuracy.pdparams Global.save_inference_dir=./inference/rec_latex_ocr_infer/ 

# The default output max length of the model is 512.
```

For LaTeX-OCR printed mathematical expression recognition model inference, the following commands can be executed:

```
python3 tools/infer/predict_rec.py --image_dir='./docs/datasets/images/pme_demo/0000295.png' --rec_algorithm="LaTeXOCR" --rec_batch_num=1 --rec_model_dir="./inference/rec_latex_ocr_infer/"  --rec_char_dict_path="./ppocr/utils/dict/latex_ocr_tokenizer.json"
```

### 4.2 C++ Inference

Not supported

### 4.3 Serving

Not supported

### 4.4 More

Not supported

## 5. FAQ
