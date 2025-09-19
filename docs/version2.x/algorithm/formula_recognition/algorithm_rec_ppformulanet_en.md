# PP-FormulaNet

## 1. Introduction


`PP-FormulaNet` is an advanced formula recognition model developed by Baidu’s PaddlePaddle Vision Team, supporting the recognition of 50,000 common LaTeX source terms. The PP-FormulaNet-S version uses PP-HGNetV2-B4 as its backbone network, leveraging techniques like parallel masking and model distillation to significantly enhance inference speed while maintaining high recognition accuracy, making it suitable for scenarios involving simple printed formulas and cross-line simple printed formulas. On the other hand, the PP-FormulaNet-L version is based on Vary_VIT_B and has undergone extensive training on a large-scale formula dataset, showing significant improvement in recognizing complex formulas, and is applicable to simple printed, complex printed, and handwritten formulas.

The accuracy of the above models on the corresponding test sets is as follows:

| Model           | Backbone       | config                  | En-BLEU↑  | GPU Inference Time (ms)| Download link |
|-----------|--------|----------------------------------------|:-----------------:|:--------------:|:--------------:|
| UniMERNet | Donut Swin | [UniMERNet.yaml](../../../configs/rec/UniMERNet.yaml) |     85.91  | 2266.96 | [trained model](https://paddleocr.bj.bcebos.com/contribution/rec_unimernet_train.tar)|
| PP-FormulaNet-S | PPHGNetV2_B4 | [PP-FormulaNet-S.yaml](../../../configs/rec/PP-FormuaNet/PP-FormulaNet-S.yaml) |   87.00   | 202.25 |[trained model](https://paddleocr.bj.bcebos.com/contribution/rec_ppformulanet_s_train.tar)|
| PP-FormulaNet-L | Vary_VIT_B | [PP-FormulaNet-L.yaml](../../../configs/rec/PP-FormuaNet/PP-FormulaNet-L.yaml) |    90.36   | 1976.52  |[trained model](https://paddleocr.bj.bcebos.com/contribution/rec_ppformulanet_l_train.tar )|
| LaTeX-OCR | Hybrid ViT |[LaTeX_OCR_rec.yaml](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/LaTeX_OCR_rec.yaml)|   74.55   | 	1244.61   |[trained model](https://paddleocr.bj.bcebos.com/contribution/rec_latex_ocr_train.tar)|


The English formula evaluation set here contains both simple and complex formulas from UniMERNet, as well as simple, medium, and complex formulas independently created by PaddleX.


## 2. Environment
Please refer to ["Environment Preparation"](../../ppocr/environment.en.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](../../ppocr/blog/clone.en.md) to clone the project code.

Furthermore, additional dependencies need to be installed:
```shell
sudo apt-get update
sudo apt-get install libmagickwand-dev
pip install -r docs/algorithm/formula_recognition/requirements.txt
```

## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Tutorial](../../ppocr/model_train/recognition.en.md). PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**.


Dataset Preparation:

```shell
# download PaddleX official example dataset
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_latexocr_dataset_example.tar
tar -xf ocr_rec_latexocr_dataset_example.tar
```

Download the Pre-trained Model:

```shell
# download the PP-FormulaNet-S pretrained model
wget https://paddleocr.bj.bcebos.com/contribution/rec_ppformulanet_s_train.tar 
tar -xf rec_ppformulanet_s_train.tar
```

Training:

Specifically, after the data preparation is completed, the training can be started. The training command is as follows:

```shell
#Single GPU training 
python3 tools/train.py -c configs/rec/PP-FormuaNet/PP-FormulaNet-S.yaml \
   -o Global.pretrained_model=./rec_ppformulanet_s_train/best_accuracy.pdparams
#Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3' --ips=127.0.0.1   tools/train.py -c configs/rec/PP-FormuaNet/PP-FormulaNet-S.yaml \
        -o Global.pretrained_model=./rec_ppformulanet_s_train/best_accuracy.pdparams
```

Evaluation:

```shell
# GPU evaluation
 python3 tools/eval.py -c configs/rec/PP-FormuaNet/PP-FormulaNet-S.yaml -o \
 Global.pretrained_model=./rec_ppformulanet_s_train/best_accuracy.pdparams
```

Prediction:

```shell
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/PP-FormuaNet/PP-FormulaNet-S.yaml \
  -o  Global.infer_img='./docs/datasets/images/pme_demo/0000295.png'\
   Global.pretrained_model=./rec_ppformulanet_s_train/best_accuracy.pdparams
```

## 4. FAQ
