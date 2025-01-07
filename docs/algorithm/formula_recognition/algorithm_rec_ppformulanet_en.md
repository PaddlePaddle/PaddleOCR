# PP-FormulaNet

## 1. Introduction


PP-FormulaNet is a formula recognition model independently developed by Baidu PaddlePaddle. It is trained on a self-built dataset of 5 million samples within PaddleX, achieving the following accuracy on the corresponding test set:

| Model           | Backbone       | config                                                  |SPE-<br/>BLEU↑ | CPE-<br/>BLEU↑  | Easy-<br/>BLEU↑ | Middle-<br/>BLEU↑ | Hard-<br/>BLEU↑| Avg-<br/>BLEU↑  | Download link |
|-----------|--------|---------------------------------------------------|:--------------:|:-----------------:|:----------:|:----------------:|:---------:|:-----------------:|:--------------:|
| UniMERNet | Donut Swin | [rec_unimernet.yml](../../../configs/rec/rec_unimernet.yml) |     0.9187  |    0.9252       | 0.8658  |    0.8228   | 0.7740 |     0.8613        |[trained model](https://paddleocr.bj.bcebos.com/contribution/rec_unimernet_train.tar)|
| PP-FormulaNet-S | PPHGNetV2_B4 | [rec_pp_formulanet_s.yml](../../../configs/rec/PP-FormuaNet/rec_pp_formulanet_s.yml) |    0.8694   |    0.8071       | 0.9294  |    0.9112    | 0.8391 |    0.8712       |[trained model](https://paddleocr.bj.bcebos.com/contribution/rec_ppformulanet_s_train.tar)|
| PP-FormulaNet-L | Vary_VIT_B | [rec_pp_formulanet_l.yml](../../../configs/rec/PP-FormuaNet/rec_pp_formulanet_l.yml) |     0.9055   |     0.9206       | 0.9392  |     0.9273    | 0.9141 |     0.9213         |[trained model](https://paddleocr.bj.bcebos.com/contribution/rec_ppformulanet_l_train.tar )|

Among them, SPE and CPE refer to the simple and complex formula datasets of UniMERNet, respectively. Easy, Middle, and Hard are simple (LaTeX code length 0-64), medium (LaTeX code length 64-256), and complex formula datasets (LaTeX code length 256+) built internally by PaddleX.


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
python3 tools/train.py -c configs/rec/PP-FormuaNet/rec_pp_formulanet_s.yml \
   -o Global.pretrained_model=./rec_ppformulanet_s_train/best_accuracy.pdparams
#Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3' --ips=127.0.0.1   tools/train.py -c configs/rec/PP-FormuaNet/rec_pp_formulanet_s.yml \
        -o Global.pretrained_model=./rec_ppformulanet_s_train/best_accuracy.pdparams
```

Evaluation:

```shell
# GPU evaluation
 python3 tools/eval.py -c configs/rec/PP-FormuaNet/rec_pp_formulanet_s.yml -o \
 Global.pretrained_model=./rec_ppformulanet_s_train/best_accuracy.pdparams
```

Prediction:

```shell
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/PP-FormuaNet/rec_pp_formulanet_s.yml \
  -o  Global.infer_img='./docs/datasets/images/pme_demo/0000099.png'\
   Global.pretrained_model=./rec_ppformulanet_s_train/best_accuracy.pdparams
```

## 4. FAQ
