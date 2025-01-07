# UniMERNet

## 1. Introduction

Original Project:
> [https://github.com/opendatalab/UniMERNet](https://github.com/opendatalab/UniMERNet)


Using UniMERNet general mathematical expression recognition datasets for training, and evaluating on its test sets, the algorithm reproduction effect is as follows:

| Model           | Backbone       | config                                                  | SPE-<br/>BLEU↑ | SPE-<br/>EditDis↓ | CPE-<br/>BLEU↑  |CPE-<br/>EditDis↓ | SCE-<br/>BLEU↑ | SCE-<br/>EditDis↓ | HWE-<br/>BLEU↑ | HWE-<br/>EditDis↓ | Download link |
|-----------|--------|---------------------------------------------------|:--------------:|:-----------------:|:----------:|:----------------:|:---------:|:-----------------:|:--------------:|:-----------------:|---|
| UniMERNet | Donut Swin | [rec_unimernet.yml](../../../configs/rec/rec_unimernet.yml) |     0.9187     |      0.0584       |  0.9252    |      0.0596      | 0.6068 |     0.2297        |   0.9157|     0.0546           |[trained model](https://paddleocr.bj.bcebos.com/contribution/rec_unimernet_train.tar)|

SPE represents simple formulas, CPE represents complex formulas, SCE represents scanned captured formulas, and HWE represents handwritten formulas. Example images of each type of formula are shown below:

![unimernet_dataset](https://github.com/user-attachments/assets/fb801a36-5614-4031-8585-700bd9f8fb2e)

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

Download UniMER-1M.zip and UniMER-Test.zip from [Hugging Face]((https://huggingface.co/datasets/wanderkid/UniMER_Dataset/tree/main)). Download the HME100K dataset from the [TAL AI Platform](https://ai.100tal.com/dataset). After that, use the following command to create a dataset directory and convert the dataset.

```shell
# create the UniMERNet dataset directory
mkdir -p train_data/UniMERNet
# unzip UniMERNet 、 UniMER-Test.zip and HME100K.zip
unzip -d train_data/UniMERNet path/UniMER-1M.zip
unzip -d train_data/UniMERNet path/UniMER-Test.zip
unzip -d train_data/UniMERNet/HME100K train_data/UniMERNet/HME100K/train.zip
unzip -d train_data/UniMERNet/HME100K train_data/UniMERNet/HME100K/test.zip
# convert the training set 
python ppocr/utils/formula_utils/unimernet_data_convert.py \
     --image_dir=train_data/UniMERNet \
     --datatype=unimernet_train \
     --unimernet_txt_path=train_data/UniMERNet/UniMER-1M/train.txt \
     --hme100k_txt_path=train_data/UniMERNet/HME100K/train_labels.txt \
     --output_path=train_data/UniMERNet/train_unimernet_1M.txt
# convert the test set
# SPE
python ppocr/utils/formula_utils/unimernet_data_convert.py \
     --image_dir=train_data/UniMERNet/UniMER-Test/spe \
     --datatype=unimernet_test \
     --unimernet_txt_path=train_data/UniMERNet/UniMER-Test/spe.txt \
     --output_path=train_data/UniMERNet/test_unimernet_spe.txt
# CPE
python ppocr/utils/formula_utils/unimernet_data_convert.py \
     --image_dir=train_data/UniMERNet/UniMER-Test/cpe \
     --datatype=unimernet_test \
     --unimernet_txt_path=train_data/UniMERNet/UniMER-Test/cpe.txt \
     --output_path=train_data/UniMERNet/test_unimernet_cpe.txt
# SCE
python ppocr/utils/formula_utils/unimernet_data_convert.py \
     --image_dir=train_data/UniMERNet/UniMER-Test/sce \
     --datatype=unimernet_test \
     --unimernet_txt_path=train_data/UniMERNet/UniMER-Test/sce.txt \
     --output_path=train_data/UniMERNet/test_unimernet_sce.txt
# HWE
python ppocr/utils/formula_utils/unimernet_data_convert.py \
     --image_dir=train_data/UniMERNet/UniMER-Test/hwe \
     --datatype=unimernet_test \
     --unimernet_txt_path=train_data/UniMERNet/UniMER-Test/hwe.txt \
     --output_path=train_data/UniMERNet/test_unimernet_hwe.txt
```

Download the Pre-trained Model:

```shell
# download the Texify pre-trained model
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/texify.pdparams
```

Training:

Specifically, after the data preparation is completed, the training can be started. The training command is as follows:

```shell
#Single GPU training 
python3 tools/train.py -c configs/rec/rec_unimernet.yml \
   -o Global.pretrained_model=./pretrain_models/texify.pdparams
#Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3' --ips=127.0.0.1   tools/train.py -c configs/rec/rec_unimernet.yml \
        -o Global.pretrained_model=./pretrain_models/texify.pdparams
```

Evaluation:

```shell
# GPU evaluation
 # SPE test set evaluation
 python3 tools/eval.py -c configs/rec/rec_unimernet.yml -o \
  Eval.dataset.data_dir=./train_data/UniMERNet/UniMER-Test/spe \
  Eval.dataset.label_file_list=["./train_data/UniMERNet/test_unimernet_spe.txt"] \
 Global.pretrained_model=./rec_unimernet_train/best_accuracy.pdparams
 # CPE test set evaluation
 python3 tools/eval.py -c configs/rec/rec_unimernet.yml -o \
  Eval.dataset.data_dir=./train_data/UniMERNet/UniMER-Test/cpe \
  Eval.dataset.label_file_list=["./train_data/UniMERNet/test_unimernet_cpe.txt"] \
 Global.pretrained_model=./rec_unimernet_train/best_accuracy.pdparams
 # SCE test set evaluation
  python3 tools/eval.py -c configs/rec/rec_unimernet.yml -o \
  Eval.dataset.data_dir=./train_data/UniMERNet/UniMER-Test/sce \
  Eval.dataset.label_file_list=["./train_data/UniMERNet/test_unimernet_sce.txt"] \
 Global.pretrained_model=./rec_unimernet_train/best_accuracy.pdparams
 # HWE test set evaluation
 python3 tools/eval.py -c configs/rec/rec_unimernet.yml -o \
  Eval.dataset.data_dir=./train_data/UniMERNet/UniMER-Test/hwe \
  Eval.dataset.label_file_list=["./train_data/UniMERNet/test_unimernet_hwe.txt"] \
 Global.pretrained_model=./rec_unimernet_train/best_accuracy.pdparams

```

Prediction:

```shell
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_unimernet.yml \
  -o  Global.infer_img='./docs/datasets/images/pme_demo/0000099.png'\
   Global.pretrained_model=./rec_unimernet_train/best_accuracy.pdparams
```

## 4. FAQ
