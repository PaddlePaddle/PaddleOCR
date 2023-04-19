English | [简体中文](../doc_ch/PP-OCRv3_det_train.md)


# The training steps of PP-OCRv3 text detection model

- [1. Introduction](#1)
- [2. PP-OCRv3 detection training](#2)
- [3. Finetune training based on PP-OCRv3 detection](#3)

<a name="1"></a>
## 1 Introduction

PP-OCRv3 is further upgraded on the basis of PP-OCRv2. This section describes the training steps of the PP-OCRv3 detection model. Refer to [documentation](./ppocr_introduction_en.md) for PP-OCRv3 introduction.


<a name="2"></a>
## 2. Detection training

The PP-OCRv3 detection model is an upgrade of the [CML](https://arxiv.org/pdf/2109.03144.pdf) (Collaborative Mutual Learning) collaborative mutual learning text detection distillation strategy in PP-OCRv2. PP-OCRv3 is further optimized for detecting teacher model and student model respectively. Among them, when optimizing the teacher model, the PAN structure LK-PAN with large receptive field and the DML (Deep Mutual Learning) distillation strategy are proposed. when optimizing the student model, the FPN structure RSE-FPN with residual attention mechanism is proposed.

PP-OCRv3 detection training consists of two steps:
- Step 1: Train detection teacher model using DML distillation method
- Step 2: Use the teacher model obtained in Step 1 to train a lightweight student model using the CML method


### 2.1 Prepare data and environment

The training data adopts icdar2015 data, and the steps to prepare the training set refer to [ocr_dataset](./dataset/ocr_datasets.md).

Runtime environment preparation reference [documentation](./installation_en.md).

### 2.2 Train the teacher model

The configuration file for teacher model training is [ch_PP-OCRv3_det_dml.yml](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.5/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml). The Backbone, Neck, and Head of the model structure of the teacher model are Resnet50, LKPAN, and DBHead, respectively, and are trained by the distillation method of DML. Refer to [documentation](./knowledge_distillation) for a detailed introduction to configuration files.


Download ImageNet pretrained models:
````
# Download the pretrained model of ResNet50_vd
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/ResNet50_vd_ssld_pretrained.pdparams
````

**Start training**
````
# Single GPU training
python3 tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml \
    -o Architecture.Models.Student.pretrained=./pretrain_models/ResNet50_vd_ssld_pretrained \
       Architecture.Models.Student2.pretrained=./pretrain_models/ResNet50_vd_ssld_pretrained \
       Global.save_model_dir=./output/

# If you want to use multi-GPU distributed training, use the following command:
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml \
    -o Architecture.Models.Student.pretrained=./pretrain_models/ResNet50_vd_ssld_pretrained \
       Architecture.Models.Student2.pretrained=./pretrain_models/ResNet50_vd_ssld_pretrained \
       Global.save_model_dir=./output/
````

The model saved during training is in the output directory and contains the following files:
````
best_accuracy.states
best_accuracy.pdparams # The model parameters with the best accuracy are saved by default
best_accuracy.pdopt # optimizer-related parameters that save optimal accuracy by default
latest.states
latest.pdparams # The latest model parameters saved by default
latest.pdopt # Optimizer related parameters of the latest model saved by default
````
Among them, best_accuracy is the saved model parameter with the highest accuracy, which can be directly evaluated using this model.

The model evaluation command is as follows:
````
python3 tools/eval.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml -o Global.checkpoints=./output/best_accuracy
````

The trained teacher model has a larger structure and higher accuracy, which is used to improve the accuracy of the student model.

**Extract teacher model parameters**
best_accuracy contains the parameters of two models, corresponding to Student and Student2 in the configuration file respectively. The method of extracting the parameters of Student is as follows:

````
import paddle
# load pretrained model
all_params = paddle.load("output/best_accuracy.pdparams")
# View the keys of the weight parameter
print(all_params.keys())
# model weight extraction
s_params = {key[len("Student."):]: all_params[key] for key in all_params if "Student." in key}
# View the keys of the model weight parameters
print(s_params.keys())
# save
paddle.save(s_params, "./pretrain_models/dml_teacher.pdparams")
````

The extracted model parameters can be used for further finetune training or distillation training of the model.


### 2.3 Train the student model

The configuration file for training the student model is [ch_PP-OCRv3_det_cml.yml](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.5/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml)
The teacher model trained in the previous section is used as supervision, and the lightweight student model is obtained by training in CML.

Download the ImageNet pretrained model for the student model:
````
# Download the pre-trained model of MobileNetV3
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/MobileNetV3_large_x0_5_pretrained.pdparams
````

**Start training**

````
# Single card training
python3 tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
    -o Architecture.Models.Student.pretrained=./pretrain_models/MobileNetV3_large_x0_5_pretrained \
       Architecture.Models.Student2.pretrained=./pretrain_models/MobileNetV3_large_x0_5_pretrained \
       Architecture.Models.Teacher.pretrained=./pretrain_models/dml_teacher \
       Global.save_model_dir=./output/
# If you want to use multi-GPU distributed training, use the following command:
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
    -o Architecture.Models.Student.pretrained=./pretrain_models/MobileNetV3_large_x0_5_pretrained \
       Architecture.Models.Student2.pretrained=./pretrain_models/MobileNetV3_large_x0_5_pretrained \
       Architecture.Models.Teacher.pretrained=./pretrain_models/dml_teacher \
       Global.save_model_dir=./output/
````

The model saved during training is in the output directory,
The model evaluation command is as follows:
````
python3 tools/eval.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml -o Global.checkpoints=./output/best_accuracy
````

best_accuracy contains three model parameters, corresponding to Student, Student2, and Teacher in the configuration file. The method to extract the Student parameter is as follows:

````
import paddle
# load pretrained model
all_params = paddle.load("output/best_accuracy.pdparams")
# View the keys of the weight parameter
print(all_params.keys())
# model weight extraction
s_params = {key[len("Student."):]: all_params[key] for key in all_params if "Student." in key}
# View the keys of the model weight parameters
print(s_params.keys())
# save
paddle.save(s_params, "./pretrain_models/cml_student.pdparams")
````

The extracted parameters of Student can be used for model deployment or further finetune training.



<a name="3"></a>
## 3. Finetune training based on PP-OCRv3 detection

This section describes how to use the finetune training of the PP-OCRv3 detection model on other scenarios.

finetune training applies to three scenarios:
- The finetune training based on the CML distillation method is suitable for the teacher model whose accuracy is higher than the PP-OCRv3 detection model in the usage scene, and a lightweight detection model is desired.
- Finetune training based on the PP-OCRv3 lightweight detection model, without the need to train the teacher model, hoping to improve the accuracy of the usage scenarios based on the PP-OCRv3 detection model.
- The finetune training based on the DML distillation method is suitable for scenarios where the DML method is used to further improve the accuracy.


**finetune training based on CML distillation method**

Download the PP-OCRv3 training model:
````
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar
tar xf ch_PP-OCRv3_det_distill_train.tar
````
ch_PP-OCRv3_det_distill_train/best_accuracy.pdparams contains the parameters of the Student, Student2, and Teacher models in the CML configuration file.

Start training:

````
# Single card training
python3 tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
    -o Global.pretrained_model=./ch_PP-OCRv3_det_distill_train/best_accuracy \
       Global.save_model_dir=./output/
# If you want to use multi-GPU distributed training, use the following command:
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
    -o Global.pretrained_model=./ch_PP-OCRv3_det_distill_train/best_accuracy \
       Global.save_model_dir=./output/
````

**finetune training based on PP-OCRv3 lightweight detection model**


Download the PP-OCRv3 training model and extract the model parameters of the Student structure:
````
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar
tar xf ch_PP-OCRv3_det_distill_train.tar
````

The method to extract the Student parameter is as follows:

````
import paddle
# load pretrained model
all_params = paddle.load("output/best_accuracy.pdparams")
# View the keys of the weight parameter
print(all_params.keys())
# model weight extraction
s_params = {key[len("Student."):]: all_params[key] for key in all_params if "Student." in key}
# View the keys of the model weight parameters
print(s_params.keys())
# save
paddle.save(s_params, "./student.pdparams")
````

Trained using the configuration file [ch_PP-OCRv3_det_student.yml](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.5/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml).

**Start training**

````
# Single card training
python3 tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml \
    -o Global.pretrained_model=./student \
       Global.save_model_dir=./output/
# If you want to use multi-GPU distributed training, use the following command:
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml \
    -o Global.pretrained_model=./student \
       Global.save_model_dir=./output/
````


**finetune training based on DML distillation method**

Taking the Teacher model in ch_PP-OCRv3_det_distill_train as an example, first extract the parameters of the Teacher structure as follows:
````
import paddle
# load pretrained model
all_params = paddle.load("ch_PP-OCRv3_det_distill_train/best_accuracy.pdparams")
# View the keys of the weight parameter
print(all_params.keys())
# model weight extraction
s_params = {key[len("Teacher."):]: all_params[key] for key in all_params if "Teacher." in key}
# View the keys of the model weight parameters
print(s_params.keys())
# save
paddle.save(s_params, "./teacher.pdparams")
````

**Start training**
````
# Single card training
python3 tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml \
     -o Architecture.Models.Student.pretrained=./teacher \
        Architecture.Models.Student2.pretrained=./teacher \
        Global.save_model_dir=./output/
# If you want to use multi-GPU distributed training, use the following command:
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml \
     -o Architecture.Models.Student.pretrained=./teacher \
        Architecture.Models.Student2.pretrained=./teacher \
        Global.save_model_dir=./output/
````
