---
comments: true
---

# PP-OCRv3 text detection model training

## 1. Introduction

PP-OCRv3 is a further upgrade of PP-OCRv2. This section introduces the training steps of the PP-OCRv3 detection model. For an introduction to the PP-OCRv3 strategy, refer to [document](../blog/PP-OCRv3_introduction.md).

## 2. Detection training

The PP-OCRv3 detection model is an upgrade of the [CML](https://arxiv.org/pdf/2109.03144.pdf) (Collaborative Mutual Learning) collaborative mutual learning text detection distillation strategy in PP-OCRv2. PP-OCRv3 further optimizes the detection teacher model and student model. Among them, when optimizing the teacher model, the PAN structure LK-PAN with a large receptive field and the DML (Deep Mutual Learning) distillation strategy are proposed; when optimizing the student model, the FPN structure RSE-FPN with a residual attention mechanism is proposed.

PP-OCRv3 detection training includes two steps:

- Step 1: Use DML distillation method to train detection teacher model

- Step 2: Use the teacher model obtained in step 1 to train a lightweight student model using CML method

### 2.1 Prepare data and operating environment

The training data uses icdar2015 data. For the steps of preparing the training set, refer to [ocr_dataset](./dataset/ocr_datasets.md).

For the preparation of the operating environment, refer to [document](./installation.md).

### 2.2 Train the teacher model

The configuration file for teacher model training is [ch_PP-OCRv3_det_dml.yml](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.5/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml). The Backbone, Neck, and Head of the teacher model structure are Resnet50, LKPAN, and DBHead respectively, and are trained using the DML distillation method. For a detailed introduction to the configuration file, refer to [Document](./knowledge_distillation.md).

Download ImageNet pre-trained model:

```bash linenums="1"
# Download ResNet50_vd pre-trained model
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/ResNet50_vd_ssld_pretrained.pdparams
```

**Start training**

```bash linenums="1"
# Single card training
python3 tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml \
-o Architecture.Models.Student.pretrained=./pretrain_models/ResNet50_vd_ssld_pretrained \
Architecture.Models.Student2.pretrained=./pretrain_models/ResNet50_vd_ssld_pretrained \
Global.save_model_dir=./output/
# If you want to use multi-GPU distributed training, please use the following command:
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml \
-o Architecture.Models.Student.pretrained=./pretrain_models/ResNet50_vd_ssld_pretrained \
Architecture.Models.Student2.pretrained=./pretrain_models/ResNet50_vd_ssld_pretrained \
Global.save_model_dir=./output/
```

The model saved during training is in the output directory, which contains the following files:

```bash linenums="1"
best_accuracy.states
best_accuracy.pdparams # The model parameters with the best accuracy are saved by default
best_accuracy.pdopt # The optimizer-related parameters with the best accuracy are saved by default
latest.states
latest.pdparams # The latest model parameters saved by default
latest.pdopt # The optimizer-related parameters of the latest model saved by default
```

Among them, best_accuracy is the model parameter with the highest accuracy saved, and the model can be directly used for evaluation.

The model evaluation command is as follows:

```bash linenums="1"
python3 tools/eval.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml -o Global.checkpoints=./output/best_accuracy
```

The trained teacher model has a larger structure and higher accuracy, which is used to improve the accuracy of the student model.

**Extract teacher model parameters**
best_accuracy contains the parameters of two models, corresponding to Student and Student2 in the configuration file. The method to extract the parameters of Student is as follows:

```bash linenums="1"
import paddle
# Load pre-trained model
all_params = paddle.load("output/best_accuracy.pdparams")
# View the keys of weight parameters
print(all_params.keys())
# Model weight extraction
s_params = {key[len("Student."):]: all_params[key] for key in all_params if "Student." in key}
# View the keys of model weight parameters
print(s_params.keys())
# Save
paddle.save(s_params, "./pretrain_models/dml_teacher.pdparams")
```

The extracted model parameters can be used for further fine-tuning or distillation training of the model.

### 2.3 Training the student model

The configuration file for training the student model is [ch_PP-OCRv3_det_cml.yml](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.5/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml)
The teacher model trained in the previous section is used as supervision, and the CML method is used to train a lightweight student model.

Download the ImageNet pre-trained model of the student model:

```bash linenums="1"
# Download the pre-trained model of MobileNetV3
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/MobileNetV3_large_x0_5_pretrained.pdparams
```

**Start training**

```bash linenums="1"
# Single card training
python3 tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
-o Architecture.Models.Student.pretrained=./pretrain_models/MobileNetV3_large_x0_5_pretrained \
Architecture.Models.Student2.pretrained=./pretrain_models/MobileNetV3_large_x0_5_pretrained \
Architecture.Models.Teacher.pretrained=./pretrain_models/dml_teacher \
Global.save_model_dir=./output/
# If you want to use multi-GPU distributed training, please use the following command:
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
-o Architecture.Models.Student.pretrained=./pretrain_models/MobileNetV3_large_x0_5_pretrained \
Architecture.Models.Student2.pretrained=./pretrain_models/MobileNetV3_large_x0_5_pretrained \
Architecture.Models.Teacher.pretrained=./pretrain_models/dml_teacher \
Global.save_model_dir=./output/
```

The model saved during the training process is in the output directory.
The model evaluation command is as follows:

```bash linenums="1"
python3 tools/eval.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml -o Global.checkpoints=./output/best_accuracy
```

best_accuracy contains the parameters of three models, corresponding to Student, Student2, and Teacher in the configuration file. The method to extract Student parameters is as follows:

```bash linenums="1"
import paddle
# Load pre-trained model
all_params = paddle.load("output/best_accuracy.pdparams")
# View the keys of weight parameters
print(all_params.keys())
# Model weight extraction
s_params = {key[len("Student."):]: all_params[key] for key in all_params if "Student." in key}
# View the keys of model weight parameters
print(s_params.keys())
# Save
paddle.save(s_params, "./pretrain_models/cml_student.pdparams")
```

The extracted Student parameters can be used for model deployment or further fine-tuning training.

## 3. Fine-tune training based on PP-OCRv3 detection

This section describes how to use the PP-OCRv3 detection model for fine-tune training in other scenarios.

Fine-tune training is applicable to three scenarios:

- Fine-tune training based on the CML distillation method is applicable to scenarios where the teacher model has higher accuracy than the PP-OCRv3 detection model in the usage scenario and a lightweight detection model is desired.

- Fine-tune training based on the PP-OCRv3 lightweight detection model does not require the training of the teacher model and is intended to improve the accuracy of the usage scenario based on the PP-OCRv3 detection model.

- Fine-tune training based on the DML distillation method is applicable to scenarios where the DML method is used to further improve accuracy.

**Finetune training based on CML distillation method**

Download PP-OCRv3 training model:

```bash linenums="1"
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar
tar xf ch_PP-OCRv3_det_distill_train.tar
```

ch_PP-OCRv3_det_distill_train/best_accuracy.pdparams contains the parameters of Student, Student2, and Teacher models in the CML configuration file.

Start training:

```bash linenums="1"
# Single card training
python3 tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
-o Global.pretrained_model=./ch_PP-OCRv3_det_distill_train/best_accuracy \
Global.save_model_dir=./output/
# If you want to use multi-GPU distributed training, please use the following command:
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
-o Global.pretrained_model=./ch_PP-OCRv3_det_distill_train/best_accuracy \
Global.save_model_dir=./output/
```

**Finetune training based on PP-OCRv3 lightweight detection model**

Download PP-OCRv3 training model and extract model parameters of Student structure:

```
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar
tar xf ch_PP-OCRv3_det_distill_train.tar
```

The method to extract Student parameters is as follows:

```bash linenums="1"
import paddle
# Load pre-trained model
all_params = paddle.load("output/best_accuracy.pdparams")
# View the keys of weight parameters
print(all_params.keys())
# Model weight extraction
s_params = {key[len("Student."):]: all_params[key] for key in all_params if "Student." in key}
# View the keys of the model weight parameters
print(s_params.keys())
# Save
paddle.save(s_params, "./student.pdparams")
```

Train using the configuration file [ch_PP-OCRv3_det_student.yml](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.5/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml).

**Start training**

```bash linenums="1"
# Single card training
python3 tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml \
-o Global.pretrained_model=./student \
Global.save_model_dir=./output/
# If you want to use multi-GPU distributed training, please use the following command:
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml \
-o Global.pretrained_model=./student \
Global.save_model_dir=./output/
```

**Finetune training based on DML distillation method**

Take the Teacher model in ch_PP-OCRv3_det_distill_train as an example. First, extract the parameters of the Teacher structure. The method is as follows:

```bash linenums="1"
import paddle
# Load pre-trained model
all_params = paddle.load("ch_PP-OCRv3_det_distill_train/best_accuracy.pdparams")
# View the keys of weight parameters
print(all_params.keys())
# Model weight extraction
s_params = {key[len("Teacher."):]: all_params[key] for key in all_params if "Teacher." in key}
# View the keys of model weight parameters
print(s_params.keys())
# Save
paddle.save(s_params, "./teacher.pdparams")
```

**Start training**

```bash linenums="1"
# Single card training
python3 tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml \
-o Architecture.Models.Student.pretrained=./teacher \
Architecture.Models.Student2.pretrained=./teacher \
Global.save_model_dir=./output/
# If you want to use multi-GPU distributed training, please use the following command:
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml \
-o Architecture.Models.Student.pretrained=./teacher \
Architecture.Models.Student2.pretrained=./teacher \
Global.save_model_dir=./output/
```
