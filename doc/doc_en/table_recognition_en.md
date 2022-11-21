# Table Recognition

This article provides a full-process guide for the PaddleOCR table recognition model, including data preparation, model training, tuning, evaluation, prediction, and detailed descriptions of each stage:

- [1. Data Preparation](#1-data-preparation)
  - [1.1. DataSet Format](#11-dataset-format)
  - [1.2. Data Download](#12-data-download)
  - [1.3. Dataset Generation](#13-dataset-generation)
- [2. Training](#2-training)
  - [2.1. Start Training](#21-start-training)
  - [2.2. Resume Training](#22-resume-training)
  - [2.3. Training with New Backbone](#23-training-with-new-backbone)
  - [2.4. Mixed Precision Training](#24-mixed-precision-training)
  - [2.5. Distributed Training](#25-distributed-training)
  - [2.6. Training on other platform(Windows/macOS/Linux DCU)](#26-training-on-other-platformwindowsmacoslinux-dcu)
  - [2.7. Fine-tuning](#27-fine-tuning)
    - [2.7.1 Dataset](#271-dataset)
    - [2.7.2 model selection](#272-model-selection)
    - [2.7.3 Training hyperparameter selection](#273-training-hyperparameter-selection)
- [3. Evaluation and Test](#3-evaluation-and-test)
  - [3.1. Evaluation](#31-evaluation)
  - [3.2. Test table structure recognition effect](#32-test-table-structure-recognition-effect)
- [4. Model export and prediction](#4-model-export-and-prediction)
  - [4.1 Model export](#41-model-export)
  - [4.2 Prediction](#42-prediction)
- [5. FAQ](#5-faq)

# 1. Data Preparation

## 1.1. DataSet Format

The format of the PaddleOCR table recognition model dataset is as follows:
```txt
img_label # Each image is marked with a string after json.dumps()
...
img_label
```

The json format of each line is:
```txt
{
   'filename': PMC5755158_010_01.png,# image name
   'split': ’train‘, # whether the image belongs to the training set or the validation set
   'imgid': 0,# index of image
   'html': {
     'structure': {'tokens': ['<thead>', '<tr>', '<td>', ...]}, # HTML string of the table
     'cells': [
       {
         'tokens': ['P', 'a', 'd', 'd', 'l', 'e', 'P', 'a', 'd', 'd', 'l', 'e'], # text in cell
         'bbox': [x0, y0, x1, y1] # bbox of cell
       }
     ]
   }
}
```

The default storage path for training data is `PaddleOCR/train_data`, if you already have a dataset on disk, just create a soft link to the dataset directory:

```
# linux and mac os
ln -sf <path/to/dataset> <path/to/paddle_ocr>/train_data/dataset
# windows
mklink /d <path/to/paddle_ocr>/train_data/dataset <path/to/dataset>
```

## 1.2. Data Download

Download the public dataset reference [table_datasets](dataset/table_datasets_en.md)。

## 1.3. Dataset Generation

Use [TableGeneration](https://github.com/WenmuZhou/TableGeneration) to generate scanned table images.

TableGeneration is an open source table dataset generation tool, which renders html strings through browser rendering to obtain table images.

Some samples are as follows:

|Type|Sample|
|---|---|
|Simple Table|![](https://raw.githubusercontent.com/WenmuZhou/TableGeneration/main/imgs/simple.jpg)|
|Simple Color Table|![](https://raw.githubusercontent.com/WenmuZhou/TableGeneration/main/imgs/color.jpg)|

# 2. Training

PaddleOCR provides training scripts, evaluation scripts, and prediction scripts. In this section, the [SLANet](../../configs/table/SLANet.yml) model will be used as an example:

## 2.1. Start Training

*If you are installing the cpu version, please modify the `use_gpu` field in the configuration file to false*

```
# GPU training Support single card and multi-card training
# The training log will be automatically saved as train.log under "{save_model_dir}"

# specify the single card training(Long training time, not recommended)
python3 tools/train.py -c configs/table/SLANet.yml

# specify the card number through --gpus
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/table/SLANet.yml
```

After starting training normally, you will see the following log output:

```
[2022/08/16 03:07:33] ppocr INFO: epoch: [1/400], global_step: 20, lr: 0.000100, acc: 0.000000, loss: 3.915012, structure_loss: 3.229450, loc_loss: 0.670590, avg_reader_cost: 2.63382 s, avg_batch_cost: 6.32390 s, avg_samples: 48.0, ips: 7.59025 samples/s, eta: 9 days, 2:29:27
[2022/08/16 03:08:41] ppocr INFO: epoch: [1/400], global_step: 40, lr: 0.000100, acc: 0.000000, loss: 1.750859, structure_loss: 1.082116, loc_loss: 0.652822, avg_reader_cost: 0.02533 s, avg_batch_cost: 3.37251 s, avg_samples: 48.0, ips: 14.23271 samples/s, eta: 6 days, 23:28:43
[2022/08/16 03:09:46] ppocr INFO: epoch: [1/400], global_step: 60, lr: 0.000100, acc: 0.000000, loss: 1.395154, structure_loss: 0.776803, loc_loss: 0.625030, avg_reader_cost: 0.02550 s, avg_batch_cost: 3.26261 s, avg_samples: 48.0, ips: 14.71214 samples/s, eta: 6 days, 5:11:48
```

The following information is automatically printed in the log:

|  Field   |   Meaning   |  
| :----: | :------: |
|  epoch | current iteration round |
|  global_step  | current iteration count |
|  lr    | current learning rate |
|  acc   | The accuracy of the current batch |
|  loss  | current loss function |
|  structure_loss | Table Structure Loss Values |
|  loc_loss | Cell Coordinate Loss Value |
|  avg_reader_cost | Current batch data processing time |
|  avg_batch_cost | The total time spent in the current batch |
|  avg_samples  | The number of samples in the current batch |
|  ips  | Number of images processed per second |


PaddleOCR supports alternating training and evaluation. You can modify `eval_batch_step` in `configs/table/SLANet.yml` to set the evaluation frequency. By default, it is evaluated once every 1000 iters. During the evaluation process, the best acc model is saved as `output/SLANet/best_accuracy` by default.

If the validation set is large, the test will be time-consuming. It is recommended to reduce the number of evaluations, or perform evaluation after training.

**Tips:** You can use the -c parameter to select various model configurations under the `configs/table/` path for training. For the table recognition algorithms supported by PaddleOCR, please refer to [Table Algorithms List](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_en/algorithm_overview_en.md#3):

**Note that the configuration file for prediction/evaluation must be the same as training. **

## 2.2. Resume Training

If the training program is interrupted, if you want to load the interrupted model to resume training, you can specify the path of the model to be loaded by specifying Global.checkpoints:

```shell
python3 tools/train.py -c configs/table/SLANet.yml -o Global.checkpoints=./your/trained/model
```
**Note**: The priority of `Global.checkpoints` is higher than that of `Global.pretrained_model`, that is, when two parameters are specified at the same time, the model specified by `Global.checkpoints` will be loaded first. If `Global.checkpoints` The specified model path is incorrect, and the model specified by `Global.pretrained_model` will be loaded.

## 2.3. Training with New Backbone

The network part completes the construction of the network, and PaddleOCR divides the network into four parts, which are under [ppocr/modeling](../../ppocr/modeling). The data entering the network will pass through these four parts in sequence(transforms->backbones->
necks->heads).

```bash
├── architectures # Code for building network
├── transforms    # Image Transformation Module
├── backbones     # Feature extraction module
├── necks         # Feature enhancement module
└── heads         # Output module
```

If the Backbone to be replaced has a corresponding implementation in PaddleOCR, you can directly modify the parameters in the `Backbone` part of the configuration yml file.

However, if you want to use a new Backbone, an example of replacing the backbones is as follows:

1. Create a new file under the [ppocr/modeling/backbones](../../ppocr/modeling/backbones) folder, such as my_backbone.py.
2. Add code in the my_backbone.py file, the sample code is as follows:

```python
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class MyBackbone(nn.Layer):
    def __init__(self, *args, **kwargs):
        super(MyBackbone, self).__init__()
        # your init code
        self.conv = nn.xxxx

    def forward(self, inputs):
        # your network forward
        y = self.conv(inputs)
        return y
```

3. Import the added module in the [ppocr/modeling/backbones/\__init\__.py](../../ppocr/modeling/backbones/__init__.py) file.

After adding the four-part modules of the network, you only need to configure them in the configuration file to use, such as:

```yaml
  Backbone:
    name: MyBackbone
    args1: args1
```

**NOTE**: More details about replace Backbone and other mudule can be found in [doc](add_new_algorithm_en.md).

## 2.4. Mixed Precision Training

If you want to speed up your training further, you can use [Auto Mixed Precision Training](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/basic_concept/amp_cn.html), taking a single machine and a single gpu as an example, the commands are as follows:

```shell
python3 tools/train.py -c configs/table/SLANet.yml \
     -o Global.pretrained_model=./pretrain_models/SLANet/best_accuracy \
     Global.use_amp=True Global.scale_loss=1024.0 Global.use_dynamic_loss_scaling=True
 ```

## 2.5. Distributed Training

During multi-machine multi-gpu training, use the `--ips` parameter to set the used machine IP address, and the `--gpus` parameter to set the used GPU ID:

```bash
python3 -m paddle.distributed.launch --ips="xx.xx.xx.xx,xx.xx.xx.xx" --gpus '0,1,2,3' tools/train.py -c configs/table/SLANet.yml \
     -o Global.pretrained_model=./pretrain_models/SLANet/best_accuracy
```


**Note:** (1) When using multi-machine and multi-gpu training, you need to replace the ips value in the above command with the address of your machine, and the machines need to be able to ping each other. (2) Training needs to be launched separately on multiple machines. The command to view the ip address of the machine is `ifconfig`. (3) For more details about the distributed training speedup ratio, please refer to [Distributed Training Tutorial](./distributed_training_en.md).

## 2.6. Training on other platform(Windows/macOS/Linux DCU)

- Windows GPU/CPU
The Windows platform is slightly different from the Linux platform:
Windows platform only supports `single gpu` training and inference, specify GPU for training `set CUDA_VISIBLE_DEVICES=0`
On the Windows platform, DataLoader only supports single-process mode, so you need to set `num_workers` to 0;

- macOS
GPU mode is not supported, you need to set `use_gpu` to False in the configuration file, and the rest of the training evaluation prediction commands are exactly the same as Linux GPU.

- Linux DCU
Running on a DCU device requires setting the environment variable `export HIP_VISIBLE_DEVICES=0,1,2,3`, and the rest of the training and evaluation prediction commands are exactly the same as the Linux GPU.


## 2.7. Fine-tuning


### 2.7.1 Dataset

Data number: It is recommended to prepare at least 2000 table recognition datasets for model fine-tuning.

### 2.7.2 model selection

It is recommended to choose the SLANet model (configuration file: [SLANet_ch.yml](../../configs/table/SLANet_ch.yml), pre-training model: [ch_ppstructure_mobile_v2.0_SLANet_train.tar](https://paddleocr.bj.bcebos .com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_train.tar)) for fine-tuning, its accuracy and generalization performance is the best Chinese table pre-training model currently available.

For more table recognition models, please refer to [PP-Structure Series Model Library](../../ppstructure/docs/models_list.md).

### 2.7.3 Training hyperparameter selection

When fine-tuning the model, the most important hyperparameters are the pretrained model path `pretrained_model`, the learning rate `learning_rate`, and some configuration files are shown below.

```yaml
Global:
  pretrained_model: ./ch_ppstructure_mobile_v2.0_SLANet_train/best_accuracy.pdparams # Pre-trained model path
Optimizer:
  lr:
    name: Cosine
    learning_rate: 0.001 #
    warmup_epoch: 0
  regularizer:
    name: 'L2'
    factor: 0
```

In the above configuration file, you first need to specify the `pretrained_model` field as the `best_accuracy.pdparams` file path.

The configuration file provided by PaddleOCR is for 4-card training (equivalent to a total batch size of `4*48=192`) and no pre-trained model is loaded. Therefore, in your scenario, the learning rate is the same as the total The batch size needs to be adjusted linearly, for example

* If your scenario is single card training, single card batch_size=48, then the total batch_size=48, it is recommended to adjust the learning rate to about `0.00025`.
* If your scenario is for single-card training, due to memory limitations, you can only set batch_size=32 for a single card, then the total batch_size=32, it is recommended to adjust the learning rate to about `0.00017`.

# 3. Evaluation and Test

## 3.1. Evaluation

The model parameters during training are saved in the `Global.save_model_dir` directory by default. When evaluating metrics, you need to set `Global.checkpoints` to point to the saved parameter file. Evaluation datasets can be modified via the `label_file_list` setting in Eval via `configs/table/SLANet.yml`.

```
# GPU evaluation, Global.checkpoints is the weight to be tested
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/table/SLANet.yml -o Global.checkpoints={path/to/weights}/best_accuracy
```

After the operation is completed, the acc indicator of the model will be output. If you evaluate the English table recognition model, you will see the following output.

```bash
[2022/08/16 07:59:55] ppocr INFO: acc:0.7622245132160782
[2022/08/16 07:59:55] ppocr INFO: fps:30.991640622573044
```

## 3.2. Test table structure recognition effect

Using the model trained by PaddleOCR, you can quickly get prediction through the following script.

The default prediction picture is stored in `infer_img`, and the trained weight is specified via `-o Global.checkpoints`:


According to the `save_model_dir` and `save_epoch_step` fields set in the configuration file, the following parameters will be saved:


```
output/SLANet/
├── best_accuracy.pdopt  
├── best_accuracy.pdparams  
├── best_accuracy.states  
├── config.yml  
├── latest.pdopt  
├── latest.pdparams  
├── latest.states  
└── train.log
```
Among them, best_accuracy.* is the best model on the evaluation set; latest.* is the model of the last epoch.

```
# Predict table image
python3 tools/infer_table.py -c configs/table/SLANet.yml -o Global.pretrained_model={path/to/weights}/best_accuracy  Global.infer_img=ppstructure/docs/table/table.jpg
```

Input image:

![](../../ppstructure/docs/table/table.jpg)

Get the prediction result of the input image:

```
['<html>', '<body>', '<table>', '<thead>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '</thead>', '<tbody>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '</tbody>', '</table>', '</body>', '</html>'],[[320.0562438964844, 197.83375549316406, 350.0928955078125, 214.4309539794922], ... , [318.959228515625, 271.0166931152344, 353.7394104003906, 286.4538269042969]]
```

The cell coordinates are visualized as

![](../../ppstructure/docs/imgs/slanet_result.jpg)

# 4. Model export and prediction

## 4.1 Model export

inference model (model saved by `paddle.jit.save`)
Generally, it is model training, a solidified model that saves the model structure and model parameters in a file, and is mostly used to predict deployment scenarios.
The model saved during the training process is the checkpoints model, and only the parameters of the model are saved, which are mostly used to resume training.
Compared with the checkpoints model, the inference model will additionally save the structural information of the model. It has superior performance in predicting deployment and accelerating reasoning, and is flexible and convenient, and is suitable for actual system integration.

The way to convert the form recognition model to the inference model is the same as the text detection and recognition, as follows:

```
# -c Set the training algorithm yml configuration file
# -o Set optional parameters
# Global.pretrained_model parameter Set the training model address to be converted without adding the file suffix .pdmodel, .pdopt or .pdparams.
# Global.save_inference_dir Set the address where the converted model will be saved.

python3 tools/export_model.py -c configs/table/SLANet.yml -o Global.pretrained_model=./pretrain_models/SLANet/best_accuracy  Global.save_inference_dir=./inference/SLANet/
```

After the conversion is successful, there are three files in the model save directory:


```
inference/SLANet/
    ├── inference.pdiparams         # The parameter file of inference model
    ├── inference.pdiparams.info    # The parameter information of inference model, which can be ignored
    └── inference.pdmodel           # The program file of model
```

## 4.2 Prediction

After the model is exported, use the following command to complete the prediction of the inference model

```python
python3.7 table/predict_structure.py \
    --table_model_dir={path/to/inference model} \
    --table_char_dict_path=../ppocr/utils/dict/table_structure_dict_ch.txt \
    --image_dir=docs/table/table.jpg \
    --output=../output/table
```

Input image:

![](../../ppstructure/docs/table/table.jpg)

Get the prediction result of the input image:

```
['<html>', '<body>', '<table>', '<thead>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '</thead>', '<tbody>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '</tbody>', '</table>', '</body>', '</html>'],[[320.0562438964844, 197.83375549316406, 350.0928955078125, 214.4309539794922], ... , [318.959228515625, 271.0166931152344, 353.7394104003906, 286.4538269042969]]
```

The cell coordinates are visualized as

![](../../ppstructure/docs/imgs/slanet_result.jpg)



# 5. FAQ

Q1: After the training model is transferred to the inference model, the prediction effect is inconsistent?

**A**: There are many such problems, and the problems are mostly caused by inconsistent preprocessing and postprocessing parameters when the trained model predicts and the preprocessing and postprocessing parameters when the inference model predicts. You can compare whether there are differences in preprocessing, postprocessing, and prediction in the configuration files used for training.
