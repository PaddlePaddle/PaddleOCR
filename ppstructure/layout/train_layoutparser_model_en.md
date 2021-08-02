# Training layout-parse 

[1. Installation](#Installation) 

​  [1.1 Requirements](#Requirements) 

​  [1.2 Install PaddleDetection](#Install PaddleDetection)

[2.  Data preparation](#Data preparation) 

[3. Configuration](#Configuration) 

[4. Training](#Training) 

[5. Prediction](#Prediction) 

[6. Deployment](#Deployment) 

​  [6.1 Export model](#Export model) 

​  [6.2 Inference](#Inference)  

<a name="Installation"></a>

## 1.  Installation

<a name="Requirements"></a>

### 1.1 Requirements

- PaddlePaddle 2.1
- OS 64 bit
- Python 3(3.5.1+/3.6/3.7/3.8/3.9)，64 bit
- pip/pip3(9.0.1+), 64 bit
- CUDA >= 10.1
- cuDNN >= 7.6

<a name="Install PaddleDetection"></a>

### 1.2 Install PaddleDetection

```bash
# Clone PaddleDetection repository
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git

cd PaddleDetection
# Install other dependencies
pip install -r requirements.txt
```

For more installation tutorials, please refer to： [Install doc](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/docs/tutorials/INSTALL_cn.md)

<a name="Data preparation"></a>

## 2. Data preparation

Download the [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) dataset

```bash
cd PaddleDetection/dataset/
mkdir publaynet
# execute the command，download PubLayNet
wget -O publaynet.tar.gz https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz?_ga=2.104193024.1076900768.1622560733-649911202.1622560733
# unpack
tar -xvf publaynet.tar.gz
```

PubLayNet directory structure after decompressing ：

| File or Folder | Description                                      | num     |
| :------------- | :----------------------------------------------- | ------- |
| `train/`       | Images in the training subset                    | 335,703 |
| `val/`         | Images in the validation subset                  | 11,245  |
| `test/`        | Images in the testing subset                     | 11,405  |
| `train.json`   | Annotations for training images                  |  1       |
| `val.json`     | Annotations for validation images                |  1       |
| `LICENSE.txt`  | Plaintext version of the CDLA-Permissive license |   1      |
| `README.txt`   | Text file with the file names and description    |   1      |

For other datasets，please refer to [the PrepareDataSet]((https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/docs/tutorials/PrepareDataSet.md) )

<a name="Configuration"></a>

## 3. Configuration 

We use the  `configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml` configuration for training，the configuration file is as follows 

```bash
_BASE_: [
  '../datasets/coco_detection.yml',
  '../runtime.yml',
  './_base_/ppyolov2_r50vd_dcn.yml',
  './_base_/optimizer_365e.yml',
  './_base_/ppyolov2_reader.yml',
]

snapshot_epoch: 8
weights: output/ppyolov2_r50vd_dcn_365e_coco/model_final
```
The `ppyolov2_r50vd_dcn_365e_coco.yml` configuration depends on other configuration files, in this case:

- coco_detection.yml：mainly explains the path of training data and verification data

- runtime.yml：mainly describes the common parameters, such as whether to use the GPU and how many epoch to save model etc.

- optimizer_365e.yml：mainly explains the learning rate and optimizer configuration

- ppyolov2_r50vd_dcn.yml：mainly describes the model and the  network

- ppyolov2_reader.yml：mainly describes the configuration of data readers, such as batch size and number of concurrent loading child processes, and also includes post preprocessing, such as resize and data augmention etc.


Modify the preceding files, such as the dataset path and batch size etc.

<a name="Training"></a>

## 4. Training

PaddleDetection provides single-card/multi-card training mode to meet various training needs of users:

* GPU single card training

```bash
export CUDA_VISIBLE_DEVICES=0 #Don't need to run this command on Windows and Mac
python tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml
```

* GPU multi-card training

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --eval
```

--eval: training while verifying

* Model recovery training

During the daily training, if training is interrupted due to some reasons, you can use the -r command to resume the training:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --eval -r output/ppyolov2_r50vd_dcn_365e_coco/10000
```

Note: If you encounter "`Out of memory error`" , try reducing `batch_size` in the `ppyolov2_reader.yml`  file

prediction<a name="Prediction"></a>

## 5. Prediction

Set parameters and use PaddleDetection to predict：

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --infer_img=images/paper-image.jpg --output_dir=infer_output/ --draw_threshold=0.5 -o weights=output/ppyolov2_r50vd_dcn_365e_coco/model_final --use_vdl=Ture
```

`--draw_threshold` is an optional parameter. According to the calculation of [NMS](https://ieeexplore.ieee.org/document/1699659), different threshold will produce different results, ` keep_top_k ` represent  the maximum amount of output target, the default value is 10. You can set different value according to your own actual situation。

<a name="Deployment"></a>

## 6. Deployment

Use your trained model in Layout Parser

<a name="Export model"></a>

### 6.1 Export model

n the process of model training, the model file saved contains the process of forward prediction and back propagation. In the actual industrial deployment, there is no need for back propagation. Therefore, the model should be translated into the model format required by the deployment. The `tools/export_model.py` script is provided in PaddleDetection to export the model.

The exported model name defaults to `model.*`, Layout Parser's code model is `inference.*`, So change [PaddleDetection/ppdet/engine/trainer. Py ](https://github.com/PaddlePaddle/PaddleDetection/blob/b87a1ea86fa18ce69e44a17ad1b49c1326f19ff9/ppdet/engine/trainer.py# L512) (click on the link to see the detailed line of code), change 'model' to 'inference'.

Execute the script to export model:

```bash
python tools/export_model.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --output_dir=./inference -o weights=output/ppyolov2_r50vd_dcn_365e_coco/model_final.pdparams
```

The prediction model is exported to `inference/ppyolov2_r50vd_dcn_365e_coco` ,including:`infer_cfg.yml`(prediction not required), `inference.pdiparams`, `inference.pdiparams.info`,`inference.pdmodel`

More model export tutorials, please refer to：[EXPORT_MODEL](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/deploy/EXPORT_MODEL.md)

<a name="Inference"></a>

### 6.2 Inference

`model_path` represent  the trained model path, and layoutparser is used to predict:

```bash
import layoutparser as lp
model = lp.PaddleDetectionLayoutModel(model_path="inference/ppyolov2_r50vd_dcn_365e_coco", threshold=0.5,label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},enforce_cpu=True,enable_mkldnn=True)
```



***

More PaddleDetection training tutorials，please reference：[PaddleDetection Training](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/docs/tutorials/GETTING_STARTED_cn.md) 

***

