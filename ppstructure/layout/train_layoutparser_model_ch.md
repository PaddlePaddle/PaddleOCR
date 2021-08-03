# 训练版面分析

[1. 安装](#安装)

​        [1.1 环境要求](#环境要求)

​        [1.2 安装PaddleDetection](#安装PaddleDetection)

[2. 准备数据](#准备数据)

[3. 配置文件改动和说明](#配置文件改动和说明)

[4. PaddleDetection训练](#训练)

[5. PaddleDetection预测](#预测)

[6. 预测部署](#预测部署)

​        [6.1 模型导出](#模型导出)

​        [6.2 layout parser预测](#layout_parser预测)

<a name="安装"></a>

## 1. 安装

<a name="环境要求"></a>

### 1.1 环境要求

- PaddlePaddle 2.1
- OS 64 bit
- Python 3(3.5.1+/3.6/3.7/3.8/3.9)，64 bit
- pip/pip3(9.0.1+), 64 bit
- CUDA >= 10.1
- cuDNN >= 7.6

<a name="安装PaddleDetection"></a>

### 1.2 安装PaddleDetection

```bash
# 克隆PaddleDetection仓库
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git

cd PaddleDetection
# 安装其他依赖
pip install -r requirements.txt
```

更多安装教程，请参考: [Install doc](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/docs/tutorials/INSTALL_cn.md)

<a name="数据准备"></a>

## 2. 准备数据

下载 [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) 数据集：

```bash
cd PaddleDetection/dataset/
mkdir publaynet
# 执行命令，下载
wget -O publaynet.tar.gz https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz?_ga=2.104193024.1076900768.1622560733-649911202.1622560733
# 解压
tar -xvf publaynet.tar.gz
```

解压之后PubLayNet目录结构：

| File or Folder | Description                                      | num     |
| :------------- | :----------------------------------------------- | ------- |
| `train/`       | Images in the training subset                    | 335,703 |
| `val/`         | Images in the validation subset                  | 11,245  |
| `test/`        | Images in the testing subset                     | 11,405  |
| `train.json`   | Annotations for training images                  | 1       |
| `val.json`     | Annotations for validation images                | 1       |
| `LICENSE.txt`  | Plaintext version of the CDLA-Permissive license | 1       |
| `README.txt`   | Text file with the file names and description    | 1       |

如果使用其它数据集，请参考[准备训练数据](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/docs/tutorials/PrepareDataSet.md)

<a name="配置文件改动和说明"></a>

## 3. 配置文件改动和说明

我们使用 `configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml`配置进行训练，配置文件摘要如下：

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
从中可以看到 `ppyolov2_r50vd_dcn_365e_coco.yml` 配置需要依赖其他的配置文件，在该例子中需要依赖:

- coco_detection.yml：主要说明了训练数据和验证数据的路径

- runtime.yml：主要说明了公共的运行参数，比如是否使用GPU、每多少个epoch存储checkpoint等

- optimizer_365e.yml：主要说明了学习率和优化器的配置

- ppyolov2_r50vd_dcn.yml：主要说明模型和主干网络的情况

- ppyolov2_reader.yml：主要说明数据读取器配置，如batch size，并发加载子进程数等，同时包含读取后预处理操作，如resize、数据增强等等


根据实际情况，修改上述文件，比如数据集路径、batch size等。

<a name="训练"></a>

## 4. PaddleDetection训练

PaddleDetection提供了单卡/多卡训练模式，满足用户多种训练需求

* GPU 单卡训练

```bash
export CUDA_VISIBLE_DEVICES=0 #windows和Mac下不需要执行该命令
python tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml
```

* GPU多卡训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --eval
```

--eval：表示边训练边验证

* 模型恢复训练

在日常训练过程中，有的用户由于一些原因导致训练中断，用户可以使用-r的命令恢复训练:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --eval -r output/ppyolov2_r50vd_dcn_365e_coco/10000
```

注意：如果遇到 "`Out of memory error`" 问题, 尝试在 `ppyolov2_reader.yml` 文件中调小`batch_size`

<a name="预测"></a>

## 5. PaddleDetection预测

设置参数，使用PaddleDetection预测：

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --infer_img=images/paper-image.jpg --output_dir=infer_output/ --draw_threshold=0.5 -o weights=output/ppyolov2_r50vd_dcn_365e_coco/model_final --use_vdl=Ture
```

`--draw_threshold` 是个可选参数. 根据 [NMS](https://ieeexplore.ieee.org/document/1699659) 的计算，不同阈值会产生不同的结果 `keep_top_k`表示设置输出目标的最大数量，默认值为100，用户可以根据自己的实际情况进行设定。

<a name="预测部署"></a>

## 6. 预测部署

在layout parser中使用自己训练好的模型。

<a name="模型导出"></a>

### 6.1 模型导出

在模型训练过程中保存的模型文件是包含前向预测和反向传播的过程，在实际的工业部署则不需要反向传播，因此需要将模型进行导成部署需要的模型格式。 在PaddleDetection中提供了 `tools/export_model.py`脚本来导出模型。

导出模型名称默认是`model.*`，layout parser代码模型名称是`inference.*`,  所以修改[PaddleDetection/ppdet/engine/trainer.py ](https://github.com/PaddlePaddle/PaddleDetection/blob/b87a1ea86fa18ce69e44a17ad1b49c1326f19ff9/ppdet/engine/trainer.py#L512) (点开链接查看详细代码行)，将`model`改为`inference`即可。

执行导出模型脚本：

```bash
python tools/export_model.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --output_dir=./inference -o weights=output/ppyolov2_r50vd_dcn_365e_coco/model_final.pdparams
```

预测模型会导出到`inference/ppyolov2_r50vd_dcn_365e_coco`目录下，分别为`infer_cfg.yml`(预测不需要), `inference.pdiparams`, `inference.pdiparams.info`,`inference.pdmodel` 。

更多模型导出教程，请参考：[EXPORT_MODEL](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/deploy/EXPORT_MODEL.md)

<a name="layout parser预测"></a>

### 6.2 layout_parser预测

`model_path`指定训练好的模型路径，使用layout parser进行预测：

```bash
import layoutparser as lp
model = lp.PaddleDetectionLayoutModel(model_path="inference/ppyolov2_r50vd_dcn_365e_coco", threshold=0.5,label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},enforce_cpu=True,enable_mkldnn=True)
```



***

更多PaddleDetection训练教程，请参考：[PaddleDetection训练](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/docs/tutorials/GETTING_STARTED_cn.md)

***
