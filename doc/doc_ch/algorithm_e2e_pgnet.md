# 端对端OCR算法-PGNet
- [一、简介](#简介)
- [二、环境配置](#环境配置)
- [三、快速使用](#快速使用)
- [四、模型训练、评估、推理](#模型训练、评估、推理)

<a name="简介"></a>
## 一、简介
OCR算法可以分为两阶段算法和端对端的算法。二阶段OCR算法一般分为两个部分，文本检测和文本识别算法，文件检测算法从图像中得到文本行的检测框，然后识别算法去识别文本框中的内容。而端对端OCR算法可以在一个算法中完成文字检测和文字识别，其基本思想是设计一个同时具有检测单元和识别模块的模型，共享其中两者的CNN特征，并联合训练。由于一个算法即可完成文字识别，端对端模型更小，速度更快。

### PGNet算法介绍
近些年来，端对端OCR算法得到了良好的发展，包括MaskTextSpotter系列、TextSnake、TextDragon、PGNet系列等算法。在这些算法中，PGNet算法具备其他算法不具备的优势，包括：
- 设计PGNet loss指导训练，不需要字符级别的标注
- 不需要NMS和ROI相关操作，加速预测
- 提出预测文本行内的阅读顺序模块；
- 提出基于图的修正模块（GRM）来进一步提高模型识别性能
- 精度更高，预测速度更快

PGNet算法细节详见[论文](https://www.aaai.org/AAAI21Papers/AAAI-2885.WangP.pdf) ,算法原理图如下所示：
![](../pgnet_framework.png)
输入图像经过特征提取送入四个分支，分别是：文本边缘偏移量预测TBO模块，文本中心线预测TCL模块，文本方向偏移量预测TDO模块，以及文本字符分类图预测TCC模块。
其中TBO以及TCL的输出经过后处理后可以得到文本的检测结果，TCL、TDO、TCC负责文本识别。

其检测识别效果图如下：

![](../imgs_results/e2e_res_img293_pgnet.png)
![](../imgs_results/e2e_res_img295_pgnet.png)

### 性能指标

#### 测试集: Total Text

#### 测试环境: NVIDIA Tesla V100-SXM2-16GB

|PGNetA|det_precision|det_recall|det_f_score|e2e_precision|e2e_recall|e2e_f_score|FPS|下载|
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|Paper|85.30|86.80|86.10|-|-|61.70|38.20 (size=640)|-|
|Ours|87.03|82.48|84.69|61.71|58.43|60.03|48.73 (size=768)|[下载链接](https://paddleocr.bj.bcebos.com/dygraph_v2.0/pgnet/en_server_pgnetA.tar)|

*note：PaddleOCR里的PGNet实现针对预测速度做了优化，在精度下降可接受范围内，可以显著提升端对端预测速度*



<a name="环境配置"></a>
## 二、环境配置
请先参考[《运行环境准备》](./environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](./clone.md)克隆项目

<a name="快速使用"></a>
## 三、快速使用
### inference模型下载
本节以训练好的端到端模型为例，快速使用模型预测，首先下载训练好的端到端inference模型[下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.0/pgnet/e2e_server_pgnetA_infer.tar)
```
mkdir inference && cd inference
# 下载英文端到端模型并解压
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/pgnet/e2e_server_pgnetA_infer.tar && tar xf e2e_server_pgnetA_infer.tar
```
* windows 环境下如果没有安装wget,下载模型时可将链接复制到浏览器中下载，并解压放置在相应目录下

解压完毕后应有如下文件结构：
```
├── e2e_server_pgnetA_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```
### 单张图像或者图像集合预测
```bash
# 预测image_dir指定的单张图像
python3 tools/infer/predict_e2e.py --e2e_algorithm="PGNet" --image_dir="./doc/imgs_en/img623.jpg" --e2e_model_dir="./inference/e2e_server_pgnetA_infer/" --e2e_pgnet_valid_set="totaltext"

# 预测image_dir指定的图像集合
python3 tools/infer/predict_e2e.py --e2e_algorithm="PGNet" --image_dir="./doc/imgs_en/" --e2e_model_dir="./inference/e2e_server_pgnetA_infer/" --e2e_pgnet_valid_set="totaltext"

# 如果想使用CPU进行预测，需设置use_gpu参数为False
python3 tools/infer/predict_e2e.py --e2e_algorithm="PGNet" --image_dir="./doc/imgs_en/img623.jpg" --e2e_model_dir="./inference/e2e_server_pgnetA_infer/" --e2e_pgnet_valid_set="totaltext" --use_gpu=False
```
### 可视化结果
可视化文本检测结果默认保存到./inference_results文件夹里面，结果文件的名称前缀为'e2e_res'。结果示例如下：
![](../imgs_results/e2e_res_img623_pgnet.jpg)

<a name="模型训练、评估、推理"></a>
## 四、模型训练、评估、推理
本节以totaltext数据集为例，介绍PaddleOCR中端到端模型的训练、评估与测试。

###  准备数据
下载解压[totaltext](https://paddleocr.bj.bcebos.com/dataset/total_text.tar) 数据集到PaddleOCR/train_data/目录，数据集组织结构：
```
/PaddleOCR/train_data/total_text/train/
  |- rgb/            # total_text数据集的训练数据
      |- img11.jpg
      | ...  
  |- train.txt       # total_text数据集的训练标注
```

train.txt标注文件格式如下，文件名和标注信息中间用"\t"分隔：
```
" 图像文件名                    json.dumps编码的图像标注信息"
rgb/img11.jpg    [{"transcription": "ASRAMA", "points": [[214.0, 325.0], [235.0, 308.0], [259.0, 296.0], [286.0, 291.0], [313.0, 295.0], [338.0, 305.0], [362.0, 320.0], [349.0, 347.0], [330.0, 337.0], [310.0, 329.0], [290.0, 324.0], [269.0, 328.0], [249.0, 336.0], [231.0, 346.0]]}, {...}]
```
json.dumps编码前的图像标注信息是包含多个字典的list，字典中的 `points` 表示文本框的四个点的坐标(x, y)，从左上角的点开始顺时针排列。
`transcription` 表示当前文本框的文字，**当其内容为“###”时，表示该文本框无效，在训练时会跳过。**
如果您想在其他数据集上训练，可以按照上述形式构建标注文件。

### 启动训练

PGNet训练分为两个步骤：step1: 在合成数据上训练，得到预训练模型，此时模型精度依然较低；step2: 加载预训练模型，在totaltext数据集上训练；为快速训练，我们直接提供了step1的预训练模型。
```shell
cd PaddleOCR/
下载step1 预训练模型
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/pgnet/train_step1.tar
可以得到以下的文件格式
./pretrain_models/train_step1/
  └─ best_accuracy.pdopt
  └─ best_accuracy.states
  └─ best_accuracy.pdparams
```
*如果您安装的是cpu版本，请将配置文件中的 `use_gpu` 字段修改为false*

```shell
# 单机单卡训练 e2e 模型
python3 tools/train.py -c configs/e2e/e2e_r50_vd_pg.yml -o Global.pretrained_model=./pretrain_models/train_step1/best_accuracy Global.load_static_weights=False
# 单机多卡训练，通过 --gpus 参数设置使用的GPU ID
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/e2e/e2e_r50_vd_pg.yml -o Global.pretrained_model=./pretrain_models/train_step1/best_accuracy  Global.load_static_weights=False
```

上述指令中，通过-c 选择训练使用configs/e2e/e2e_r50_vd_pg.yml配置文件。
有关配置文件的详细解释，请参考[链接](./config.md)。

您也可以通过-o参数在不需要修改yml文件的情况下，改变训练的参数，比如，调整训练的学习率为0.0001
```shell
python3 tools/train.py -c configs/e2e/e2e_r50_vd_pg.yml -o Optimizer.base_lr=0.0001
```

#### 断点训练
如果训练程序中断，如果希望加载训练中断的模型从而恢复训练，可以通过指定Global.checkpoints指定要加载的模型路径：
```shell
python3 tools/train.py -c configs/e2e/e2e_r50_vd_pg.yml -o Global.checkpoints=./your/trained/model
```

**注意**：`Global.checkpoints`的优先级高于`Global.pretrain_weights`的优先级，即同时指定两个参数时，优先加载`Global.checkpoints`指定的模型，如果`Global.checkpoints`指定的模型路径有误，会加载`Global.pretrain_weights`指定的模型。

PaddleOCR计算三个OCR端到端相关的指标，分别是：Precision、Recall、Hmean。

运行如下代码，根据配置文件`e2e_r50_vd_pg.yml`中`save_res_path`指定的测试集检测结果文件，计算评估指标。

评估时设置后处理参数`max_side_len=768`，使用不同数据集、不同模型训练，可调整参数进行优化
训练中模型参数默认保存在`Global.save_model_dir`目录下。在评估指标时，需要设置`Global.checkpoints`指向保存的参数文件。
```shell
python3 tools/eval.py -c configs/e2e/e2e_r50_vd_pg.yml  -o Global.checkpoints="{path/to/weights}/best_accuracy"
```

### 模型预测
测试单张图像的端到端识别效果
```shell
python3 tools/infer_e2e.py -c configs/e2e/e2e_r50_vd_pg.yml -o Global.infer_img="./doc/imgs_en/img_10.jpg" Global.pretrained_model="./output/e2e_pgnet/best_accuracy" Global.load_static_weights=false
```

测试文件夹下所有图像的端到端识别效果
```shell
python3 tools/infer_e2e.py -c configs/e2e/e2e_r50_vd_pg.yml -o Global.infer_img="./doc/imgs_en/" Global.pretrained_model="./output/e2e_pgnet/best_accuracy" Global.load_static_weights=false
```

### 预测推理
#### (1). 四边形文本检测模型（ICDAR2015）  
首先将PGNet端到端训练过程中保存的模型，转换成inference model。以基于Resnet50_vd骨干网络，以英文数据集训练的模型为例[模型下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.0/pgnet/en_server_pgnetA.tar) ，可以使用如下命令进行转换：
```
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/pgnet/en_server_pgnetA.tar && tar xf en_server_pgnetA.tar
python3 tools/export_model.py -c configs/e2e/e2e_r50_vd_pg.yml -o Global.pretrained_model=./en_server_pgnetA/best_accuracy Global.load_static_weights=False Global.save_inference_dir=./inference/e2e
```
**PGNet端到端模型推理，需要设置参数`--e2e_algorithm="PGNet"` and `--e2e_pgnet_valid_set="partvgg"`**，可以执行如下命令：
```
python3 tools/infer/predict_e2e.py --e2e_algorithm="PGNet" --image_dir="./doc/imgs_en/img_10.jpg" --e2e_model_dir="./inference/e2e/"  --e2e_pgnet_valid_set="partvgg" --e2e_pgnet_valid_set="totaltext"
```
可视化文本检测结果默认保存到`./inference_results`文件夹里面，结果文件的名称前缀为'e2e_res'。结果示例如下：

![](../imgs_results/e2e_res_img_10_pgnet.jpg)

#### (2). 弯曲文本检测模型（Total-Text）
对于弯曲文本样例

**PGNet端到端模型推理，需要设置参数`--e2e_algorithm="PGNet"`，同时，还需要增加参数`--e2e_pgnet_valid_set="totaltext"`，**可以执行如下命令：
```
python3 tools/infer/predict_e2e.py --e2e_algorithm="PGNet" --image_dir="./doc/imgs_en/img623.jpg" --e2e_model_dir="./inference/e2e/" --e2e_pgnet_valid_set="totaltext"
```
可视化文本端到端结果默认保存到`./inference_results`文件夹里面，结果文件的名称前缀为'e2e_res'。结果示例如下：

![](../imgs_results/e2e_res_img623_pgnet.jpg)
