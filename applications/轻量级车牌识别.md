# 一种基于PaddleOCR的轻量级车牌识别模型

- [1. 项目介绍](#1-项目介绍)
- [2. 环境搭建](#2-环境搭建)
- [3. 数据集准备](#3-数据集准备)
  - [3.1 数据集标注规则](#31-数据集标注规则)
  - [3.2 制作符合PP-OCR训练格式的标注文件](#32-制作符合pp-ocr训练格式的标注文件)
- [4. 实验](#4-实验)
  - [4.1 检测](#41-检测)
    - [4.1.1 预训练模型直接预测](#411-预训练模型直接预测)
    - [4.1.2 CCPD车牌数据集fine-tune](#412-ccpd车牌数据集fine-tune)
    - [4.1.3 CCPD车牌数据集fine-tune+量化训练](#413-ccpd车牌数据集fine-tune量化训练)
    - [4.1.4 模型导出](#414-模型导出)
  - [4.2 识别](#42-识别)
    - [4.2.1 预训练模型直接预测](#421-预训练模型直接预测)
    - [4.2.2 预训练模型直接预测+改动后处理](#422-预训练模型直接预测改动后处理)
    - [4.2.3 CCPD车牌数据集fine-tune](#423-ccpd车牌数据集fine-tune)
    - [4.2.4 CCPD车牌数据集fine-tune+量化训练](#424-ccpd车牌数据集fine-tune量化训练)
    - [4.2.5 模型导出](#425-模型导出)
  - [4.3 计算End2End指标](#43-计算End2End指标)
  - [4.4 部署](#44-部署)
  - [4.5 实验总结](#45-实验总结)

## 1. 项目介绍

车牌识别(Vehicle License Plate Recognition，VLPR) 是计算机视频图像识别技术在车辆牌照识别中的一种应用。车牌识别技术要求能够将运动中的汽车牌照从复杂背景中提取并识别出来，在高速公路车辆管理，停车场管理和城市交通中得到广泛应用。

本项目难点如下：

1. 车牌在图像中的尺度差异大、在车辆上的悬挂位置不固定
2. 车牌图像质量层次不齐: 角度倾斜、图片模糊、光照不足、过曝等问题严重
3. 边缘和端测场景应用对模型大小有限制，推理速度有要求

针对以上问题， 本例选用 PP-OCRv3 这一开源超轻量OCR系统进行车牌识别系统的开发。基于PP-OCRv3模型，在CCPD数据集达到99%的检测和94%的识别精度，模型大小12.8M(2.5M+10.3M)。基于量化对模型体积进行进一步压缩到5.8M(1M+4.8M), 同时推理速度提升25%。



aistudio项目链接: [基于PaddleOCR的轻量级车牌识别范例](https://aistudio.baidu.com/aistudio/projectdetail/3919091?contributionType=1)

## 2. 环境搭建

本任务基于Aistudio完成, 具体环境如下：

- 操作系统: Linux
- PaddlePaddle: 2.3
- paddleslim: 2.2.2
- PaddleOCR: Release/2.5

下载 PaddleOCR代码

```bash
git clone -b dygraph https://github.com/PaddlePaddle/PaddleOCR
```

安装依赖库

```bash
pip install -r PaddleOCR/requirements.txt
```

## 3. 数据集准备

所使用的数据集为 CCPD2020 新能源车牌数据集，该数据集为

该数据集分布如下：

|数据集类型|数量|
|---|---|
|训练集| 5769|
|验证集| 1001|
|测试集| 5006|

数据集图片示例如下:
![](https://ai-studio-static-online.cdn.bcebos.com/3bce057a8e0c40a0acbd26b2e29e4e2590a31bc412764be7b9e49799c69cb91c)

数据集可以从这里下载 https://aistudio.baidu.com/aistudio/datasetdetail/101595

下载好数据集后对数据集进行解压

```bash
unzip -d /home/aistudio/data /home/aistudio/data/data101595/CCPD2020.zip
```

### 3.1 数据集标注规则

CPPD数据集的图片文件名具有特殊规则，详细可查看：https://github.com/detectRecog/CCPD

具体规则如下：

例如: 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg

每个名称可以分为七个字段，以-符号作为分割。这些字段解释如下。

- 025：车牌面积与整个图片区域的面积比。025 (25%)

- 95_113：水平倾斜程度和垂直倾斜度。水平 95度 垂直 113度

- 154&383_386&473：左上和右下顶点的坐标。左上(154,383) 右下(386,473)

- 386&473_177&454_154&383_363&402：整个图像中车牌的四个顶点的精确（x，y）坐标。这些坐标从右下角顶点开始。(386,473) (177,454) (154,383) (363,402)

- 0_0_22_27_27_33_16：CCPD中的每个图像只有一个车牌。每个车牌号码由一个汉字，一个字母和五个字母或数字组成。有效的中文车牌由七个字符组成：省（1个字符），字母（1个字符），字母+数字（5个字符）。“ 0_0_22_27_27_33_16”是每个字符的索引。这三个数组定义如下。每个数组的最后一个字符是字母O，而不是数字0。我们将O用作“无字符”的符号，因为中文车牌字符中没有O。因此以上车牌拼起来即为 皖AY339S

- 37：牌照区域的亮度。 37 (37%)

- 15：车牌区域的模糊度。15 (15%)

```python
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W','X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X','Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
```

### 3.2 制作符合PP-OCR训练格式的标注文件

在开始训练之前，可使用如下代码制作符合PP-OCR训练格式的标注文件。


```python
import cv2
import os
import json
from tqdm import tqdm
import numpy as np

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def make_label(img_dir, save_gt_folder, phase):
    crop_img_save_dir = os.path.join(save_gt_folder, phase, 'crop_imgs')
    os.makedirs(crop_img_save_dir, exist_ok=True)

    f_det = open(os.path.join(save_gt_folder, phase, 'det.txt'), 'w', encoding='utf-8')
    f_rec = open(os.path.join(save_gt_folder, phase, 'rec.txt'), 'w', encoding='utf-8')

    i = 0
    for filename in tqdm(os.listdir(os.path.join(img_dir, phase))):
        str_list = filename.split('-')
        if len(str_list) < 5:
            continue
        coord_list = str_list[3].split('_')
        txt_list = str_list[4].split('_')
        boxes = []
        for coord in coord_list:
            boxes.append([int(x) for x in coord.split("&")])
        boxes = [boxes[2], boxes[3], boxes[0], boxes[1]]
        lp_number = provinces[int(txt_list[0])] + alphabets[int(txt_list[1])] + ''.join([ads[int(x)] for x in txt_list[2:]])

        # det
        det_info = [{'points':boxes, 'transcription':lp_number}]
        f_det.write('{}\t{}\n'.format(os.path.join(phase, filename), json.dumps(det_info, ensure_ascii=False)))

        # rec
        boxes = np.float32(boxes)
        img = cv2.imread(os.path.join(img_dir, phase, filename))
        # crop_img = img[int(boxes[:,1].min()):int(boxes[:,1].max()),int(boxes[:,0].min()):int(boxes[:,0].max())]
        crop_img = get_rotate_crop_image(img, boxes)
        crop_img_save_filename = '{}_{}.jpg'.format(i,'_'.join(txt_list))
        crop_img_save_path = os.path.join(crop_img_save_dir, crop_img_save_filename)
        cv2.imwrite(crop_img_save_path, crop_img)
        f_rec.write('{}/crop_imgs/{}\t{}\n'.format(phase, crop_img_save_filename, lp_number))
        i+=1
    f_det.close()
    f_rec.close()

def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

img_dir = '/home/aistudio/data/CCPD2020/ccpd_green'
save_gt_folder = '/home/aistudio/data/CCPD2020/PPOCR'
# phase = 'train' # change to val and test to make val dataset and test dataset
for phase in ['train','val','test']:
    make_label(img_dir, save_gt_folder, phase)
```

通过上述命令可以完成了`训练集`，`验证集`和`测试集`的制作，制作完成的数据集信息如下：

| 类型 | 数据集 | 图片地址 | 标签地址 | 图片数量 |
| --- | --- | --- | --- | --- |
| 检测 | 训练集 | /home/aistudio/data/CCPD2020/ccpd_green/train | /home/aistudio/data/CCPD2020/PPOCR/train/det.txt | 5769 |
| 检测 | 验证集 | /home/aistudio/data/CCPD2020/ccpd_green/val | /home/aistudio/data/CCPD2020/PPOCR/val/det.txt | 1001 |
| 检测 | 测试集 | /home/aistudio/data/CCPD2020/ccpd_green/test | /home/aistudio/data/CCPD2020/PPOCR/test/det.txt | 5006 |
| 识别 | 训练集 | /home/aistudio/data/CCPD2020/PPOCR/train/crop_imgs | /home/aistudio/data/CCPD2020/PPOCR/train/rec.txt | 5769 |
| 识别 | 验证集 | /home/aistudio/data/CCPD2020/PPOCR/val/crop_imgs | /home/aistudio/data/CCPD2020/PPOCR/val/rec.txt | 1001 |
| 识别 | 测试集 | /home/aistudio/data/CCPD2020/PPOCR/test/crop_imgs | /home/aistudio/data/CCPD2020/PPOCR/test/rec.txt | 5006 |

在普遍的深度学习流程中，都是在训练集训练，在验证集选择最优模型后在测试集上进行测试。在本例中，我们省略中间步骤，直接在训练集训练，在测试集选择最优模型，因此我们只使用训练集和测试集。

## 4. 实验

由于数据集比较少，为了模型更好和更快的收敛，这里选用 PaddleOCR 中的 PP-OCRv3 模型进行文本检测和识别，并且使用 PP-OCRv3 模型参数作为预训练模型。PP-OCRv3在PP-OCRv2的基础上，中文场景端到端Hmean指标相比于PP-OCRv2提升5%, 英文数字模型端到端效果提升11%。详细优化细节请参考[PP-OCRv3](../doc/doc_ch/PP-OCRv3_introduction.md)技术报告。

由于车牌场景均为端侧设备部署，因此对速度和模型大小有比较高的要求，因此还需要采用量化训练的方式进行模型大小的压缩和模型推理速度的加速。模型量化可以在基本不损失模型的精度的情况下，将FP32精度的模型参数转换为Int8精度，减小模型参数大小并加速计算，使用量化后的模型在移动端等部署时更具备速度优势。

因此，本实验中对于车牌检测和识别有如下3种方案：

1. PP-OCRv3中英文超轻量预训练模型直接预测
2. CCPD车牌数据集在PP-OCRv3模型上fine-tune
3. CCPD车牌数据集在PP-OCRv3模型上fine-tune后量化

### 4.1 检测
#### 4.1.1 预训练模型直接预测

从下表中下载PP-OCRv3文本检测预训练模型

|模型名称|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- | --- |
|ch_PP-OCRv3_det| 【最新】原始超轻量模型，支持中英文、多语种文本检测 |[ch_PP-OCRv3_det_cml.yml](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml)| 3.8M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar)|

使用如下命令下载预训练模型

```bash
mkdir models
cd models
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar
tar -xf ch_PP-OCRv3_det_distill_train.tar
cd /home/aistudio/PaddleOCR
```

预训练模型下载完成后，我们使用[ch_PP-OCRv3_det_student.yml](../configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml) 配置文件进行后续实验，在开始评估之前需要对配置文件中部分字段进行设置，具体如下：

1. 模型存储和训练相关:
   1. Global.pretrained_model: 指向PP-OCRv3文本检测预训练模型地址
2. 数据集相关
   1. Eval.dataset.data_dir：指向测试集图片存放目录
   2. Eval.dataset.label_file_list：指向测试集标注文件

上述字段均为必须修改的字段，可以通过修改配置文件的方式改动，也可在不需要修改配置文件的情况下，改变训练的参数。这里使用不改变配置文件的方式 。使用如下命令进行PP-OCRv3文本检测预训练模型的评估


```bash
python tools/eval.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o \
    Global.pretrained_model=models/ch_PP-OCRv3_det_distill_train/student.pdparams \
    Eval.dataset.data_dir=/home/aistudio/data/CCPD2020/ccpd_green \
    Eval.dataset.label_file_list=[/home/aistudio/data/CCPD2020/PPOCR/test/det.txt]
```
上述指令中，通过-c 选择训练使用配置文件，通过-o参数在不需要修改配置文件的情况下，改变训练的参数。

使用预训练模型进行评估，指标如下所示：

| 方案                        |hmeans|
|---------------------------|---|
| PP-OCRv3中英文超轻量检测预训练模型直接预测 |76.12%|

#### 4.1.2 CCPD车牌数据集fine-tune

**训练**

为了进行fine-tune训练，我们需要在配置文件中设置需要使用的预训练模型地址，学习率和数据集等参数。 具体如下:

1. 模型存储和训练相关:
   1. Global.pretrained_model: 指向PP-OCRv3文本检测预训练模型地址
   2. Global.eval_batch_step: 模型多少step评估一次，这里设为从第0个step开始没隔772个step评估一次，772为一个epoch总的step数。
2. 优化器相关:
   1. Optimizer.lr.name: 学习率衰减器设为常量 Const
   2. Optimizer.lr.learning_rate: 做 fine-tune 实验，学习率需要设置的比较小，此处学习率设为配置文件中的0.05倍
   3. Optimizer.lr.warmup_epoch: warmup_epoch设为0
3. 数据集相关:
   1. Train.dataset.data_dir：指向训练集图片存放目录
   2. Train.dataset.label_file_list：指向训练集标注文件
   3. Eval.dataset.data_dir：指向测试集图片存放目录
   4. Eval.dataset.label_file_list：指向测试集标注文件

使用如下代码即可启动在CCPD车牌数据集上的fine-tune。

```bash
python tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o \
    Global.pretrained_model=models/ch_PP-OCRv3_det_distill_train/student.pdparams \
    Global.save_model_dir=output/CCPD/det \
    Global.eval_batch_step="[0, 772]" \
    Optimizer.lr.name=Const \
    Optimizer.lr.learning_rate=0.0005 \
    Optimizer.lr.warmup_epoch=0 \
    Train.dataset.data_dir=/home/aistudio/data/CCPD2020/ccpd_green \
    Train.dataset.label_file_list=[/home/aistudio/data/CCPD2020/PPOCR/train/det.txt] \
    Eval.dataset.data_dir=/home/aistudio/data/CCPD2020/ccpd_green \
    Eval.dataset.label_file_list=[/home/aistudio/data/CCPD2020/PPOCR/test/det.txt]
```

在上述命令中，通过`-o`的方式修改了配置文件中的参数。


**评估**

训练完成后使用如下命令进行评估


```bash
python tools/eval.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o \
    Global.pretrained_model=output/CCPD/det/best_accuracy.pdparams \
    Eval.dataset.data_dir=/home/aistudio/data/CCPD2020/ccpd_green \
    Eval.dataset.label_file_list=[/home/aistudio/data/CCPD2020/PPOCR/test/det.txt]
```

使用预训练模型和CCPD车牌数据集fine-tune，指标分别如下：

|方案|hmeans|
|---|---|
|PP-OCRv3中英文超轻量检测预训练模型直接预测|76.12%|
|PP-OCRv3中英文超轻量检测预训练模型 fine-tune|99.00%|

可以看到进行fine-tune能显著提升车牌检测的效果。

#### 4.1.3 CCPD车牌数据集fine-tune+量化训练

此处采用 PaddleOCR 中提供好的[量化教程](../deploy/slim/quantization/README.md)对模型进行量化训练。

量化训练可通过如下命令启动:

```bash
python3.7 deploy/slim/quantization/quant.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o \
    Global.pretrained_model=output/CCPD/det/best_accuracy.pdparams \
    Global.save_model_dir=output/CCPD/det_quant \
    Global.eval_batch_step="[0, 772]" \
    Optimizer.lr.name=Const \
    Optimizer.lr.learning_rate=0.0005 \
    Optimizer.lr.warmup_epoch=0 \
    Train.dataset.data_dir=/home/aistudio/data/CCPD2020/ccpd_green \
    Train.dataset.label_file_list=[/home/aistudio/data/CCPD2020/PPOCR/train/det.txt] \
    Eval.dataset.data_dir=/home/aistudio/data/CCPD2020/ccpd_green \
    Eval.dataset.label_file_list=[/home/aistudio/data/CCPD2020/PPOCR/test/det.txt]
```

量化后指标对比如下

|方案|hmeans| 模型大小 | 预测速度(lite) |
|---|---|------|------------|
|PP-OCRv3中英文超轻量检测预训练模型 fine-tune|99.00%| 2.5M | 223ms      |
|PP-OCRv3中英文超轻量检测预训练模型 fine-tune+量化|98.91%| 1.0M   | 189ms      |

可以看到通过量化训练在精度几乎无损的情况下，降低模型体积60%并且推理速度提升15%。

速度测试基于[PaddleOCR lite教程](../deploy/lite/readme_ch.md)完成。

#### 4.1.4 模型导出

使用如下命令可以将训练好的模型进行导出

* 非量化模型
```bash
python tools/export_model.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o \
    Global.pretrained_model=output/CCPD/det/best_accuracy.pdparams \
    Global.save_inference_dir=output/det/infer
```
* 量化模型
```bash
python deploy/slim/quantization/export_model.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o \
    Global.pretrained_model=output/CCPD/det_quant/best_accuracy.pdparams \
    Global.save_inference_dir=output/det/infer
```

### 4.2 识别
#### 4.2.1 预训练模型直接预测

从下表中下载PP-OCRv3文本识别预训练模型

|模型名称|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- | --- |
|ch_PP-OCRv3_rec|【最新】原始超轻量模型，支持中英文、数字识别|[ch_PP-OCRv3_rec_distillation.yml](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml)| 12.4M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar) |

使用如下命令下载预训练模型

```bash
mkdir models
cd models
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar
tar -xf ch_PP-OCRv3_rec_train.tar
cd /home/aistudio/PaddleOCR
```

PaddleOCR提供的PP-OCRv3识别模型采用蒸馏训练策略，因此提供的预训练模型中会包含`Teacher`和`Student`模型的参数，详细信息可参考[knowledge_distillation.md](../doc/doc_ch/knowledge_distillation.md)。 因此，模型下载完成后需要使用如下代码提取`Student`模型的参数：

```python
import paddle
# 加载预训练模型
all_params = paddle.load("models/ch_PP-OCRv3_rec_train/best_accuracy.pdparams")
# 查看权重参数的keys
print(all_params.keys())
# 学生模型的权重提取
s_params = {key[len("Student."):]: all_params[key] for key in all_params if "Student." in key}
# 查看学生模型权重参数的keys
print(s_params.keys())
# 保存
paddle.save(s_params, "models/ch_PP-OCRv3_rec_train/student.pdparams")
```

预训练模型下载完成后，我们使用[ch_PP-OCRv3_rec.yml](../configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml) 配置文件进行后续实验，在开始评估之前需要对配置文件中部分字段进行设置，具体如下：

1. 模型存储和训练相关:
   1. Global.pretrained_model: 指向PP-OCRv3文本识别预训练模型地址
2. 数据集相关
   1. Eval.dataset.data_dir：指向测试集图片存放目录
   2. Eval.dataset.label_file_list：指向测试集标注文件

使用如下命令进行PP-OCRv3文本识别预训练模型的评估

```bash
python tools/eval.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o \
    Global.pretrained_model=models/ch_PP-OCRv3_rec_train/student.pdparams \
    Eval.dataset.data_dir=/home/aistudio/data/CCPD2020/PPOCR \
    Eval.dataset.label_file_list=[/home/aistudio/data/CCPD2020/PPOCR/test/rec.txt]
```

如需获取已训练模型，请加入PaddleX官方交流频道，获取20G OCR学习大礼包（内含《动手学OCR》电子书、课程回放视频、前沿论文等重磅资料）

- PaddleX官方交流频道：https://aistudio.baidu.com/community/channel/610



评估部分日志如下：
```bash
[2022/05/12 19:52:02] ppocr INFO: load pretrain successful from models/ch_PP-OCRv3_rec_train/best_accuracy
eval model:: 100%|██████████████████████████████| 40/40 [00:15<00:00,  2.57it/s]
[2022/05/12 19:52:17] ppocr INFO: metric eval ***************
[2022/05/12 19:52:17] ppocr INFO: acc:0.0
[2022/05/12 19:52:17] ppocr INFO: norm_edit_dis:0.8656084923002452
[2022/05/12 19:52:17] ppocr INFO: Teacher_acc:0.000399520574511545
[2022/05/12 19:52:17] ppocr INFO: Teacher_norm_edit_dis:0.8657902943394548
[2022/05/12 19:52:17] ppocr INFO: fps:1443.1801978719905

```
使用预训练模型进行评估，指标如下所示：

|方案|acc|
|---|---|
|PP-OCRv3中英文超轻量识别预训练模型直接预测|0%|

从评估日志中可以看到，直接使用PP-OCRv3预训练模型进行评估，acc非常低，但是norm_edit_dis很高。因此，我们猜测是模型大部分文字识别是对的，只有少部分文字识别错误。使用如下命令进行infer查看模型的推理结果进行验证：


```bash
python tools/infer_rec.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o \
    Global.pretrained_model=models/ch_PP-OCRv3_rec_train/student.pdparams \
    Global.infer_img=/home/aistudio/data/CCPD2020/PPOCR/test/crop_imgs/0_0_0_3_32_30_31_30_30.jpg
```

输出部分日志如下：
```bash
[2022/05/01 08:51:57] ppocr INFO: train with paddle 2.2.2 and device CUDAPlace(0)
W0501 08:51:57.127391 11326 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
W0501 08:51:57.132315 11326 device_context.cc:465] device: 0, cuDNN Version: 7.6.
[2022/05/01 08:52:00] ppocr INFO: load pretrain successful from models/ch_PP-OCRv3_rec_train/student
[2022/05/01 08:52:00] ppocr INFO: infer_img: /home/aistudio/data/CCPD2020/PPOCR/test/crop_imgs/0_0_3_32_30_31_30_30.jpg
[2022/05/01 08:52:00] ppocr INFO:      result: {"Student": {"label": "皖A·D86766", "score": 0.9552637934684753}, "Teacher": {"label": "皖A·D86766", "score": 0.9917094707489014}}
[2022/05/01 08:52:00] ppocr INFO: success!
```

从infer结果可以看到，车牌中的文字大部分都识别正确，只是多识别出了一个`·`。针对这种情况，有如下两种方案：
1. 直接通过后处理去掉多识别的`·`。
2. 进行 fine-tune。

#### 4.2.2 预训练模型直接预测+改动后处理

直接通过后处理去掉多识别的`·`，在后处理的改动比较简单，只需在 [ppocr/postprocess/rec_postprocess.py](../ppocr/postprocess/rec_postprocess.py) 文件的76行添加如下代码:
```python
text = text.replace('·','')
```

改动前后指标对比:

|方案|acc|
|---|---|
|PP-OCRv3中英文超轻量识别预训练模型直接预测|0.20%|
|PP-OCRv3中英文超轻量识别预训练模型直接预测+后处理去掉多识别的`·`|90.97%|

可以看到，去掉多余的`·`能大幅提高精度。

#### 4.2.3 CCPD车牌数据集fine-tune

**训练**

为了进行fine-tune训练，我们需要在配置文件中设置需要使用的预训练模型地址，学习率和数据集等参数。 具体如下:

1. 模型存储和训练相关:
   1. Global.pretrained_model: 指向PP-OCRv3文本识别预训练模型地址
   2. Global.eval_batch_step: 模型多少step评估一次，这里设为从第0个step开始没隔45个step评估一次，45为一个epoch总的step数。
2. 优化器相关
   1. Optimizer.lr.name: 学习率衰减器设为常量 Const
   2. Optimizer.lr.learning_rate: 做 fine-tune 实验，学习率需要设置的比较小，此处学习率设为配置文件中的0.05倍
   3. Optimizer.lr.warmup_epoch: warmup_epoch设为0
3. 数据集相关
   1. Train.dataset.data_dir：指向训练集图片存放目录
   2. Train.dataset.label_file_list：指向训练集标注文件
   3. Eval.dataset.data_dir：指向测试集图片存放目录
   4. Eval.dataset.label_file_list：指向测试集标注文件

使用如下命令启动 fine-tune

```bash
python tools/train.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o \
    Global.pretrained_model=models/ch_PP-OCRv3_rec_train/student.pdparams \
    Global.save_model_dir=output/CCPD/rec/ \
    Global.eval_batch_step="[0, 90]" \
    Optimizer.lr.name=Const \
    Optimizer.lr.learning_rate=0.0005 \
    Optimizer.lr.warmup_epoch=0 \
    Train.dataset.data_dir=/home/aistudio/data/CCPD2020/PPOCR \
    Train.dataset.label_file_list=[/home/aistudio/data/CCPD2020/PPOCR/train/rec.txt] \
    Eval.dataset.data_dir=/home/aistudio/data/CCPD2020/PPOCR \
    Eval.dataset.label_file_list=[/home/aistudio/data/CCPD2020/PPOCR/test/rec.txt]
```

**评估**

训练完成后使用如下命令进行评估

```bash
python tools/eval.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o \
    Global.pretrained_model=output/CCPD/rec/best_accuracy.pdparams \
    Eval.dataset.data_dir=/home/aistudio/data/CCPD2020/PPOCR \
    Eval.dataset.label_file_list=[/home/aistudio/data/CCPD2020/PPOCR/test/rec.txt]
```

使用预训练模型和CCPD车牌数据集fine-tune，指标分别如下：

|方案| acc    |
|---|--------|
|PP-OCRv3中英文超轻量识别预训练模型直接预测| 0.00%     |
|PP-OCRv3中英文超轻量识别预训练模型直接预测+后处理去掉多识别的`·`| 90.97% |
|PP-OCRv3中英文超轻量识别预训练模型 fine-tune| 94.54% |

可以看到进行fine-tune能显著提升车牌识别的效果。

#### 4.2.4 CCPD车牌数据集fine-tune+量化训练

此处采用 PaddleOCR 中提供好的[量化教程](../deploy/slim/quantization/README.md)对模型进行量化训练。

量化训练可通过如下命令启动:

```bash
python3.7 deploy/slim/quantization/quant.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o \
    Global.pretrained_model=output/CCPD/rec/best_accuracy.pdparams \
    Global.save_model_dir=output/CCPD/rec_quant/ \
    Global.eval_batch_step="[0, 90]" \
    Optimizer.lr.name=Const \
    Optimizer.lr.learning_rate=0.0005 \
    Optimizer.lr.warmup_epoch=0 \
    Train.dataset.data_dir=/home/aistudio/data/CCPD2020/PPOCR \
    Train.dataset.label_file_list=[/home/aistudio/data/CCPD2020/PPOCR/train/rec.txt] \
    Eval.dataset.data_dir=/home/aistudio/data/CCPD2020/PPOCR \
    Eval.dataset.label_file_list=[/home/aistudio/data/CCPD2020/PPOCR/test/rec.txt]
```

量化后指标对比如下

|方案| acc    | 模型大小  | 预测速度(lite) |
|---|--------|-------|------------|
|PP-OCRv3中英文超轻量识别预训练模型 fine-tune| 94.54% | 10.3M | 4.2ms      |
|PP-OCRv3中英文超轻量识别预训练模型 fine-tune + 量化| 93.40%  | 4.8M  | 1.8ms      |

可以看到量化后能降低模型体积53%并且推理速度提升57%，但是由于识别数据过少，量化带来了1%的精度下降。

速度测试基于[PaddleOCR lite教程](../deploy/lite/readme_ch.md)完成。

#### 4.2.5 模型导出

使用如下命令可以将训练好的模型进行导出。

* 非量化模型
```bash
python tools/export_model.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o \
    Global.pretrained_model=output/CCPD/rec/best_accuracy.pdparams \
    Global.save_inference_dir=output/CCPD/rec/infer
```
* 量化模型
```bash
python deploy/slim/quantization/export_model.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o \
    Global.pretrained_model=output/CCPD/rec_quant/best_accuracy.pdparams \
    Global.save_inference_dir=output/CCPD/rec_quant/infer
```

### 4.3 计算End2End指标

端到端指标可通过 [PaddleOCR内置脚本](../tools/end2end/readme.md) 进行计算，具体步骤如下：

1. 导出模型

通过如下命令进行模型的导出。注意，量化模型导出时，需要配置eval数据集

```bash
# 检测模型

# 预训练模型
python tools/export_model.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o \
    Global.pretrained_model=models/ch_PP-OCRv3_det_distill_train/student.pdparams \
    Global.save_inference_dir=output/ch_PP-OCRv3_det_distill_train/infer

# 非量化模型
python tools/export_model.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o \
    Global.pretrained_model=output/CCPD/det/best_accuracy.pdparams \
    Global.save_inference_dir=output/CCPD/det/infer

# 量化模型
python deploy/slim/quantization/export_model.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o \
    Global.pretrained_model=output/CCPD/det_quant/best_accuracy.pdparams \
    Global.save_inference_dir=output/CCPD/det_quant/infer \
    Eval.dataset.data_dir=/home/aistudio/data/CCPD2020/ccpd_green \
    Eval.dataset.label_file_list=[/home/aistudio/data/CCPD2020/PPOCR/test/det.txt] \
    Eval.loader.num_workers=0

# 识别模型

# 预训练模型
python tools/export_model.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o \
    Global.pretrained_model=models/ch_PP-OCRv3_rec_train/student.pdparams \
    Global.save_inference_dir=output/ch_PP-OCRv3_rec_train/infer

# 非量化模型
python tools/export_model.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o \
    Global.pretrained_model=output/CCPD/rec/best_accuracy.pdparams \
    Global.save_inference_dir=output/CCPD/rec/infer

# 量化模型
python deploy/slim/quantization/export_model.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o \
    Global.pretrained_model=output/CCPD/rec_quant/best_accuracy.pdparams \
    Global.save_inference_dir=output/CCPD/rec_quant/infer \
    Eval.dataset.data_dir=/home/aistudio/data/CCPD2020/PPOCR \
    Eval.dataset.label_file_list=[/home/aistudio/data/CCPD2020/PPOCR/test/rec.txt]
```

2. 用导出的模型对测试集进行预测

此处，分别使用PP-OCRv3预训练模型，fintune模型和量化模型对测试集的所有图像进行预测，命令如下：

```bash
# PP-OCRv3中英文超轻量检测预训练模型，PP-OCRv3中英文超轻量识别预训练模型
python3 tools/infer/predict_system.py --det_model_dir=models/ch_PP-OCRv3_det_distill_train/infer --rec_model_dir=models/ch_PP-OCRv3_rec_train/infer --det_limit_side_len=736 --det_limit_type=min --image_dir=/home/aistudio/data/CCPD2020/ccpd_green/test/ --draw_img_save_dir=infer/pretrain --use_dilation=true

# PP-OCRv3中英文超轻量检测预训练模型+fine-tune，PP-OCRv3中英文超轻量识别预训练模型+fine-tune
python3 tools/infer/predict_system.py --det_model_dir=output/CCPD/det/infer --rec_model_dir=output/CCPD/rec/infer --det_limit_side_len=736 --det_limit_type=min --image_dir=/home/aistudio/data/CCPD2020/ccpd_green/test/ --draw_img_save_dir=infer/fine-tune --use_dilation=true

# PP-OCRv3中英文超轻量检测预训练模型 fine-tune +量化，PP-OCRv3中英文超轻量识别预训练模型 fine-tune +量化 结果转换和评估
python3 tools/infer/predict_system.py --det_model_dir=output/CCPD/det_quant/infer --rec_model_dir=output/CCPD/rec_quant/infer --det_limit_side_len=736 --det_limit_type=min --image_dir=/home/aistudio/data/CCPD2020/ccpd_green/test/ --draw_img_save_dir=infer/quant --use_dilation=true
```

3. 转换label并计算指标

将gt和上一步保存的预测结果转换为端对端评测需要的数据格式，并根据转换后的数据进行端到端指标计算

```bash
python3 tools/end2end/convert_ppocr_label.py --mode=gt --label_path=/home/aistudio/data/CCPD2020/PPOCR/test/det.txt --save_folder=end2end/gt

# PP-OCRv3中英文超轻量检测预训练模型，PP-OCRv3中英文超轻量识别预训练模型 结果转换和评估
python3 tools/end2end/convert_ppocr_label.py --mode=pred --label_path=infer/pretrain/system_results.txt --save_folder=end2end/pretrain
python3 tools/end2end/eval_end2end.py end2end/gt end2end/pretrain

# PP-OCRv3中英文超轻量检测预训练模型，PP-OCRv3中英文超轻量识别预训练模型+后处理去掉多识别的`·` 结果转换和评估
# 需手动修改后处理函数
python3 tools/end2end/convert_ppocr_label.py --mode=pred --label_path=infer/post/system_results.txt --save_folder=end2end/post
python3 tools/end2end/eval_end2end.py end2end/gt end2end/post

# PP-OCRv3中英文超轻量检测预训练模型 fine-tune，PP-OCRv3中英文超轻量识别预训练模型 fine-tune 结果转换和评估
python3 tools/end2end/convert_ppocr_label.py --mode=pred --label_path=infer/fine-tune/system_results.txt --save_folder=end2end/fine-tune
python3 tools/end2end/eval_end2end.py end2end/gt end2end/fine-tune

# PP-OCRv3中英文超轻量检测预训练模型 fine-tune +量化，PP-OCRv3中英文超轻量识别预训练模型 fine-tune +量化 结果转换和评估
python3 tools/end2end/convert_ppocr_label.py --mode=pred --label_path=infer/quant/system_results.txt --save_folder=end2end/quant
python3 tools/end2end/eval_end2end.py end2end/gt end2end/quant
```

日志如下:
```bash
The convert label saved in end2end/gt
The convert label saved in end2end/pretrain
start testing...
hit, dt_count, gt_count 2 5988 5006
character_acc: 70.42%
avg_edit_dist_field: 2.37
avg_edit_dist_img: 2.37
precision: 0.03%
recall: 0.04%
fmeasure: 0.04%
The convert label saved in end2end/post
start testing...
hit, dt_count, gt_count 4224 5988 5006
character_acc: 81.59%
avg_edit_dist_field: 1.47
avg_edit_dist_img: 1.47
precision: 70.54%
recall: 84.38%
fmeasure: 76.84%
The convert label saved in end2end/fine-tune
start testing...
hit, dt_count, gt_count 4286 4898 5006
character_acc: 94.16%
avg_edit_dist_field: 0.47
avg_edit_dist_img: 0.47
precision: 87.51%
recall: 85.62%
fmeasure: 86.55%
The convert label saved in end2end/quant
start testing...
hit, dt_count, gt_count 4349 4951 5006
character_acc: 94.13%
avg_edit_dist_field: 0.47
avg_edit_dist_img: 0.47
precision: 87.84%
recall: 86.88%
fmeasure: 87.36%
```

各个方案端到端指标如下：

|模型| 指标     |
|---|--------|
|PP-OCRv3中英文超轻量检测预训练模型 <br> PP-OCRv3中英文超轻量识别预训练模型| 0.04%  |
|PP-OCRv3中英文超轻量检测预训练模型 <br> PP-OCRv3中英文超轻量识别预训练模型 + 后处理去掉多识别的`·`| 78.27% |
|PP-OCRv3中英文超轻量检测预训练模型+fine-tune <br> PP-OCRv3中英文超轻量识别预训练模型+fine-tune| 87.14% |
|PP-OCRv3中英文超轻量检测预训练模型+fine-tune+量化 <br> PP-OCRv3中英文超轻量识别预训练模型+fine-tune+量化| 88.00%    |

从结果中可以看到对预训练模型不做修改，只根据场景下的具体情况进行后处理的修改就能大幅提升端到端指标到78.27%，在CCPD数据集上进行 fine-tune 后指标进一步提升到87.14%, 在经过量化训练之后，由于检测模型的recall变高，指标进一步提升到88%。但是这个结果仍旧不符合检测模型+识别模型的真实性能(99%*94%=93%)，因此我们需要对 base case 进行具体分析。

在之前的端到端预测结果中，可以看到很多不符合车牌标注的文字被识别出来, 因此可以进行简单的过滤来提升precision

为了快速评估，我们在 ` tools/end2end/convert_ppocr_label.py` 脚本的 58 行加入如下代码，对非8个字符的结果进行过滤
```python
if len(txt) != 8: # 车牌字符串长度为8
    continue
```

此外，通过可视化box可以发现有很多框都是竖直翻转之后的框，并且没有完全框住车牌边界，因此需要进行框的竖直翻转以及轻微扩大，示意图如下：

![](https://ai-studio-static-online.cdn.bcebos.com/59ab0411c8eb4dfd917fb2b6e5b69a17ee7ca48351444aec9ac6104b79ff1028)

修改前后个方案指标对比如下：


各个方案端到端指标如下：

|模型|base|A:识别结果过滤|B:use_dilation|C:flip_box|best|
|---|---|---|---|---|---|
|PP-OCRv3中英文超轻量检测预训练模型 <br> PP-OCRv3中英文超轻量识别预训练模型|0.04%|0.08%|0.02%|0.05%|0.00%(A)|
|PP-OCRv3中英文超轻量检测预训练模型 <br> PP-OCRv3中英文超轻量识别预训练模型 + 后处理去掉多识别的`·`|78.27%|90.84%|78.61%|79.43%|91.66%(A+B+C)|
|PP-OCRv3中英文超轻量检测预训练模型+fine-tune <br> PP-OCRv3中英文超轻量识别预训练模型+fine-tune|87.14%|90.40%|87.66%|89.98%|92.50%(A+B+C)|
|PP-OCRv3中英文超轻量检测预训练模型+fine-tune+量化 <br> PP-OCRv3中英文超轻量识别预训练模型+fine-tune+量化|88.00%|90.54%|88.50%|89.46%|92.02%(A+B+C)|


从结果中可以看到对预训练模型不做修改，只根据场景下的具体情况进行后处理的修改就能大幅提升端到端指标到91.66%，在CCPD数据集上进行 fine-tune 后指标进一步提升到92.5%, 在经过量化训练之后，指标变为92.02%。

### 4.4 部署

- 基于 Paddle Inference 的python推理

检测模型和识别模型分别 fine-tune 并导出为inference模型之后，可以使用如下命令基于 Paddle Inference 进行端到端推理并对结果进行可视化。

```bash
python tools/infer/predict_system.py \
    --det_model_dir=output/CCPD/det/infer/ \
    --rec_model_dir=output/CCPD/rec/infer/ \
    --image_dir="/home/aistudio/data/CCPD2020/ccpd_green/test/04131106321839081-92_258-159&509_530&611-527&611_172&599_159&509_530&525-0_0_3_32_30_31_30_30-109-106.jpg" \
    --rec_image_shape=3,48,320
```
推理结果如下

![](https://ai-studio-static-online.cdn.bcebos.com/76b6a0939c2c4cf49039b6563c4b28e241e11285d7464e799e81c58c0f7707a7)

- 端侧部署

端侧部署我们采用基于 PaddleLite 的 cpp 推理。Paddle Lite是飞桨轻量化推理引擎，为手机、IOT端提供高效推理能力，并广泛整合跨平台硬件，为端侧部署及应用落地问题提供轻量化的部署方案。具体可参考 [PaddleOCR lite教程](../deploy/lite/readme_ch.md)


### 4.5 实验总结

我们分别使用PP-OCRv3中英文超轻量预训练模型在车牌数据集上进行了直接评估和 fine-tune 和 fine-tune +量化3种方案的实验，并基于[PaddleOCR lite教程](../deploy/lite/readme_ch.md)进行了速度测试，指标对比如下：

- 检测

|方案|hmeans| 模型大小 | 预测速度(lite) |
|---|---|------|------------|
|PP-OCRv3中英文超轻量检测预训练模型直接预测|76.12%|2.5M| 233ms      |
|PP-OCRv3中英文超轻量检测预训练模型 fine-tune|99.00%| 2.5M | 233ms      |
|PP-OCRv3中英文超轻量检测预训练模型 fine-tune + 量化|98.91%| 1.0M   | 189ms      |fine-tune

- 识别

|方案| acc    | 模型大小  | 预测速度(lite) |
|---|--------|-------|------------|
|PP-OCRv3中英文超轻量识别预训练模型直接预测| 0.00%     |10.3M| 4.2ms      |
|PP-OCRv3中英文超轻量识别预训练模型直接预测+后处理去掉多识别的`·`| 90.97% |10.3M| 4.2ms      |
|PP-OCRv3中英文超轻量识别预训练模型 fine-tune| 94.54% | 10.3M | 4.2ms      |
|PP-OCRv3中英文超轻量识别预训练模型 fine-tune + 量化| 93.40%  | 4.8M  | 1.8ms      |


- 端到端指标如下：

|方案|fmeasure|模型大小|预测速度(lite) |
|---|---|---|---|
|PP-OCRv3中英文超轻量检测预训练模型 <br> PP-OCRv3中英文超轻量识别预训练模型|0.08%|12.8M|298ms|
|PP-OCRv3中英文超轻量检测预训练模型 <br> PP-OCRv3中英文超轻量识别预训练模型 + 后处理去掉多识别的`·`|91.66%|12.8M|298ms|
|PP-OCRv3中英文超轻量检测预训练模型+fine-tune <br> PP-OCRv3中英文超轻量识别预训练模型+fine-tune|92.50%|12.8M|298ms|
|PP-OCRv3中英文超轻量检测预训练模型+fine-tune+量化 <br> PP-OCRv3中英文超轻量识别预训练模型+fine-tune+量化|92.02%|5.80M|224ms|


**结论**

PP-OCRv3的检测模型在未经过fine-tune的情况下，在车牌数据集上也有一定的精度，经过 fine-tune 后能够极大的提升检测效果，精度达到99%。在使用量化训练后检测模型的精度几乎无损，并且模型大小压缩60%。

PP-OCRv3的识别模型在未经过fine-tune的情况下，在车牌数据集上精度为0，但是经过分析可以知道，模型大部分字符都预测正确，但是会多预测一个特殊字符，去掉这个特殊字符后，精度达到90%。PP-OCRv3识别模型在经过 fine-tune 后识别精度进一步提升，达到94.4%。在使用量化训练后识别模型大小压缩53%，但是由于数据量多少，带来了1%的精度损失。

从端到端结果中可以看到对预训练模型不做修改，只根据场景下的具体情况进行后处理的修改就能大幅提升端到端指标到91.66%，在CCPD数据集上进行 fine-tune 后指标进一步提升到92.5%, 在经过量化训练之后，指标轻微下降到92.02%但模型大小降低54%。
