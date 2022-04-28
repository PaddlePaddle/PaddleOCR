# PaddleOCR 快速开始

**说明：** 本文主要介绍PaddleOCR wheel包对PP-OCR系列模型的快速使用，如要体验文档分析相关功能，请参考[PP-Structure快速使用教程](../../ppstructure/docs/quickstart.md)。

- [1. 安装](#1)
  - [1.1 安装PaddlePaddle](#11)
  - [1.2 安装PaddleOCR whl包](#12)
- [2. 便捷使用](#2)
  - [2.1 命令行使用](#21)
      - [2.1.1 中英文模型](#211)
      - [2.1.2 多语言模型](#212)
  - [2.2 Python脚本使用](#22)
      - [2.2.1 中英文与多语言使用](#221)
- [3.小结](#3)


<a name="1"></a>
## 1. 安装

<a name="11"></a>
### 1.1 安装PaddlePaddle

> 如果您没有基础的Python运行环境，请参考[运行环境准备](./environment.md)。

- 您的机器安装的是CUDA9或CUDA10，请运行以下命令安装

  ```bash
  python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
  ```

- 您的机器是CPU，请运行以下命令安装

  ```bash
  python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
  ```

更多的版本需求，请参照[飞桨官网安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。

<a name="12"></a>
### 1.2 安装PaddleOCR whl包

```bash
pip install "paddleocr>=2.0.1" # 推荐使用2.0.1+版本
```

- 对于Windows环境用户：直接通过pip安装的shapely库可能出现`[winRrror 126] 找不到指定模块的问题`。建议从[这里](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely)下载shapely安装包完成安装。


<a name="2"></a>
## 2. 便捷使用
<a name="21"></a>
### 2.1 命令行使用

PaddleOCR提供了一系列测试图片，点击[这里](https://paddleocr.bj.bcebos.com/dygraph_v2.1/ppocr_img.zip)下载并解压，然后在终端中切换到相应目录

```
cd /path/to/ppocr_img
```

如果不使用提供的测试图片，可以将下方`--image_dir`参数替换为相应的测试图片路径。

<a name="211"></a>
#### 2.1.1 中英文模型

* 检测+方向分类器+识别全流程：`--use_angle_cls true`设置使用方向分类器识别180度旋转文字，`--use_gpu false`设置不使用GPU

  ```bash
  paddleocr --image_dir ./imgs/11.jpg --use_angle_cls true --use_gpu false
  ```

  结果是一个list，每个item包含了文本框，文字和识别置信度

  ```bash
  [[[24.0, 36.0], [304.0, 34.0], [304.0, 72.0], [24.0, 74.0]], ['纯臻营养护发素', 0.964739]]
  [[[24.0, 80.0], [172.0, 80.0], [172.0, 104.0], [24.0, 104.0]], ['产品信息/参数', 0.98069626]]
  [[[24.0, 109.0], [333.0, 109.0], [333.0, 136.0], [24.0, 136.0]], ['（45元/每公斤，100公斤起订）', 0.9676722]]
  ......
  ```

- 单独使用检测：设置`--rec`为`false`

  ```bash
  paddleocr --image_dir ./imgs/11.jpg --rec false
  ```

  结果是一个list，每个item只包含文本框

  ```bash
  [[26.0, 457.0], [137.0, 457.0], [137.0, 477.0], [26.0, 477.0]]
  [[25.0, 425.0], [372.0, 425.0], [372.0, 448.0], [25.0, 448.0]]
  [[128.0, 397.0], [273.0, 397.0], [273.0, 414.0], [128.0, 414.0]]
  ......
  ```

- 单独使用识别：设置`--det`为`false`

  ```bash
  paddleocr --image_dir ./imgs_words/ch/word_1.jpg --det false
  ```

  结果是一个list，每个item只包含识别结果和识别置信度

  ```bash
  ['韩国小馆', 0.9907421]
  ```


如需使用2.0模型，请指定参数`--version PP-OCR`，paddleocr默认使用2.1模型(`--versioin PP-OCRv2`)。更多whl包使用可参考[whl包文档](./whl.md)

<a name="212"></a>

#### 2.1.2 多语言模型

Paddleocr目前支持80个语种，可以通过修改`--lang`参数进行切换，对于英文模型，指定`--lang=en`。

``` bash
paddleocr --image_dir ./imgs_en/254.jpg --lang=en
```

<div align="center">
    <img src="../imgs_en/254.jpg" width="300" height="600">
    <img src="../imgs_results/multi_lang/img_02.jpg" width="600" height="600">
</div>

结果是一个list，每个item包含了文本框，文字和识别置信度

```text
[('PHO CAPITAL', 0.95723116), [[66.0, 50.0], [327.0, 44.0], [327.0, 76.0], [67.0, 82.0]]]
[('107 State Street', 0.96311164), [[72.0, 90.0], [451.0, 84.0], [452.0, 116.0], [73.0, 121.0]]]
[('Montpelier Vermont', 0.97389287), [[69.0, 132.0], [501.0, 126.0], [501.0, 158.0], [70.0, 164.0]]]
[('8022256183', 0.99810505), [[71.0, 175.0], [363.0, 170.0], [364.0, 202.0], [72.0, 207.0]]]
[('REG 07-24-201706:59 PM', 0.93537045), [[73.0, 299.0], [653.0, 281.0], [654.0, 318.0], [74.0, 336.0]]]
[('045555', 0.99346405), [[509.0, 331.0], [651.0, 325.0], [652.0, 356.0], [511.0, 362.0]]]
[('CT1', 0.9988654), [[535.0, 367.0], [654.0, 367.0], [654.0, 406.0], [535.0, 406.0]]]
......
```

常用的多语言简写包括

| 语种     | 缩写        |      | 语种     | 缩写   |      | 语种     | 缩写   |
| -------- | ----------- | ---- | -------- | ------ | ---- | -------- | ------ |
| 中文     | ch          |      | 法文     | fr     |      | 日文     | japan  |
| 英文     | en          |      | 德文     | german |      | 韩文     | korean |
| 繁体中文 | chinese_cht |      | 意大利文 | it     |      | 俄罗斯文 | ru     |

全部语种及其对应的缩写列表可查看[多语言模型教程](./multi_languages.md)


<a name="22"></a>
### 2.2 Python脚本使用
<a name="221"></a>
#### 2.2.1 中英文与多语言使用

通过Python脚本使用PaddleOCR whl包，whl包会自动下载ppocr轻量级模型作为默认模型。

* 检测+方向分类器+识别全流程

```python
from paddleocr import PaddleOCR, draw_ocr

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
img_path = './imgs/11.jpg'
result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)

# 显示结果
from PIL import Image

image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

结果是一个list，每个item包含了文本框，文字和识别置信度

```bash
[[[24.0, 36.0], [304.0, 34.0], [304.0, 72.0], [24.0, 74.0]], ['纯臻营养护发素', 0.964739]]
[[[24.0, 80.0], [172.0, 80.0], [172.0, 104.0], [24.0, 104.0]], ['产品信息/参数', 0.98069626]]
[[[24.0, 109.0], [333.0, 109.0], [333.0, 136.0], [24.0, 136.0]], ['（45元/每公斤，100公斤起订）', 0.9676722]]
......
```

结果可视化

<div align="center">
    <img src="../imgs_results/whl/11_det_rec.jpg" width="800">
</div>


<a name="3"></a>

## 3. 小结

通过本节内容，相信您已经熟练掌握PaddleOCR whl包的使用方法并获得了初步效果。

PaddleOCR是一套丰富领先实用的OCR工具库，打通数据、模型训练、压缩和推理部署全流程，您可以参考[文档教程](../../README_ch.md#文档教程)，正式开启PaddleOCR的应用之旅。
