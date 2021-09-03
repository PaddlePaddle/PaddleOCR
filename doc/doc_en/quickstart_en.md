
# PaddleOCR Quick Start

[TOC]

## 1. 轻量安装

### 1.0 Environment Preparation

环境配置

python环境、pip安装

```bash
pip3 install --upgrade pip
```

### 1.1 Install PaddlePaddle2.0

```bash
# If you have cuda9 or cuda10 installed on your machine, please run the following command to install
python3 -m pip install paddlepaddle-gpu==2.0.0 -i https://mirror.baidu.com/pypi/simple

# If you only have cpu on your machine, please run the following command to install
python3 -m pip install paddlepaddle==2.0.0 -i https://mirror.baidu.com/pypi/simple
```

For more software version requirements, please refer to the instructions in [Installation Document](https://www.paddlepaddle.org.cn/install/quick) for operation.

### 1.2 Install PaddleOCR Whl Package

```bash
pip install "paddleocr>=2.0.1" # Recommend to use version 2.0.1+
```

是否会出现sharply问题？



如果需要使用版面分析功能，还需**安装 Layout-Parser**

```bash
pip3 install -U https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl
```

## 2. 便捷使用

### 2.1 Use by command line

#### 2.1.1 English and Chinese Model

* detection classification and recognition

```bash
paddleocr --image_dir PaddleOCR/doc/imgs_en/img_12.jpg --use_angle_cls true --lang en
```

Output will be a list, each item contains bounding box, text and recognition confidence

```bash
[[[442.0, 173.0], [1169.0, 173.0], [1169.0, 225.0], [442.0, 225.0]], ['ACKNOWLEDGEMENTS', 0.99283075]]
[[[393.0, 340.0], [1207.0, 342.0], [1207.0, 389.0], [393.0, 387.0]], ['We would like to thank all the designers and', 0.9357758]]
[[[399.0, 398.0], [1204.0, 398.0], [1204.0, 433.0], [399.0, 433.0]], ['contributors whohave been involved in the', 0.9592447]]
......
```

* 更多whl包使用包括， whl包参数说明：

#### 2.1.2 Multi-language Model

Paddleocr currently supports 80 languages, which can be switched by modifying the --lang parameter.The specific supported [language](language_abbreviations) can be viewed in the table.

``` bash
paddleocr --image_dir ./doc/imgs_en/254.jpg --lang=en
```

<div align="center">
    <img src="../imgs_en/254.jpg" width="300" height="600">
    <img src="../imgs_results/multi_lang/img_02.jpg" width="600" height="600">
</div>


The result is a list, each item contains a text box, text and recognition confidence

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

#### 2.1.3 版面分析

```bash
paddleocr --image_dir=../doc/table/1.png --type=structure
```

1. **返回结果说明**

PP-Structure的返回结果为一个dict组成的list，示例如下

```shell
[
  {   'type': 'Text',
      'bbox': [34, 432, 345, 462],
      'res': ([[36.0, 437.0, 341.0, 437.0, 341.0, 446.0, 36.0, 447.0], [41.0, 454.0, 125.0, 453.0, 125.0, 459.0, 41.0, 460.0]],
                [('Tigure-6. The performance of CNN and IPT models using difforen', 0.90060663), ('Tent  ', 0.465441)])
  }
]
```

dict 里各个字段说明如下

| 字段 | 说明                                                         |
| ---- | ------------------------------------------------------------ |
| type | 图片区域的类型                                               |
| bbox | 图片区域的在原图的坐标，分别[左上角x，左上角y，右下角x，右下角y] |
| res  | 图片区域的OCR或表格识别结果。<br> 表格: 表格的HTML字符串; <br> OCR: 一个包含各个单行文字的检测坐标和识别结果的元组 |

2. **参数说明**

| 字段            | 说明                                     | 默认值                                       |
| --------------- | ---------------------------------------- | -------------------------------------------- |
| output          | excel和识别结果保存的地址                | ./output/table                               |
| table_max_len   | 表格结构模型预测时，图像的长边resize尺度 | 488                                          |
| table_model_dir | 表格结构模型 inference 模型地址          | None                                         |
| table_char_type | 表格结构模型所用字典地址                 | ../ppocr/utils/dict/table_structure_dict.txt |

大部分参数和paddleocr whl包保持一致，见 [whl包文档](../doc/doc_ch/whl.md)

运行完成后，每张图片会在`output`字段指定的目录下有一个同名目录，图片里的每个表格会存储为一个excel，图片区域会被裁剪之后保存下来，excel文件和图片名名为表格在图片里的坐标。

### 2.2 Python脚本使用

#### 2.2.1 中英文与多语言使用

paddleocr whl包会自动下载ppocr轻量级模型作为默认模型，可以根据第3节**自定义模型**进行自定义更换。

* 检测+方向分类器+识别全流程

```python
from paddleocr import PaddleOCR, draw_ocr

# Paddleocr目前支持中英文、英文、法语、德语、韩语、日语，可以通过修改lang参数进行切换
# 参数依次为`ch`, `en`, `french`, `german`, `korean`, `japan`。
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
img_path = 'Path/to/Your/Img/11.jpg'
result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)

# 显示结果
from PIL import Image

image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/fonts/simfang.ttf')
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


#### 2.2.2 版面分析使用
