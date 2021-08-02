# PPStructure

PaddleStructure是一个用于复杂版面分析的OCR工具包，其能够对图片形式的文档数据划分**文字、表格、标题、图片以及列表**5类区域，并将表格区域提取为excel

## 1. 快速开始

### 1.1 安装

**安装PaddlePaddle2.0**

```bash
pip3 install --upgrade pip

# 如果您的机器安装的是CUDA9或CUDA10，请运行以下命令安装
python3 -m pip install paddlepaddle-gpu==2.0.0 -i https://mirror.baidu.com/pypi/simple

# 如果您的机器是CPU，请运行以下命令安装
 
python3 -m pip install paddlepaddle==2.0.0 -i https://mirror.baidu.com/pypi/simple

# 更多的版本需求，请参照[安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。
```

**克隆PaddleOCR repo代码**

```bash
【推荐】git clone https://github.com/PaddlePaddle/PaddleOCR

如果因为网络问题无法pull成功，也可选择使用码云上的托管：

git clone https://gitee.com/paddlepaddle/PaddleOCR

注：码云托管代码可能无法实时同步本github项目更新，存在3~5天延时，请优先使用推荐方式。

```

**安装 paddleocr**

pip安装
```bash
cd PaddleOCR
pip install "paddleocr>=2.0.1" # 推荐使用2.0.1+版本
```

本地构建并安装
```bash
python3 setup.py bdist_wheel
pip3 install dist/paddleocr-x.x.x-py3-none-any.whl # x.x.x是paddleocr的版本号
```

**安装 layoutparser**
```sh
pip3 install -U premailer paddleocr https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl
```

### 1.2 PPStructure whl包使用

#### 1.2.1 命令行使用

```bash
paddleocr --image_dir=../doc/table/1.png --type=structure
```

#### 1.2.2 Python脚本使用

```python
import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res

table_engine = PPStructure(show_log=True)

save_folder = './output/table'
img_path = '../doc/table/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

from PIL import Image

font_path = '../doc/fonts/simfang.ttf' # PaddleOCR下提供字体包
image = Image.open(img_path).convert('RGB')
im_show = draw_structure_result(image, result,font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

#### 1.2.3 返回结果说明
PaddleStructure 的返回结果为一个dict组成的list，示例如下

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

| 字段            | 说明           | 
| --------------- | -------------|
|type|图片区域的类型|
|bbox|图片区域的在原图的坐标，分别[左上角x，左上角y，右下角x，右下角y]|
|res|图片区域的OCR或表格识别结果。<br> 表格: 表格的HTML字符串; <br> OCR: 一个包含各个单行文字的检测坐标和识别结果的元组|


#### 1.2.4 参数说明

| 字段            | 说明                                     | 默认值                                      |
| --------------- | ---------------------------------------- | ------------------------------------------- |
| output          | excel和识别结果保存的地址                | ./output/table                              |
| table_max_len   | 表格结构模型预测时，图像的长边resize尺度 | 488                                         |
| table_model_dir | 表格结构模型 inference 模型地址          | None                                        |
| table_char_type | 表格结构模型所用字典地址                 | ../ppocr/utils/dict/table_structure_dict.tx |

大部分参数和paddleocr whl包保持一致，见 [whl包文档](../doc/doc_ch/whl.md)

运行完成后，每张图片会在`output`字段指定的目录下有一个同名目录，图片里的每个表格会存储为一个excel，图片区域会被裁剪之后保存下来，excel文件和图片名名为表格在图片里的坐标。


## 2. PPStructure Pipeline

流程如下
![pipeline](../doc/table/pipeline.jpg)

在PPStructure中，图片会先经由layoutparser进行版面分析，在版面分析中，会对图片里的区域进行分类，包括**文字、标题、图片、列表和表格**5类。对于前4类区域，直接使用PP-OCR完成对应区域文字检测与识别。对于表格类区域，经过Table OCR处理后，表格图片转换为相同表格样式的Excel文件。

### 2.1 版面分析

版面分析对文档数据进行区域分类，其中包括版面分析工具的Python脚本使用、提取指定类别检测框、性能指标以及自定义训练版面分析模型，详细内容可以参考[文档](layout/README.md)。

### 2.2 表格结构化

Table OCR将表格图片转换为excel文档，其中包含对于表格文本的检测和识别以及对于表格结构和单元格坐标的预测，详细说明参考[文档](table/README_ch.md)

## 3. 预测引擎推理

使用如下命令即可完成预测引擎的推理

```python
cd ppstructure

# 下载模型
mkdir inference && cd inference
# 下载超轻量级中文OCR模型的检测模型并解压
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_ppocr_mobile_v2.0_det_infer.tar
# 下载超轻量级中文OCR模型的识别模型并解压
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar
# 下载超轻量级英文表格英寸模型并解压
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar && tar xf en_ppocr_mobile_v2.0_table_structure_infer.tar
cd ..

python3 table/predict_system.py --det_model_dir=inference/ch_ppocr_mobile_v2.0_det_infer --rec_model_dir=inference/ch_ppocr_mobile_v2.0_rec_infer --table_model_dir=inference/en_ppocr_mobile_v2.0_table_structure_infer --image_dir=../doc/table/1.png --rec_char_dict_path=../ppocr/utils/ppocr_keys_v1.txt --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt --rec_char_type=ch --det_limit_side_len=736 --det_limit_type=min --output=../output/table --vis_font_path=../doc/fonts/simfang.ttf
```
运行完成后，每张图片会在`output`字段指定的目录下有一个同名目录，图片里的每个表格会存储为一个excel，图片区域会被裁剪之后保存下来，excel文件和图片名名为表格在图片里的坐标。

**Model List**

|模型名称|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- | --- |
|en_ppocr_mobile_v2.0_table_structure|英文表格场景的表格结构预测|[table_mv3.yml](../configs/table/table_mv3.yml)|18.6M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar) |