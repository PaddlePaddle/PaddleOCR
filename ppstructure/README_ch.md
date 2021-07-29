# PaddleStructure

PaddleStructure是一个用于复杂版面分析的OCR工具包，其能够对图片形式的文档数据划分**文字、表格、标题、图片以及列表**5类区域，并将表格区域提取为excel

## 1. 快速开始

### 1.1 安装

**安装 layoutparser**
```sh
pip3 install https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl
```
**安装 paddlestructure**

pip安装
```bash
pip install paddlestructure
```

本地构建并安装
```bash
python3 setup.py bdist_wheel
pip3 install dist/paddlestructure-x.x.x-py3-none-any.whl # x.x.x是 paddlestructure 的版本号
```

### 1.2 PaddleStructure whl包使用

#### 1.2.1 命令行使用

```bash
paddlestructure --image_dir=../doc/table/1.png
```

#### 1.2.2 Python脚本使用

```python
import os
import cv2
from paddlestructure import PaddleStructure,draw_result,save_res

table_engine = PaddleStructure(show_log=True)

save_folder = './output/table'
img_path = '../doc/table/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
save_res(result, save_folder,os.path.basename(img_path).split('.')[0])

for line in result:
    print(line)

from PIL import Image

font_path = '../doc/fonts/simfang.ttf' # PaddleOCR下提供字体包
image = Image.open(img_path).convert('RGB')
im_show = draw_result(image, result,font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```


#### 1.2.3 参数说明

| 字段            | 说明                                     | 默认值                                      |
| --------------- | ---------------------------------------- | ------------------------------------------- |
| output          | excel和识别结果保存的地址                | ./output/table                              |
| table_max_len   | 表格结构模型预测时，图像的长边resize尺度 | 488                                         |
| table_model_dir | 表格结构模型 inference 模型地址          | None                                        |
| table_char_type | 表格结构模型所用字典地址                 | ../ppocr/utils/dict/table_structure_dict.tx |

大部分参数和paddleocr whl包保持一致，见 [whl包文档](../doc/doc_ch/whl.md)

运行完成后，每张图片会在`output`字段指定的目录下有一个同名目录，图片里的每个表格会存储为一个excel，excel文件名为表格在图片里的坐标。


## 2. PaddleStructure Pipeline

流程如下
![pipeline](../doc/table/pipeline.jpg)

在PaddleStructure中，图片会先经由layoutparser进行版面分析，在版面分析中，会对图片里的区域进行分类，包括**文字、标题、图片、列表和表格**5类。对于前4类区域，直接使用PP-OCR完成对应区域文字检测与识别。对于表格类区域，经过Table OCR处理后，表格图片转换为相同表格样式的Excel文件。

### 2.1 LayoutParser

版面分析对文档数据进行区域分类，其中包括版面分析工具的Python脚本使用、提取指定类别检测框、性能指标以及自定义训练版面分析模型，详细内容可以参考[文档](layout/README.md)。

### 2.2 Table OCR

Table OCR将表格图片转换为excel文档，其中包含对于表格文本的检测和识别以及对于表格结构和单元格坐标的预测，详细说明参考[文档](table/README_ch.md)

### 3. 预测引擎推理

使用如下命令即可完成预测引擎的推理

```python
python3 table/predict_system.py --det_model_dir=path/to/det_model_dir --rec_model_dir=path/to/rec_model_dir --table_model_dir=path/to/table_model_dir --image_dir=../doc/table/1.png --rec_char_dict_path=../ppocr/utils/dict/table_dict.txt --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt --rec_char_type=EN --det_limit_side_len=736 --det_limit_type=min --output ../output/table
```
运行完成后，每张图片会output字段指定的目录下有一个同名目录，图片里的每个表格会存储为一个excel，excel文件名为表格在图片里的坐标。

# 3. Model List


|模型名称|模型简介|配置文件|推理模型大小|下载地址|
| --- | --- | --- | --- | --- |
|en_ppocr_mobile_v2.0_table_det|英文表格场景的文字检测|[ch_det_mv3_db_v2.0.yml](../configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml)| 4.7M |[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar) |
|en_ppocr_mobile_v2.0_table_rec|英文表格场景的文字识别|[rec_chinese_lite_train_v2.0.yml](..//configs/rec/rec_mv3_none_bilstm_ctc.yml)|6.9M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar) |
|en_ppocr_mobile_v2.0_table_structure|英文表格场景的表格结构预测|[table_mv3.yml](../configs/table/table_mv3.yml)|18.6M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar) |