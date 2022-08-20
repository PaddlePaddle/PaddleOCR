# PP-Structure 快速开始

- [1. 安装依赖包](#1-安装依赖包)
- [2. 便捷使用](#2-便捷使用)
  - [2.1 命令行使用](#21-命令行使用)
    - [2.1.1 图像方向分类+版面分析+表格识别](#211-图像方向分类版面分析表格识别)
    - [2.1.2 版面分析+表格识别](#212-版面分析表格识别)
    - [2.1.3 版面分析](#213-版面分析)
    - [2.1.4 表格识别](#214-表格识别)
    - [2.1.5 DocVQA](#215-dockie)
  - [2.2 代码使用](#22-代码使用)
    - [2.2.1 图像方向分类版面分析表格识别](#221-图像方向分类版面分析表格识别)
    - [2.2.2 版面分析+表格识别](#222-版面分析表格识别)
    - [2.2.3 版面分析](#223-版面分析)
    - [2.2.4 表格识别](#224-表格识别)
    - [2.2.5 DocVQA](#225-dockie)
  - [2.3 返回结果说明](#23-返回结果说明)
    - [2.3.1 版面分析+表格识别](#231-版面分析表格识别)
    - [2.3.2 DocVQA](#232-dockie)
  - [2.4 参数说明](#24-参数说明)


<a name="1"></a>
## 1. 安装依赖包

```bash
# 安装 paddleocr，推荐使用2.5+版本
pip3 install "paddleocr>=2.5"
# 安装 DocVQA依赖包paddlenlp（如不需要DocVQA功能，可跳过）
pip install paddlenlp

```

<a name="2"></a>
## 2. 便捷使用

<a name="21"></a>
### 2.1 命令行使用  

<a name="211"></a>
#### 2.1.1 图像方向分类+版面分析+表格识别
```bash
paddleocr --image_dir=PaddleOCR/ppstructure/docs/table/1.png --type=structure --image_orientation=true
```

<a name="212"></a>
#### 2.1.2 版面分析+表格识别
```bash
paddleocr --image_dir=PaddleOCR/ppstructure/docs/table/1.png --type=structure
```

<a name="213"></a>
#### 2.1.3 版面分析
```bash
paddleocr --image_dir=PaddleOCR/ppstructure/docs/table/1.png --type=structure --table=false --ocr=false
```

<a name="214"></a>
#### 2.1.4 表格识别
```bash
paddleocr --image_dir=PaddleOCR/ppstructure/docs/table/table.jpg --type=structure --layout=false
```

<a name="215"></a>
#### 2.1.5 DocVQA

请参考：[文档视觉问答](../kie/README.md)。

<a name="22"></a>
### 2.2 代码使用

<a name="221"></a>
#### 2.2.1 图像方向分类版面分析表格识别

```python
import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res

table_engine = PPStructure(show_log=True, image_orientation=True)

save_folder = './output'
img_path = 'PaddleOCR/ppstructure/docs/table/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

from PIL import Image

font_path = 'PaddleOCR/doc/fonts/simfang.ttf' # PaddleOCR下提供字体包
image = Image.open(img_path).convert('RGB')
im_show = draw_structure_result(image, result,font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

<a name="222"></a>
#### 2.2.2 版面分析+表格识别

```python
import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res

table_engine = PPStructure(show_log=True)

save_folder = './output'
img_path = 'PaddleOCR/ppstructure/docs/table/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

from PIL import Image

font_path = 'PaddleOCR/doc/fonts/simfang.ttf' # PaddleOCR下提供字体包
image = Image.open(img_path).convert('RGB')
im_show = draw_structure_result(image, result,font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

<a name="223"></a>
#### 2.2.3 版面分析

```python
import os
import cv2
from paddleocr import PPStructure,save_structure_res

table_engine = PPStructure(table=False, ocr=False, show_log=True)

save_folder = './output'
img_path = 'PaddleOCR/ppstructure/docs/table/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)
```

<a name="224"></a>
#### 2.2.4 表格识别

```python
import os
import cv2
from paddleocr import PPStructure,save_structure_res

table_engine = PPStructure(layout=False, show_log=True)

save_folder = './output'
img_path = 'PaddleOCR/ppstructure/docs/table/table.jpg'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)
```

<a name="225"></a>
#### 2.2.5 DocVQA

请参考：[文档视觉问答](../kie/README.md)。

<a name="23"></a>
### 2.3 返回结果说明
PP-Structure的返回结果为一个dict组成的list，示例如下

<a name="231"></a>
#### 2.3.1 版面分析+表格识别
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

| 字段   | 说明|
| --- |---|
|type| 图片区域的类型 |
|bbox| 图片区域的在原图的坐标，分别[左上角x，左上角y，右下角x，右下角y]|
|res| 图片区域的OCR或表格识别结果。<br> 表格: 一个dict，字段说明如下<br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; `html`: 表格的HTML字符串<br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 在代码使用模式下，前向传入return_ocr_result_in_table=True可以拿到表格中每个文本的检测识别结果，对应为如下字段: <br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; `boxes`: 文本检测坐标<br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; `rec_res`: 文本识别结果。<br> OCR: 一个包含各个单行文字的检测坐标和识别结果的元组 |

运行完成后，每张图片会在`output`字段指定的目录下有一个同名目录，图片里的每个表格会存储为一个excel，图片区域会被裁剪之后保存下来，excel文件和图片名为表格在图片里的坐标。

  ```
  /output/table/1/
    └─ res.txt
    └─ [454, 360, 824, 658].xlsx  表格识别结果
    └─ [16, 2, 828, 305].jpg            被裁剪出的图片区域
    └─ [17, 361, 404, 711].xlsx        表格识别结果
  ```

<a name="232"></a>
#### 2.3.2 DocVQA

请参考：[文档视觉问答](../kie/README.md)。

<a name="24"></a>
### 2.4 参数说明

| 字段 | 说明  | 默认值  |
|---|---|---|
| output | 结果保存地址 | ./output/table |
| table_max_len | 表格结构模型预测时，图像的长边resize尺度 | 488 |
| table_model_dir | 表格结构模型 inference 模型地址| None |
| table_char_dict_path | 表格结构模型所用字典地址 | ../ppocr/utils/dict/table_structure_dict.txt  |
| merge_no_span_structure | 表格识别模型中，是否对'\<td>'和'\</td>' 进行合并 | False |
| layout_model_dir  | 版面分析模型 inference 模型地址 | None |
| layout_dict_path  | 版面分析模型字典| ../ppocr/utils/dict/layout_publaynet_dict.txt |
| layout_score_threshold  | 版面分析模型检测框阈值| 0.5|
| layout_nms_threshold  | 版面分析模型nms阈值| 0.5|
| kie_algorithm  | kie模型算法| LayoutXLM|
| ser_model_dir  | ser模型  inference 模型地址| None|
| ser_dict_path  | ser模型字典| ../train_data/XFUND/class_list_xfun.txt|
| mode | structure or kie  | structure   |
| image_orientation | 前向中是否执行图像方向分类  | False   |
| layout | 前向中是否执行版面分析  | True   |
| table  | 前向中是否执行表格识别  | True   |
| ocr    | 对于版面分析中的非表格区域，是否执行ocr。当layout为False时会被自动设置为False| True |
| recovery    | 前向中是否执行版面恢复| False |
| structure_version |  模型版本，可选 PP-structure和PP-structurev2  | PP-structure |

大部分参数和PaddleOCR whl包保持一致，见 [whl包文档](../../doc/doc_ch/whl.md)
