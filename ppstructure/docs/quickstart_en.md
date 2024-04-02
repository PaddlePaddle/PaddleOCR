# PP-Structure Quick Start

- [1. Environment Preparation](#1-environment-preparation)
- [2. Quick Use](#2-quick-use)
  - [2.1 Use by command line](#21-use-by-command-line)
    - [2.1.1 image orientation + layout analysis + table recognition](#211-image-orientation--layout-analysis--table-recognition)
    - [2.1.2 layout analysis + table recognition](#212-layout-analysis--table-recognition)
    - [2.1.3 layout analysis](#213-layout-analysis)
    - [2.1.4 table recognition](#214-table-recognition)
    - [2.1.5 Key Information Extraction](#215-Key-Information-Extraction)
    - [2.1.6 layout recovery](#216-layout-recovery)
  - [2.2 Use by python script](#22-use-by-python-script)
    - [2.2.1 image orientation + layout analysis + table recognition](#221-image-orientation--layout-analysis--table-recognition)
    - [2.2.2 layout analysis + table recognition](#222-layout-analysis--table-recognition)
    - [2.2.3 layout analysis](#223-layout-analysis)
    - [2.2.4 table recognition](#224-table-recognition)
    - [2.2.5 Key Information Extraction](#225-Key-Information-Extraction)
    - [2.2.6 layout recovery](#226-layout-recovery)  
  - [2.3 Result description](#23-result-description)
    - [2.3.1 layout analysis + table recognition](#231-layout-analysis--table-recognition)
    - [2.3.2 Key Information Extraction](#232-Key-Information-Extraction)
  - [2.4 Parameter Description](#24-parameter-description)
- [3. Summary](#3-summary)


<a name="1"></a>
## 1. Environment Preparation
### 1.1 Install PaddlePaddle

> If you do not have a Python environment, please refer to [Environment Preparation](./environment_en.md).

- If you have CUDA 9 or CUDA 10 installed on your machine, please run the following command to install

  ```bash
  python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
  ```

- If you have no available GPU on your machine, please run the following command to install the CPU version

  ```bash
  python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
  ```

For more software version requirements, please refer to the instructions in [Installation Document](https://www.paddlepaddle.org.cn/install/quick) for operation.

### 1.2 Install PaddleOCR Whl Package

```bash
# Install paddleocr, version 2.6 is recommended
pip3 install "paddleocr>=2.6.0.3"

# Install the image direction classification dependency package paddleclas (if you do not use the image direction classification, you can skip it)
pip3 install paddleclas>=2.4.3
```

<a name="2"></a>

## 2. Quick Use

<a name="21"></a>
### 2.1 Use by command line

<a name="211"></a>
#### 2.1.1 image orientation + layout analysis + table recognition
```bash
paddleocr --image_dir=ppstructure/docs/table/1.png --type=structure --image_orientation=true
```

<a name="212"></a>
#### 2.1.2 layout analysis + table recognition
```bash
paddleocr --image_dir=ppstructure/docs/table/1.png --type=structure
```

<a name="213"></a>
#### 2.1.3 layout analysis
```bash
paddleocr --image_dir=ppstructure/docs/table/1.png --type=structure --table=false --ocr=false
```

<a name="214"></a>
#### 2.1.4 table recognition
```bash
paddleocr --image_dir=ppstructure/docs/table/table.jpg --type=structure --layout=false
```

<a name="215"></a>

#### 2.1.5 Key Information Extraction

Key information extraction does not currently support use by the whl package. For detailed usage tutorials, please refer to: [inference document](./inference_en.md).

<a name="216"></a>
#### 2.1.6 layout recovery(PDF to Word)

Two layout recovery methods are provided, For detailed usage tutorials, please refer to: [Layout Recovery](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/ppstructure/recovery/README.md).

- PDF parse
- OCR

Recovery by using PDF parse (only support pdf as input):

```bash
paddleocr --image_dir=ppstructure/recovery/UnrealText.pdf --type=structure --recovery=true --use_pdf2docx_api=true
```

Recovery by using OCR：

```bash
paddleocr --image_dir=ppstructure/docs/table/1.png --type=structure --recovery=true --lang='en'
```

<a name="22"></a>
### 2.2 Use by python script

<a name="221"></a>
#### 2.2.1 image orientation + layout analysis + table recognition

```python
import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res

table_engine = PPStructure(show_log=True, image_orientation=True)

save_folder = './output'
img_path = 'ppstructure/docs/table/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

from PIL import Image

font_path = 'doc/fonts/simfang.ttf' # PaddleOCR下提供字体包
image = Image.open(img_path).convert('RGB')
im_show = draw_structure_result(image, result,font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

<a name="222"></a>
#### 2.2.2 layout analysis + table recognition

```python
import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res

table_engine = PPStructure(show_log=True)

save_folder = './output'
img_path = 'ppstructure/docs/table/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

from PIL import Image

font_path = 'doc/fonts/simfang.ttf' # font provided in PaddleOCR
image = Image.open(img_path).convert('RGB')
im_show = draw_structure_result(image, result,font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

<a name="223"></a>
#### 2.2.3 layout analysis

```python
import os
import cv2
from paddleocr import PPStructure,save_structure_res

table_engine = PPStructure(table=False, ocr=False, show_log=True)

save_folder = './output'
img_path = 'ppstructure/docs/table/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)
```

<a name="224"></a>
#### 2.2.4 table recognition

```python
import os
import cv2
from paddleocr import PPStructure,save_structure_res

table_engine = PPStructure(layout=False, show_log=True)

save_folder = './output'
img_path = 'ppstructure/docs/table/table.jpg'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)
```

<a name="225"></a>
#### 2.2.5 Key Information Extraction

Key information extraction does not currently support use by the whl package. For detailed usage tutorials, please refer to: [Key Information Extraction](../kie/README.md).

<a name="226"></a>
#### 2.2.6 layout recovery

```python
import os
import cv2
from paddleocr import PPStructure,save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx

# Chinese image
table_engine = PPStructure(recovery=True)
# English image
# table_engine = PPStructure(recovery=True, lang='en')

save_folder = './output'
img_path = 'ppstructure/docs/table/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

h, w, _ = img.shape
res = sorted_layout_boxes(result, w)
convert_info_docx(img, res, save_folder, os.path.basename(img_path).split('.')[0])
```

<a name="23"></a>
### 2.3 Result description

The return of PP-Structure is a list of dicts, the example is as follows:

<a name="231"></a>
#### 2.3.1 layout analysis + table recognition
```shell
[
  {   'type': 'Text',
      'bbox': [34, 432, 345, 462],
      'res': ([[36.0, 437.0, 341.0, 437.0, 341.0, 446.0, 36.0, 447.0], [41.0, 454.0, 125.0, 453.0, 125.0, 459.0, 41.0, 460.0]],
                [('Tigure-6. The performance of CNN and IPT models using difforen', 0.90060663), ('Tent  ', 0.465441)])
  }
]
```
Each field in dict is described as follows:

| field | description  |
| --- |---|
|type| Type of image area. |
|bbox| The coordinates of the image area in the original image, respectively [upper left corner x, upper left corner y, lower right corner x, lower right corner y]. |
|res| OCR or table recognition result of the image area. <br> table: a dict with field descriptions as follows: <br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; `html`: html str of table.<br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; In the code usage mode, set return_ocr_result_in_table=True whrn call can get the detection and recognition results of each text in the table area, corresponding to the following fields: <br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; `boxes`: text detection boxes.<br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; `rec_res`: text recognition results.<br> OCR: A tuple containing the detection boxes and recognition results of each single text. |

After the recognition is completed, each image will have a directory with the same name under the directory specified by the `output` field. Each table in the image will be stored as an excel, and the picture area will be cropped and saved. The filename of  excel and picture is their coordinates in the image.
  ```
  /output/table/1/
    └─ res.txt
    └─ [454, 360, 824, 658].xlsx        table recognition result
    └─ [16, 2, 828, 305].jpg            picture in Image
    └─ [17, 361, 404, 711].xlsx        table recognition result
  ```

<a name="232"></a>
#### 2.3.2 Key Information Extraction

Please refer to: [Key Information Extraction](../kie/README.md) .

<a name="24"></a>
### 2.4 Parameter Description

| field | description | default |
|---|---|---|
| output | result save path | ./output/table |
| table_max_len | long side of the image resize in table structure model | 488 |
| table_model_dir | Table structure model inference model path| None |
| table_char_dict_path | The dictionary path of table structure model | ../ppocr/utils/dict/table_structure_dict.txt  |
| merge_no_span_structure | In the table recognition model, whether to merge '\<td>' and '\</td>' | False |
| layout_model_dir  | Layout analysis model inference model path| None |
| layout_dict_path  | The dictionary path of layout analysis model| ../ppocr/utils/dict/layout_publaynet_dict.txt |
| layout_score_threshold  | The box threshold path of layout analysis model| 0.5|
| layout_nms_threshold  | The nms threshold path of layout analysis model| 0.5|
| kie_algorithm  | kie model algorithm| LayoutXLM|
| ser_model_dir  | Ser model inference model path| None|
| ser_dict_path  | The dictionary path of Ser model| ../train_data/XFUND/class_list_xfun.txt|
| mode | structure or kie  | structure   |
| image_orientation | Whether to perform image orientation classification in forward  | False   |
| layout | Whether to perform layout analysis in forward  | True   |
| table  | Whether to perform table recognition in forward  | True   |
| ocr    | Whether to perform ocr for non-table areas in layout analysis. When layout is False, it will be automatically set to False| True |
| recovery    | Whether to perform layout recovery in forward| False |
| save_pdf    | Whether to convert docx to pdf when recovery| False |
| structure_version |  Structure version, optional PP-structure and PP-structurev2  | PP-structure |

Most of the parameters are consistent with the PaddleOCR whl package, see [whl package documentation](../../doc/doc_en/whl_en.md)

<a name="3"></a>
## 3. Summary

Through the content in this section, you can master the use of PP-Structure related functions through PaddleOCR whl package. Please refer to [documentation tutorial](../../README.md) for more detailed usage tutorials including model training, inference and deployment, etc.
