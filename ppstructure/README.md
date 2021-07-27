# PaddleStructure

install layoutparser
```sh
wget  https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl
pip3 install layoutparser-0.0.0-py3-none-any.whl
```

## 1. Introduction to pipeline

PaddleStructure is a toolkit for complex layout text OCR, the process is as follows

![pipeline](../doc/table/pipeline.jpg)

In PaddleStructure, the image will be analyzed by layoutparser first. In the layout analysis, the area in the image will be classified, and the OCR process will be carried out according to the category.

Currently layoutparser will output five categories:
1. Text
2. Title
3. Figure
4. List
5. Table
   
Types 1-4 follow the traditional OCR process, and 5 follow the Table OCR process.

## 2. LayoutParser


## 3. Table OCR

[doc](table/README.md)

## 4. Predictive by inference engine

Use the following commands to complete the inference
```python
python3 table/predict_system.py --det_model_dir=path/to/det_model_dir --rec_model_dir=path/to/rec_model_dir --table_model_dir=path/to/table_model_dir --image_dir=../doc/table/1.png --rec_char_dict_path=../ppocr/utils/dict/table_dict.txt --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt --rec_char_type=EN --det_limit_side_len=736 --det_limit_type=min --output ../output/table
```
After running, each image will have a directory with the same name under the directory specified in the output field. Each table in the picture will be stored as an excel, and the excel file name will be the coordinates of the table in the image.

## 5. PaddleStructure whl package introduction

### 5.1 Use

5.1.1 Use by code
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

font_path = 'path/tp/PaddleOCR/doc/fonts/simfang.ttf'
image = Image.open(img_path).convert('RGB')
im_show = draw_result(image, result,font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

5.1.2 Use by command line
```bash
paddlestructure --image_dir=../doc/table/1.png
```

### Parameter Description
Most of the parameters are consistent with the paddleocr whl package, see [whl package documentation](../doc/doc_ch/whl.md)

| Parameter                    | Description                                            | Default           |
|------------------------|------------------------------------------------------|------------------|
| output                 | The path where excel and recognition results are saved                    | ./output/table            |
| structure_max_len      |  When the table structure model predicts, the long side of the image is resized             |  488            |
| structure_model_dir      |  Table structure inference model path             |  None            |
| structure_char_type      | Dictionary path used by table structure model             |  ../ppocr/utils/dict/table_structure_dict.tx            |


