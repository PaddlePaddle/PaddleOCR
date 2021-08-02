# PPStructure

PPStructure is an OCR toolkit for complex layout analysis. It can divide document data in the form of pictures into **text, table, title, picture and list** 5 types of areas, and extract the table area as excel
## 1. Quick start

### install
**install PaddlePaddle2.0**

```bash
pip3 install --upgrade pip

# If you have cuda9 or cuda10 installed on your machine, please run the following command to install
python3 -m pip install paddlepaddle-gpu==2.0.0 -i https://mirror.baidu.com/pypi/simple

# If you only have cpu on your machine, please run the following command to install

python3 -m pip install paddlepaddle==2.0.0 -i https://mirror.baidu.com/pypi/simple

For more version requirements, please refer to the instructions in the [installation document](https://www.paddlepaddle.org.cn/install/quick) .
```

**Clone PaddleOCR repo**

```bash
# Recommend
git clone https://github.com/PaddlePaddle/PaddleOCR

# If you cannot pull successfully due to network problems, you can also choose to use the code hosting on the cloud:
git clone https://gitee.com/paddlepaddle/PaddleOCR

# Note: The cloud-hosting code may not be able to synchronize the update with this GitHub project in real time. There might be a delay of 3-5 days. Please give priority to the recommended method.
```

**install paddleocr**

install by pypi
```bash
cd PaddleOCR
pip install "paddleocr>=2.2" #  # Recommend to use version 2.2
```

build own whl package and install

```bash
python3 setup.py bdist_wheel
pip3 install dist/paddleocr-x.x.x-py3-none-any.whl # x.x.x is the version of paddleocr
```
**install layoutparser**
```sh
pip3 install -U premailer https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl
```

### 1.2 Use

#### 1.2.1 Use by command line

```bash
paddleocr --image_dir=../doc/table/1.png --type=structure
```

#### 1.2.2 Use by code

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

font_path = '../doc/fonts/simfang.ttf'
image = Image.open(img_path).convert('RGB')
im_show = draw_structure_result(image, result,font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```
#### 1.2.3 返回结果说明
The return result of PPStructure is a list composed of a dict, an example is as follows

```shell
[
  {   'type': 'Text', 
      'bbox': [34, 432, 345, 462], 
      'res': ([[36.0, 437.0, 341.0, 437.0, 341.0, 446.0, 36.0, 447.0], [41.0, 454.0, 125.0, 453.0, 125.0, 459.0, 41.0, 460.0]], 
                [('Tigure-6. The performance of CNN and IPT models using difforen', 0.90060663), ('Tent  ', 0.465441)])
  }
]
```
The description of each field in dict is as follows

| Parameter            | Description           | 
| --------------- | -------------|
|type|Type of image area|
|bbox|The coordinates of the image area in the original image, respectively [left upper x, left upper y, right bottom x, right bottom y]|
|res|OCR or table recognition result of image area。<br> Table: HTML string of the table; <br> OCR: A tuple containing the detection coordinates and recognition results of each single line of text|


#### 1.2.4 Parameter Description：

| Parameter            | Description                                     | Default value                                        |
| --------------- | ---------------------------------------- | ------------------------------------------- |
| output          | The path where excel and recognition results are saved                | ./output/table                              |
| table_max_len   | The long side of the image is resized in table structure model  | 488                                         |
| table_model_dir | inference model path of table structure model          | None                                        |
| table_char_type | dict path of table structure model                 | ../ppocr/utils/dict/table_structure_dict.tx |

Most of the parameters are consistent with the paddleocr whl package, see [doc of whl](../doc/doc_en/whl_en.md)

After running, each image will have a directory with the same name under the directory specified in the output field. Each table in the picture will be stored as an excel and figure area will be cropped and saved, the excel and image file name will be the coordinates of the table in the image.

## 2. PPStructure Pipeline

the process is as follows
![pipeline](../doc/table/pipeline_en.jpg)

In PPStructure, the image will be analyzed by layoutparser first. In the layout analysis, the area in the image will be classified, including **text, title, image, list and table** 5 categories. For the first 4 types of areas, directly use the PP-OCR to complete the text detection and recognition. The table area will  be converted to an excel file of the same table style via Table OCR.

### 2.1 LayoutParser

Layout analysis divides the document data into regions, including the use of Python scripts for layout analysis tools, extraction of special category detection boxes, performance indicators, and custom training layout analysis models. For details, please refer to [document](layout/README_en.md).

### 2.2 Table Structure

Table OCR converts table image into excel documents, which include the detection and recognition of table text and the prediction of table structure and cell coordinates. For detailed, please refer to [document](table/README.md)

## 3. Predictive by inference engine

Use the following commands to complete the inference. 

```python
cd PaddleOCR/ppstructure

# download model
mkdir inference && cd inference
# Download the detection model of the ultra-lightweight Chinese OCR model and uncompress it
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_ppocr_mobile_v2.0_det_infer.tar
# Download the recognition model of the ultra-lightweight Chinese OCR model and uncompress it
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar
# Download the table structure model of the ultra-lightweight Chinese OCR model and uncompress it
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar && tar xf en_ppocr_mobile_v2.0_table_structure_infer.tar
cd ..

python3 table/predict_system.py --det_model_dir=inference/ch_ppocr_mobile_v2.0_det_infer --rec_model_dir=inference/ch_ppocr_mobile_v2.0_rec_infer --table_model_dir=inference/en_ppocr_mobile_v2.0_table_structure_infer --image_dir=../doc/table/1.png --rec_char_dict_path=../ppocr/utils/ppocr_keys_v1.txt --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt --rec_char_type=ch --det_limit_side_len=736 --det_limit_type=min --output=../output/table --vis_font_path=../doc/fonts/simfang.ttf
```
After running, each image will have a directory with the same name under the directory specified in the output field. Each table in the picture will be stored as an excel and figure area will be cropped and saved, the excel and image file name will be the coordinates of the table in the image.

**Model List**


|model name|description|config|model size|download|
| --- | --- | --- | --- | --- |
|en_ppocr_mobile_v2.0_table_structure|Table structure prediction for English table scenarios|[table_mv3.yml](../configs/table/table_mv3.yml)|18.6M|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar) |