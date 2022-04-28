# Python Inference

- [1. Structure](#1)
  - [1.1 layout analysis + table recognition](#1.1)
  - [1.2 layout analysis](#1.2)
  - [1.3 table recognition](#1.3)
- [2. DocVQA](#2)

<a name="1"></a>
## 1. Structure
Go to the `ppstructure` directory

```bash
cd ppstructure
````

download model

```bash
mkdir inference && cd inference
# Download the PP-OCRv2 text detection model and unzip it
wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_slim_quant_infer.tar && tar xf ch_PP-OCRv2_det_slim_quant_infer.tar
# Download the PP-OCRv2 text recognition model and unzip it
wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_infer.tar && tar xf ch_PP-OCRv2_rec_slim_quant_infer.tar
# Download the ultra-lightweight English table structure model and unzip it
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar && tar xf en_ppocr_mobile_v2.0_table_structure_infer.tar
cd ..
```
<a name="1.1"></a>
### 1.1 layout analysis + table recognition
```bash
python3 predict_system.py --det_model_dir=inference/ch_PP-OCRv2_det_slim_quant_infer \
                          --rec_model_dir=inference/ch_PP-OCRv2_rec_slim_quant_infer \
                          --table_model_dir=inference/en_ppocr_mobile_v2.0_table_structure_infer \
                          --image_dir=./docs/table/1.png \
                          --rec_char_dict_path=../ppocr/utils/ppocr_keys_v1.txt \
                          --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt \
                          --output=../output \
                          --vis_font_path=../doc/fonts/simfang.ttf
```
After the operation is completed, each image will have a directory with the same name in the `structure` directory under the directory specified by the `output` field. Each table in the image will be stored as an excel, and the picture area will be cropped and saved. The filename of excel and picture is their coordinates in the image. Detailed results are stored in the `res.txt` file.

<a name="1.2"></a>
### 1.2 layout analysis
```bash
python3 predict_system.py --image_dir=./docs/table/1.png --table=false --ocr=false --output=../output/
```
After the operation is completed, each image will have a directory with the same name in the `structure` directory under the directory specified by the `output` field. Each picture in image will be cropped and saved. The filename of picture area is their coordinates in the image. Layout analysis results will be stored in the `res.txt` file

<a name="1.3"></a>
### 1.3 table recognition
```bash
python3 predict_system.py --det_model_dir=inference/ch_PP-OCRv2_det_slim_quant_infer \
                          --rec_model_dir=inference/ch_PP-OCRv2_rec_slim_quant_infer \
                          --table_model_dir=inference/en_ppocr_mobile_v2.0_table_structure_infer \
                          --image_dir=./docs/table/table.jpg \
                          --rec_char_dict_path=../ppocr/utils/ppocr_keys_v1.txt \
                          --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt \
                          --output=../output \
                          --vis_font_path=../doc/fonts/simfang.ttf \
                          --layout=false
```
After the operation is completed, each image will have a directory with the same name in the `structure` directory under the directory specified by the `output` field. Each table in the image will be stored as an excel. The filename of excel is their coordinates in the image.

<a name="2"></a>
## 2. DocVQA

```bash
cd ppstructure

# download model
mkdir inference && cd inference
wget https://paddleocr.bj.bcebos.com/pplayout/PP-Layout_v1.0_ser_pretrained.tar && tar xf PP-Layout_v1.0_ser_pretrained.tar
cd ..

python3 predict_system.py --model_name_or_path=vqa/PP-Layout_v1.0_ser_pretrained/ \
                          --mode=vqa \
                          --image_dir=vqa/images/input/zh_val_0.jpg  \
                          --vis_font_path=../doc/fonts/simfang.ttf
```
After the operation is completed, each image will store the visualized image in the `vqa` directory under the directory specified by the `output` field, and the image name is the same as the input image name.
