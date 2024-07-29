---
comments: true
---

# Python Inference

## 1. Layout Structured Analysis

Go to the `ppstructure` directory

```bash linenums="1"
cd ppstructure

# download model
mkdir inference && cd inference
# Download the PP-StructureV2 layout analysis model and unzip it
wget https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_layout_infer.tar && tar xf picodet_lcnet_x1_0_layout_infer.tar
# Download the PP-OCRv3 text detection model and unzip it
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar && tar xf ch_PP-OCRv3_det_infer.tar
# Download the PP-OCRv3 text recognition model and unzip it
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar && tar xf ch_PP-OCRv3_rec_infer.tar
# Download the PP-StructureV2 form recognition model and unzip it
wget https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_infer.tar && tar xf ch_ppstructure_mobile_v2.0_SLANet_infer.tar
cd ..
```

### 1.1 layout analysis + table recognition

```bash linenums="1"
python3 predict_system.py --det_model_dir=inference/ch_PP-OCRv3_det_infer \
                          --rec_model_dir=inference/ch_PP-OCRv3_rec_infer \
                          --table_model_dir=inference/ch_ppstructure_mobile_v2.0_SLANet_infer \
                          --layout_model_dir=inference/picodet_lcnet_x1_0_layout_infer \
                          --image_dir=./docs/table/1.png \
                          --rec_char_dict_path=../ppocr/utils/ppocr_keys_v1.txt \
                          --table_char_dict_path=../ppocr/utils/dict/table_structure_dict_ch.txt \
                          --output=../output \
                          --vis_font_path=../doc/fonts/simfang.ttf
```

After the operation is completed, each image will have a directory with the same name in the `structure` directory under the directory specified by the `output` field. Each table in the image will be stored as an excel, and the picture area will be cropped and saved. The filename of excel and picture is their coordinates in the image. Detailed results are stored in the `res.txt` file.

### 1.2 layout analysis

```bash linenums="1"
python3 predict_system.py --layout_model_dir=inference/picodet_lcnet_x1_0_layout_infer \
                          --image_dir=./docs/table/1.png \
                          --output=../output \
                          --table=false \
                          --ocr=false
```

After the operation is completed, each image will have a directory with the same name in the `structure` directory under the directory specified by the `output` field. Each picture in image will be cropped and saved. The filename of picture area is their coordinates in the image. Layout analysis results will be stored in the `res.txt` file

### 1.3 table recognition

```bash linenums="1"
python3 predict_system.py --det_model_dir=inference/ch_PP-OCRv3_det_infer \
                          --rec_model_dir=inference/ch_PP-OCRv3_rec_infer \
                          --table_model_dir=inference/ch_ppstructure_mobile_v2.0_SLANet_infer \
                          --image_dir=./docs/table/table.jpg \
                          --rec_char_dict_path=../ppocr/utils/ppocr_keys_v1.txt \
                          --table_char_dict_path=../ppocr/utils/dict/table_structure_dict_ch.txt \
                          --output=../output \
                          --vis_font_path=../doc/fonts/simfang.ttf \
                          --layout=false
```

After the operation is completed, each image will have a directory with the same name in the `structure` directory under the directory specified by the `output` field. Each table in the image will be stored as an excel. The filename of excel is their coordinates in the image.

## 2. Key Information Extraction

### 2.1 SER

```bash linenums="1"
cd ppstructure

mkdir inference && cd inference
# download model
wget https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/ser_vi_layoutxlm_xfund_infer.tar && tar -xf ser_vi_layoutxlm_xfund_infer.tar
cd ..
python3 predict_system.py \
  --kie_algorithm=LayoutXLM \
  --ser_model_dir=./inference/ser_vi_layoutxlm_xfund_infer \
  --image_dir=./docs/kie/input/zh_val_42.jpg \
  --ser_dict_path=../ppocr/utils/dict/kie_dict/xfund_class_list.txt \
  --vis_font_path=../doc/fonts/simfang.ttf \
  --ocr_order_method="tb-yx" \
  --mode=kie
```

After the operation is completed, each image will store the visualized image in the `kie` directory under the directory specified by the `output` field, and the image name is the same as the input image name.

### 2.2 RE+SER

```bash linenums="1"
cd ppstructure

mkdir inference && cd inference
# download model
wget https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/ser_vi_layoutxlm_xfund_infer.tar && tar -xf ser_vi_layoutxlm_xfund_infer.tar
wget https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/re_vi_layoutxlm_xfund_infer.tar && tar -xf re_vi_layoutxlm_xfund_infer.tar
cd ..

python3 predict_system.py \
  --kie_algorithm=LayoutXLM \
  --re_model_dir=./inference/re_vi_layoutxlm_xfund_infer \
  --ser_model_dir=./inference/ser_vi_layoutxlm_xfund_infer \
  --image_dir=./docs/kie/input/zh_val_42.jpg \
  --ser_dict_path=../ppocr/utils/dict/kie_dict/xfund_class_list.txt \
  --vis_font_path=../doc/fonts/simfang.ttf \
  --ocr_order_method="tb-yx" \
  --mode=kie
```

After the operation is completed, each image will have a directory with the same name in the `kie` directory under the directory specified by the `output` field, where the visual images and prediction results are stored.
