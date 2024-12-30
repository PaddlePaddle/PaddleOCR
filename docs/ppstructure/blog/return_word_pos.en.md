---
typora-copy-images-to: images
comments: true
---

# Return recognition position

According to the horizontal document, the recognition model not only returns the recognized content, but also the position of each word.

## English document recovery

### Download the inference model first

```bash linenums="1"
cd PaddleOCR/ppstructure

## download model
mkdir inference && cd inference
## Download the detection model of the ultra-lightweight English PP-OCRv3 model and unzip it
https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar && tar xf en_PP-OCRv3_det_infer.tar
## Download the recognition model of the ultra-lightweight English PP-OCRv3 model and unzip it
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar && tar xf en_PP-OCRv3_rec_infer.tar
## Download the ultra-lightweight English table inch model and unzip it
wget https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/paddle3.0b2/en_ppstructure_mobile_v2.0_SLANet_infer.tar
tar xf en_ppstructure_mobile_v2.0_SLANet_infer.tar
## Download the layout model of publaynet dataset and unzip it
wget https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar
tar xf picodet_lcnet_x1_0_fgd_layout_infer.tar
cd ..
```

### Then use the following command inference in the /ppstructure/ directory

```bash linenums="1"
python predict_system.py \
--image_dir=../docs/ppstructure/images/table_1.png \
--det_model_dir=inference/en_PP-OCRv3_det_infer \
--rec_model_dir=inference/en_PP-OCRv3_rec_infer \
--rec_char_dict_path=../ppocr/utils/en_dict.txt \
--table_model_dir=inference/en_ppstructure_mobile_v2.0_SLANet_infer \
--table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt \
--layout_model_dir=inference/picodet_lcnet_x1_0_fgd_layout_infer \
--layout_dict_path=../ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt \
--vis_font_path=../doc/fonts/simfang.ttf \
--recovery=True \
--output=../output/ \
--return_word_box=True
```

### View the visualization of the inference results under `../output/structure/table_1/show_0.jpg`, as shown below

![show_0_mdf_v2](./images/799450d4-d2c5-4b61-b490-e160dc0f515c.jpeg)

## Recover Chinese documents

### Download the inference model first

```bash linenums="1"
cd PaddleOCR/ppstructure

## download model
cd inference
## Download the detection model of the ultra-lightweight Chinese PP-OCRv3 model and unzip it
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar && tar xf ch_PP-OCRv3_det_infer.tar
## Download the recognition model of the ultra-lightweight Chinese PP-OCRv3 model and unzip it
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar && tar xf ch_PP-OCRv3_rec_infer.tar
## Download the ultra-lightweight Chinese table inch model and unzip it
wget https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/paddle3.0b2/ch_ppstructure_mobile_v2.0_SLANet_infer.tar
tar xf ch_ppstructure_mobile_v2.0_SLANet_infer.tar
## Download the layout model of CDLA dataset and unzip it
wget https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar
tar xf picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar
cd ..
```

### Upload the following test image "2.png" to the directory ./docs/table/

![2](./images/d0858341-a889-483c-8373-5ecaa57f3b20.png)

### Then use the following command inference in the /ppstructure/ directory

```bash linenums="1"
python predict_system.py \
--image_dir=./docs/table/2.png \
--det_model_dir=inference/ch_PP-OCRv3_det_infer \
--rec_model_dir=inference/ch_PP-OCRv3_rec_infer \
--rec_char_dict_path=../ppocr/utils/ppocr_keys_v1.txt \
--table_model_dir=inference/ch_ppstructure_mobile_v2.0_SLANet_infer \
--table_char_dict_path=../ppocr/utils/dict/table_structure_dict_ch.txt \
--layout_model_dir=inference/picodet_lcnet_x1_0_fgd_layout_cdla_infer \
--layout_dict_path=../ppocr/utils/dict/layout_dict/layout_cdla_dict.txt \
--vis_font_path=../doc/fonts/chinese_cht.ttf \
--recovery=True \
--output=../output/ \
--return_word_box=True
```

### View the visualization of the inference results under `../output/structure/2/show_0.jpg`, as shown below

![show_1_mdf_v2](./images/3c200538-f2e6-4d79-847a-4c4587efa9f0.jpeg)
