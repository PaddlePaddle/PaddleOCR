cd ppstructure


CUDA_VISIBLE_DEVICES=1 python3.7 predict_system.py \
    --image_dir=docs/table/1.png  \
    --det_model_dir=../models/ch_PP-OCRv3_det_infer \
    --rec_model_dir=../models/ch_PP-OCRv3_rec_infer \
    --rec_char_dict_path=../ppocr/utils/ppocr_keys_v1.txt \
    --show_log=False \
    --output=../output/en/speed/ \
    --table_model_dir=layout/table \
    --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt \
    --table_max_len=488 \
    --layout_model_dir=layout/picodet_lcnet_x2_5_640_publayernet_shape \
    --use_gpu=False \
    --enable_mkldnn=True \
    --vis_font_path=../doc/fonts/simfang.ttf \
    --use_tensorrt=False

python3.7 predict_system1.py \
    --image_dir=docs/table/1.png  \
    --det_model_dir=../models/ch_PP-OCRv3_det_infer \
    --rec_model_dir=../models/ch_PP-OCRv3_rec_infer \
    --rec_char_dict_path=../ppocr/utils/ppocr_keys_v1.txt \
    --show_log=True \
    --output=../output/en/speed/ \
    --table_model_dir=../en_ppocr_mobile_v2.0_table_structure_infer \
    --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt \
    --table_max_len=488 \
    --use_gpu=False \
    --enable_mkldnn=True \
    --vis_font_path=../doc/fonts/simfang.ttf \
    --use_tensorrt=False

CUDA_VISIBLE_DEVICES=7 python3.7 layout/predict_layout.py \
    --layout_model_dir=layout/picodet_lcnet_x2_5_640_publayernet_shape \
    --image_dir=docs/table/1.png \
    --use_gpu=False
    
cd ../


