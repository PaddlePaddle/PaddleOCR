cd ppstructure


# CUDA_VISIBLE_DEVICES=1 python3.7 table/eval_table.py \
#     --det_model_dir=/ssd1/zhoujun20/table/ch/PaddleOCR/output/baseline/en_ppocr_mobile_v2.0_table_det_infer \
#     --rec_model_dir=/ssd1/zhoujun20/table/ch/PaddleOCR/output/baseline/en_ppocr_mobile_v2.0_table_rec_infer \
#     --rec_char_dict_path=../ppocr/utils/dict/table_dict.txt \
#     --det_limit_side_len=736 \
#     --det_limit_type=min \
#     --rec_image_shape=3,32,320 \
#     --show_log=False \
#     --gt_path=/ssd1/zhoujun20/table/ch/PaddleOCR/output/baseline/PubTabNet_eval_gt.txt \
#     --image_dir=/home/zhoujun20/table/PubTabNe/pubtabnet/val/ \
#     --output=../output/en/PaddleOCR_TorchModel_TorchMatch/ \
#     --table_model_dir=/ssd1/zhoujun20/flk/PaddleOCR/table_structure_tablemaster_infer/ \
#     --table_algorithm=TableMaster \
#     --table_char_dict_path=../ppocr/utils/dict/table_master_structure_dict.txt \
#     --table_max_len=480 \
#     --chunk=2

# CUDA_VISIBLE_DEVICES=6 python3.7 table/eval_table1.py \
#     --det_model_dir=/ssd1/zhoujun20/table/ch/PaddleOCR/output/baseline/en_ppocr_mobile_v2.0_table_det_infer \
#     --rec_model_dir=/ssd1/zhoujun20/table/ch/PaddleOCR/output/baseline/en_ppocr_mobile_v2.0_table_rec_infer \
#     --rec_char_dict_path=../ppocr/utils/dict/table_dict.txt \
#     --det_limit_side_len=736 \
#     --det_limit_type=min \
#     --rec_image_shape=3,32,320 \
#     --rec_algorithm='CRNN' \
#     --show_log=False \
#     --gt_path=/ssd1/zhoujun20/table/ch/PaddleOCR/output/baseline/PubTabNet_eval_gt.txt \
#     --image_dir=/home/zhoujun20/table/PubTabNe/pubtabnet/val/ \
#     --output=../output/en/result_new/ \
#     --table_model_dir=../output/table_master/infer \
#     --table_algorithm=TableMaster \
#     --table_char_dict_path=../ppocr/utils/dict/table_master_structure_dict.txt \
#     --table_max_len=480 \
#     --chunk=1

# CUDA_VISIBLE_DEVICES=1 python3.7 table/eval_table2.py \
#     --det_model_dir=/ssd1/zhoujun20/table/ch/PaddleOCR/output/baseline/en_ppocr_mobile_v2.0_table_det_infer \
#     --rec_model_dir=/ssd1/zhoujun20/table/ch/PaddleOCR/output/baseline/en_ppocr_mobile_v2.0_table_rec_infer \
#     --rec_char_dict_path=../ppocr/utils/dict/table_dict.txt \
#     --det_limit_side_len=736 \
#     --det_limit_type=min \
#     --rec_image_shape=3,32,320 \
#     --show_log=False \
#     --gt_path=/ssd1/zhoujun20/table/ch/PaddleOCR/output/baseline/PubTabNet_eval_gt.txt \
#     --image_dir=/home/zhoujun20/table/PubTabNe/pubtabnet/val/ \
#     --output=../output/en/table_lcnet_1_0_csp_pan_pretrain_ssld/ \
#     --table_model_dir=/ssd1/zhoujun20/flk/PaddleOCR/table_structure_tablemaster_infer/ \
#     --table_algorithm=TableMaster \
#     --table_char_dict_path=../ppocr/utils/dict/table_master_structure_dict.txt \
#     --table_max_len=480 

# /ssd1/zhoujun20/flk/PaddleOCR/table_structure_tablemaster_infer/
# output/table_master/infer

CUDA_VISIBLE_DEVICES=0 python3.7 table/eval_table2.py \
    --det_model_dir=/ssd1/zhoujun20/table/ch/PaddleOCR/output/baseline/en_ppocr_mobile_v2.0_table_det_infer \
    --rec_model_dir=/ssd1/zhoujun20/table/ch/PaddleOCR/output/baseline/en_ppocr_mobile_v2.0_table_rec_infer \
    --table_model_dir=/ssd1/zhoujun20/table/ch/PaddleOCR/output/en/table_lcnet_1_0_csp_pan_headsv3_smooth_l1_pretrain_ssld_weight81_sync_bn/infer \
    --rec_char_dict_path=/ssd1/zhoujun20/table/ch/PaddleOCR/ppocr/utils/dict/table_dict.txt \
    --table_char_dict_path=/ssd1/zhoujun20/table/ch/PaddleOCR/ppocr/utils/dict/table_structure_dict.txt \
    --det_limit_side_len=736 \
    --det_limit_type=min \
    --rec_algorithm='CRNN' \
    --show_log=False \
    --gt_path=/ssd1/zhoujun20/table/ch/PaddleOCR/output/baseline/PubTabNet_eval_gt.txt \
    --image_dir=/home/zhoujun20/table/PubTabNe/pubtabnet/val/ \
    --output=../output/en/77.7_500/ \
    --table_max_len=488 \
    --rec_image_shape=3,32,320

# CUDA_VISIBLE_DEVICES=0 python3 table/predict_det_rec.py --det_model_dir=../output/baseline/en_ppocr_mobile_v2.0_table_det_infer --rec_model_dir=../output/baseline/en_ppocr_mobile_v2.0_table_rec_infer --rec_char_dict_path=../ppocr/utils/dict/table_dict.txt --det_limit_side_len=736 --det_limit_type=min --show_log=False --image_dir=/home/zhoujun20/table/PubTabNe/pubtabnet/train/ --output=../output/det_rec_train/

cd --