export CUDA_VISIBLE_DEVICES=6
# python3.7 infer_ser_e2e.py \
#     --model_name_or_path "output/ser_distributed/best_model" \
#     --max_seq_length 512 \
#     --output_dir "output_res_e2e/" \
#     --infer_imgs "/ssd1/zhoujun20/VQA/data/XFUN_v1.0_data/zh.val/zh_val_0.jpg"


# python3.7 infer_ser_re_e2e.py \
#     --model_name_or_path "output/ser_distributed/best_model" \
#     --re_model_name_or_path "output/re_test/best_model" \
#     --max_seq_length 512 \
#     --output_dir "output_ser_re_e2e_train/" \
#     --infer_imgs "images/input/zh_val_21.jpg"

# python3.7 infer_ser.py \
#     --model_name_or_path "output/ser_LayoutLM/best_model" \
#     --ser_model_type "LayoutLM" \
#     --output_dir "ser_LayoutLM/" \
#     --infer_imgs "images/input/zh_val_21.jpg" \
#     --ocr_json_path "/ssd1/zhoujun20/VQA/data/XFUN_v1.0_data/xfun_normalize_val.json"

python3.7 infer_ser.py \
    --model_name_or_path "output/ser_new/best_model" \
    --ser_model_type "LayoutXLM" \
    --output_dir "ser_new/" \
    --infer_imgs "images/input/zh_val_21.jpg" \
    --ocr_json_path "/ssd1/zhoujun20/VQA/data/XFUN_v1.0_data/xfun_normalize_val.json"

# python3.7 infer_ser_e2e.py \
#     --model_name_or_path "output/ser_new/best_model" \
#     --ser_model_type "LayoutXLM" \
#     --max_seq_length 512 \
#     --output_dir "output/ser_new/" \
#     --infer_imgs "images/input/zh_val_0.jpg"


# python3.7 infer_ser_e2e.py \
#     --model_name_or_path "output/ser_LayoutLM/best_model" \
#     --ser_model_type "LayoutLM" \
#     --max_seq_length 512 \
#     --output_dir "output/ser_LayoutLM/" \
#     --infer_imgs "images/input/zh_val_0.jpg"

# python3 infer_re.py \
#     --model_name_or_path "/ssd1/zhoujun20/VQA/PaddleOCR/ppstructure/vqa/output/re_test/best_model/" \
#     --max_seq_length 512 \
#     --eval_data_dir "/ssd1/zhoujun20/VQA/data/XFUN_v1.0_data/zh.val" \
#     --eval_label_path "/ssd1/zhoujun20/VQA/data/XFUN_v1.0_data/xfun_normalize_val.json" \
#     --label_map_path 'labels/labels_ser.txt' \
#     --output_dir "output_res"  \
#     --per_gpu_eval_batch_size 1 \
#     --seed 2048

# python3.7 infer_ser_re_e2e.py \
#     --model_name_or_path "output/ser_LayoutLM/best_model" \
#     --ser_model_type "LayoutLM" \
#     --re_model_name_or_path "output/re_new/best_model" \
#     --max_seq_length 512 \
#     --output_dir "output_ser_re_e2e/" \
#     --infer_imgs "images/input/zh_val_21.jpg"