python tools/infer/predict_system.py --image_dir="E:/image/OCR/FangZheng/0107/OCR" --det_model_dir="./inference/ch_ppocr_mobile_v1.1_det_infer/"  --rec_model_dir="./inference/ch_ppocr_mobile_v1.1_rec_infer/" --cls_model_dir="./inference/ch_ppocr_mobile_v1.1_cls_infer/" --use_angle_cls=True --use_space_char=True

python tools/infer/predict_system.py --image_dir="E:\\image\\OCR\\err" --det_model_dir="./inference/ch_ppocr_mobile_v2.0_det_infer/"  --rec_model_dir="./inference/ch_ppocr_mobile_v2.0_rec_infer/" --cls_model_dir="./inference/ch_ppocr_mobile_v2.0_cls_infer/" --use_angle_cls=True --use_space_char=True

python -m paddle.distributed.launch --gpu '0'  tools/train.py -c configs/rec/ch_ppocr_v1.1/rec_chinese_common_train_v1.1.yml