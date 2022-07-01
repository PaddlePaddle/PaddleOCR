python3.7 vqa/predict_vqa_token_ser.py --vqa_algorithm=LayoutXLM --ser_model_dir=../models/ser_LayoutXLM_xfun_zh/infer --ser_dict_path=../train_data/XFUND/class_list_xfun.txt --image_dir=docs/vqa/input/zh_val_42.jpg


python3.7 tools/infer_vqa_token_ser_re.py -c configs/vqa/re/layoutxlm.yml -o Architecture.Backbone.checkpoints=models/re_LayoutXLM_xfun_zh/ Global.infer_img=ppstructure/docs/vqa/input/zh_val_21.jpg -c_ser configs/vqa/ser/layoutxlm.yml -o_ser Architecture.Backbone.checkpoints=models/ser_LayoutXLM_xfun_zh/