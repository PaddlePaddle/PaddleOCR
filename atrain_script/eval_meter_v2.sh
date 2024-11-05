#!/bin/bash


# eval data/ folder

python tools/eval.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml \
                         -o Global.checkpoints=./output/rec_ppocr_v3_distillation_meter/best_model/model \
                         Eval.dataset.data_dir=./train_data\
                         Eval.dataset.label_file_list=["./train_data/rec/val.list"]

