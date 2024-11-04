#!/bin/bash


# eval data/ folder

pdm run python tools/eval.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml \
                         -o Global.checkpoints=ckpt/ch_PP-OCRv3_rec_train/best_accuracy \
                         Eval.dataset.data_dir=./data \
                         Eval.dataset.label_file_list=["./data/val.list"]