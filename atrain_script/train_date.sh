#!/bin/bash

python tools/train.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml \
                       -o Global.pretrained_model=./ckpt/ch_PP-OCRv3_rec_train/best_accuracy \
                       Global.epoch_num=20 \
                       Global.eval_batch_step='[0, 20]' \
                       Train.dataset.data_dir=./data \
                       Train.dataset.label_file_list=['./data/render_train.list'] \
                       Train.loader.batch_size_per_card=64 \
                       Eval.dataset.data_dir=./data \
                       Eval.dataset.label_file_list=["./data/val.list"] \
                       Eval.loader.batch_size_per_card=64