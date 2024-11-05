#!/bin/bash
# train from bare empty model, with no weight.

python -m paddle.distributed.launch --gpus '0,1' \
                       tools/train.py -c ./atrain_script/configs/ch_PP-OCRv3_rec_distillation.yml \
                       -o Global.epoch_num=20 \
                       Global.eval_batch_step='[1000, 2000]' \
                       Train.dataset.data_dir=./train_data \
                       Train.dataset.label_file_list=['./train_data/rec/train_list.txt'] \
                       Train.loader.batch_size_per_card=64 \
                       Eval.dataset.data_dir=./train_data \
                       Eval.dataset.label_file_list=["./train_data/rec/valid.list"] \
                       Eval.loader.batch_size_per_card=64