#!/bin/bash
# python tools/train.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml \
                     #   -o Global.pretrained_model=./ckpt/ch_PP-OCRv3_rec_train/best_accuracy \
                     #   Global.epoch_num=20 \
                     #   Global.eval_batch_step='[0, -1]' \
                     #   Train.dataset.data_dir=./train_data \
                     #   Train.dataset.label_file_list=['./train_data/rec/train_list.txt'] \
                     #   Train.loader.batch_size_per_card=64 \
                     #   Eval.dataset.data_dir=./train_data \
                     #   Eval.dataset.label_file_list=["./train_data/rec/val.list"] \
                     #   Eval.loader.batch_size_per_card=64

# validation at first and end of epoch.
python -m paddle.distributed.launch --gpus '0,1' \
                       tools/train.py -c ./atrain_script/configs/ch_PP-OCRv3_rec_distillation.yml \
                       -o Global.pretrained_model=./ckpt/ch_PP-OCRv3_rec_train/best_accuracy \
                       Global.epoch_num=20 \
                       Global.eval_batch_step='[1000, 2000]' \
                       Train.dataset.data_dir=./train_data \
                       Train.dataset.label_file_list=['./train_data/rec/train_list.txt'] \
                       Train.loader.batch_size_per_card=64 \
                       Eval.dataset.data_dir=./train_data \
                       Eval.dataset.label_file_list=["./train_data/rec/val.list"] \
                       Eval.loader.batch_size_per_card=64