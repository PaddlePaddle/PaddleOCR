export NVIDIA_TF32_OVERRIDE=0
python -m paddle.distributed.launch \
--log_dir=./output/AMP/table_mv3_fp16_o1_with_matmul/ \
--gpus '4,5,6,7' tools/train.py \
-c configs/table/table_mv3_zhoujun_small.yml \
-o Global.checkpoints=./output/table_ampO1_matmul_all/best_accuracy
