export NVIDIA_TF32_OVERRIDE=0
python -m paddle.distributed.launch \
--log_dir=./output/AMP/table_fp16_o2_tmp/ \
--gpus '0,1,2,3' tools/train.py \
-c configs/table/table_mv3_zhoujun_small_o2.yml 
#-o Global.checkpoints=./output/rerun_table_mv3_fp16_o2/best_accuracy
