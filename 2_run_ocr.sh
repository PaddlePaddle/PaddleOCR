export NVIDIA_TF32_OVERRIDE=0
python -m paddle.distributed.launch --gpus=4,5,6,7 tools/train.py -c config_o2_cast_conv.yml
