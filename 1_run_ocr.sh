export NVIDIA_TF32_OVERRIDE=0
python -m paddle.distributed.launch --gpus=0,1,2,3 tools/train.py -c config.yml
