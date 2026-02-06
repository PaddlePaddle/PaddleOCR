# single GPU
python3.7 train.py  -c mv3_large_x0_5.yml

# distribute training
python3.7 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1'  train.py  -c mv3_large_x0_5.yml
