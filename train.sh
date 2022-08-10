# recommended paddle.__version__ == 2.0.0
python3.7 -m paddle.distributed.launch --log_dir=./output/SLANet_en --gpus '4,5,6,7'  tools/train.py -c configs/table/SLANet_ch.yml
