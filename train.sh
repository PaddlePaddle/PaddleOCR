# recommended paddle.__version__ == 2.0.0
export CUDA_VISIBLE_DEVICES=0,2,6,7
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,2,6,7'  tools/train.py -c configs/rec/rec_vit_parseq.yml #rec_r31_sar
# python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '2,6'  tools/train.py -c configs/rec/rec_r31_sar.yml
