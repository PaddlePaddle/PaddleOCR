# recommended paddle.__version__ == 3.0.0
python3 -m paddle.distributed.launch --log_dir=./log/ --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml
