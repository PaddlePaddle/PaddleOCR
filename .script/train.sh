# recommended paddle.__version__ == 2.0.0
# os.environ['CPATH'] = os.getenv('CPATH', '') + f':{os.environ["HOME"]}/local/cudnn/include'
export CPATH=$CPATH:${HOME}/local/cudnn/include
# os.environ['LD_LIBRARY_PATH'] = os.getenv('LD_LIBRARY_PATH', '') + f':{os.environ["HOME"]}/local/cudnn/lib'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/local/cudnn/lib
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1'  tools/train.py -c .script/configs/en_PP-OCRv3_rec.yml
