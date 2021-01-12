# for paddle.__version__ >= 2.0rc1
python3 -m paddle.distributed.launch --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml

# for paddle.__version__ < 2.0rc1
# python3 -m paddle.distributed.launch --selected_gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml
