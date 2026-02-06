# export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m paddle.distributed.launch tools/train.py --config_file "config/icdar2015_resnet50_FPN_DBhead_polyLR.yaml"
