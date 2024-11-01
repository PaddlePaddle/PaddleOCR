# recommended paddle.__version__ == 2.0.0

set -e

# os.environ['CPATH'] = os.getenv('CPATH', '') + f':{os.environ["HOME"]}/local/cudnn/include'
export CPATH=$CPATH:${HOME}/local/cudnn/include
# os.environ['LD_LIBRARY_PATH'] = os.getenv('LD_LIBRARY_PATH', '') + f':{os.environ["HOME"]}/local/cudnn/lib'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/local/cudnn/lib

# Update CUDA device selection (keeping single GPU setup)
# export CUDA_VISIBLE_DEVICES=7

# 修改 CUDA 架构设置
# export CUDA_ARCH_FLAGS="compute_60,sm_60"  # Pascal 架构对应的配置
# export PADDLE_CUDA_ARCH_NAME="6.0"         # 明确指定 Pascal 架构版本

# Add environment variable to force Pascal architecture compatibility
export CUDA_ARCH_NAME=Pascal

run_check() {
    python -c "import paddle; paddle.utils.run_check()"
}

run_check || exit 1

# Add environment variable to force Pascal architecture compatibility
# export CUDA_ARCH_NAME=Pascal

# multiple gpu, seems not work.
# python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '5,6,7'  tools/train.py -c atrain_script/configs/en_PP-OCRv3_rec.yml


# 在 train.sh 中添加
python3 -c "import sys; sys.setrecursionlimit(3000)" # 增加递归深度限制

python3 tools/train.py \
    -c atrain_script/configs/en_PP-OCRv3_rec.yml \
    -o Global.pretrained_model=./pretrain_models/en_PP-OCRv3_rec_train/best_accuracy