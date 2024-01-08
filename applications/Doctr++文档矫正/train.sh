export OPENCV_IO_ENABLE_OPENEXR=1
export FLAGS_logtostderr=0
export CUDA_VISIBLE_DEVICES=4

python train.py --data-root /ssd1/chenjiajun05/chenjiajun05/doc3d \
    --img-size 288 \
    --name "DocTr++" \
    --batch-size 12 \
    --lr 1e-4 \
    --exist-ok \
    --epochs 150 \
