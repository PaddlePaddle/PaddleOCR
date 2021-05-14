#!/bin/bash

enable_mkldnn=False
use_gpu=True
cpu_threads=1
use_trt=False
precision='fp32'

det_model_dir="./inference/ch_ppocr_mobile_v2.0_det_infer"
img_dir="./doc/imgs/"
rec_model_dir="./inference/ch_ppocr_mobile_v2.0_rec_infer"
rec_img_dir="./doc/imgs/"

benchmark=True

# det
python tools/infer/predict_det.py --enable_mkldnn=${enable_mkldnn} \
                                  --use_gpu=${use_gpu} \
                                  --cpu_threads=${cpu_threads} \
                                  --use_tensorrt=${use_trt} \
                                  --precision=${precision} \
                                  --det_model_dir=${det_model_dir} \
                                  --image_dir=${img_dir} \
                                  --benchmark=${benchmark}

# rec
python tools/infer/predict_rec.py --enable_mkldnn=${enable_mkldnn} \
                                  --use_gpu=${use_gpu} \
                                  --cpu_threads=${cpu_threads} \
                                  --use_tensorrt=${use_trt} \
                                  --precision=${precision} \
                                  --rec_batch_num=1 \
                                  --rec_model_dir=${rec_model_dir} \
                                  --image_dir=${rec_img_dir} \
                                  --benchmark=True
