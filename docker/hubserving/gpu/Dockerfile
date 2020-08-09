# Version: 1.0.0
FROM hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.0-cudnn7-dev

# PaddleOCR需要在Python3.7下执行
RUN pip3.7 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN python3.7 -m pip install paddlepaddle-gpu==1.7.2.post107 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3.7 install paddlehub --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN git clone https://gitee.com/PaddlePaddle/PaddleOCR

WORKDIR /home/PaddleOCR

RUN pip3.7 install -r requirments.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN mkdir -p /home/PaddleOCR/inference
# 下载超轻量级中文OCR模型的检测模型并解压（可以换成通用中文OCR模型，ch_det_r50_vd_db_infer，记得要修改deploy/hubserving/ocr_system/params.py中的det_model_dir）
ADD https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db_infer.tar /PaddleOCR/inference
RUN tar xf /PaddleOCR/inference/ch_det_mv3_db_infer.tar -C /PaddleOCR/inference

# 下载超轻量级中文OCR模型的识别模型并解压（可以换成通用中文OCR模型，ch_rec_r34_vd_crnn_enhance_infer，记得要修改deploy/hubserving/ocr_system/params.py中的rec_model_dir）
ADD https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_infer.tar /PaddleOCR/inference
RUN tar xf /PaddleOCR/inference/ch_rec_mv3_crnn_infer.tar -C /PaddleOCR/inference

EXPOSE 8866

CMD ["/bin/bash","-c","export PYTHONPATH=. && hub install deploy/hubserving/ocr_system/ && hub serving start -m ocr_system"]