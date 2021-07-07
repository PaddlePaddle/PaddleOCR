# Version: 1.0.0
FROM hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.0-cudnn7-dev

# PaddleOCR base on Python3.7
RUN pip3.7 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN python3.7 -m pip install paddlepaddle-gpu==1.7.2.post107 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3.7 install paddlehub --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN git clone https://gitee.com/paddlepaddle/PaddleOCR.git /PaddleOCR

WORKDIR /PaddleOCR

RUN pip3.7 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN mkdir -p /PaddleOCR/inference/
# Download orc detect model(light version). if you want to change normal version, you can change ch_ppocr_mobile_v1.1_det_infer to ch_ppocr_server_v1.1_det_infer, also remember change det_model_dir in deploy/hubserving/ocr_system/params.py）
ADD https://paddleocr.bj.bcebos.com/20-09-22/mobile/det/ch_ppocr_mobile_v1.1_det_infer.tar /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/ch_ppocr_mobile_v1.1_det_infer.tar -C /PaddleOCR/inference/

# Download direction classifier(light version). If you want to change normal version, you can change ch_ppocr_mobile_v1.1_cls_infer to ch_ppocr_mobile_v1.1_cls_infer, also remember change cls_model_dir in deploy/hubserving/ocr_system/params.py）
ADD https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_infer.tar /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/ch_ppocr_mobile_v1.1_cls_infer.tar -C /PaddleOCR/inference/

# Download orc recognition model(light version). If you want to change normal version, you can change ch_ppocr_mobile_v1.1_rec_infer to ch_ppocr_server_v1.1_rec_infer, also remember change rec_model_dir in deploy/hubserving/ocr_system/params.py）
ADD https://paddleocr.bj.bcebos.com/20-09-22/mobile/rec/ch_ppocr_mobile_v1.1_rec_infer.tar /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/ch_ppocr_mobile_v1.1_rec_infer.tar -C /PaddleOCR/inference/

EXPOSE 8866

CMD ["/bin/bash","-c","hub install deploy/hubserving/ocr_system/ && hub serving start -m ocr_system"]