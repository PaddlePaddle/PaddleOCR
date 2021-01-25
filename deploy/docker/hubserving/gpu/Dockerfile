# Version: 2.0.0
FROM egistry.baidubce.com/paddlepaddle/paddle:2.0.0rc1-gpu-cuda10.0-cudnn7

# PaddleOCR base on Python3.7
RUN pip3.7 install --upgrade pip -i https://mirror.baidu.com/pypi/simple

RUN pip3.7 install paddlehub --upgrade -i https://mirror.baidu.com/pypi/simple

RUN git clone https://github.com/PaddlePaddle/PaddleOCR.git /PaddleOCR

WORKDIR /PaddleOCR

RUN pip3.7 install -r requirements.txt -i https://mirror.baidu.com/pypi/simple

RUN mkdir -p /PaddleOCR/inference/
# Download orc detect model(light version). if you want to change normal version, you can change ch_ppocr_mobile_v2.0_det_infer to ch_ppocr_server_v2.0_det_infer, also remember change det_model_dir in deploy/hubserving/ocr_system/params.py）
ADD {link} /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/{file}.tar -C /PaddleOCR/inference/

# Download direction classifier(light version). If you want to change normal version, you can change ch_ppocr_mobile_v2.0_cls_infer to ch_ppocr_mobile_v2.0_cls_infer, also remember change cls_model_dir in deploy/hubserving/ocr_system/params.py）
ADD {link} /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/{file} -C /PaddleOCR/inference/

# Download orc recognition model(light version). If you want to change normal version, you can change ch_ppocr_mobile_v2.0_rec_infer to ch_ppocr_server_v2.0_rec_infer, also remember change rec_model_dir in deploy/hubserving/ocr_system/params.py）
ADD {link} /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/{file}.tar -C /PaddleOCR/inference/

EXPOSE 8868

CMD ["/bin/bash","-c","hub install deploy/hubserving/ocr_system/ && hub serving start -m ocr_system"]