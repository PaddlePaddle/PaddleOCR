# Version: 2.0.0
FROM registry.baidubce.com/paddlepaddle/paddle:2.1.0

RUN /bin/cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo 'Asia/Shanghai' >/etc/timezone
# PaddleOCR base on Python3.7
RUN pip3.7 install --upgrade pip -i https://mirror.baidu.com/pypi/simple

RUN pip3.7 install paddlehub --upgrade -i https://mirror.baidu.com/pypi/simple

RUN git clone https://github.com/llooper-dev/PaddleOCR.git /PaddleOCR

WORKDIR /PaddleOCR

RUN pip3.7 install -r requirements.txt -i https://mirror.baidu.com/pypi/simple

RUN mkdir -p /PaddleOCR/inference/
# Download orc detect model(light version). if you want to change normal version, you can change ch_ppocr_mobile_v2.0_det_infer to ch_ppocr_server_v2.0_det_infer, also remember change det_model_dir in deploy/hubserving/ocr_system/params.py）
ADD http://124.232.163.91:9000/openaifacen/stor/20210913/det_ch_db_res18_v20210902.tar /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/det_ch_db_res18_v20210902.tar -C /PaddleOCR/inference/

# Download direction classifier(light version). If you want to change normal version, you can change ch_ppocr_mobile_v2.0_cls_infer to ch_ppocr_mobile_v2.0_cls_infer, also remember change cls_model_dir in deploy/hubserving/ocr_system/params.py）
ADD https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/ch_ppocr_mobile_v2.0_cls_infer.tar -C /PaddleOCR/inference/

# Download orc recognition model(light version). If you want to change normal version, you can change ch_ppocr_mobile_v2.0_rec_infer to ch_ppocr_server_v2.0_rec_infer, also remember change rec_model_dir in deploy/hubserving/ocr_system/params.py）
ADD http://124.232.163.91:9000/openaifacen/stor/20210913/rec_chinese_common_train_v2.0_v20210821.tar /PaddleOCR/inference/
RUN tar xf /PaddleOCR/inference/rec_chinese_common_train_v2.0_v20210821.tar -C /PaddleOCR/inference/

RUN ls -lshR /PaddleOCR/inference/

EXPOSE 8868

CMD ["/bin/bash","-c","hub install deploy/hubserving/ocr_system/ && hub serving start -m ocr_system -c deploy/hubserving/ocr_system/config.json"]