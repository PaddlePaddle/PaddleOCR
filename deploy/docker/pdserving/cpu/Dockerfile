FROM debian:10
# 更换镜像源并安装依赖
RUN echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ buster main contrib non-free" > /etc/apt/sources.list  \
&&  echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ buster-updates main contrib non-free" >> /etc/apt/sources.list  \
&&  echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ buster-backports main contrib non-free" >> /etc/apt/sources.list  \
&&  echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian-security buster/updates main contrib non-free" >> /etc/apt/sources.list \
&&  apt update && apt -y install python3-pip \
&&  apt clean

RUN apt -y install ca-certificates git patch cmake libgeos-dev libsm6 libxrender1 libxext6 libxrender-dev wget \
&&  apt clean

# 安装python依赖
RUN pip3 config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple \
&&  pip3 install --no-cache-dir sentencepiece==0.1.82  \
&&  pip3 install --no-cache-dir https://paddle-serving.bj.bcebos.com/whl/paddle_serving_server-0.0.0-py3-none-any.whl

RUN pip3 install --no-cache-dir common dual tight data prox imgaug \
&&  pip3 install --no-cache-dir paddle \
&&  pip3 install --no-cache-dir paddlepaddle \
&&  pip3 uninstall -y protobuf \
&&  pip3 install --no-binary protobuf protobuf
&&  pip3 install --no-cache-dir https://paddle-serving.bj.bcebos.com/whl/paddle_serving_client-0.0.0-cp37-none-any.whl

#克隆项目
RUN git clone https://gitee.com/paddlepaddle/PaddleOCR.git /PaddleOCR
&&  mkdir -p /PaddleOCR/inference/ \
&&  cd /PaddleOCR/inference/ \
&&  wget https://paddleocr.bj.bcebos.com/20-09-22/mobile/det/ch_ppocr_mobile_v1.1_det_infer.tar \
&&  wget https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_infer.tar \
&&  wget https://paddleocr.bj.bcebos.com/20-09-22/mobile/rec/ch_ppocr_mobile_v1.1_rec_infer.tar \
&&  tar xf /PaddleOCR/inference/ch_ppocr_mobile_v1.1_det_infer.tar -C /PaddleOCR/inference/  \
&&  tar xf /PaddleOCR/inference/ch_ppocr_mobile_v1.1_cls_infer.tar -C /PaddleOCR/inference/  \
&&  tar xf /PaddleOCR/inference/ch_ppocr_mobile_v1.1_rec_infer.tar -C /PaddleOCR/inference/  \
&&  rm -f /PaddleOCR/inference/ch_ppocr_mobile_v1.1_cls_infer.tar  \
&&  rm -f /PaddleOCR/inference/ch_ppocr_mobile_v1.1_det_infer.tar  \
&&  rm -f /PaddleOCR/inference/ch_ppocr_mobile_v1.1_rec_infer.tar  \
&&  python3 /PaddleOCR/deploy/pdserving/inference_to_serving.py --model_dir /PaddleOCR/inference/ch_ppocr_mobile_v1.1_cls_infer \
&&  python3 /PaddleOCR/deploy/pdserving/inference_to_serving.py --model_dir /PaddleOCR/inference/ch_ppocr_mobile_v1.1_det_infer \
&&  python3 /PaddleOCR/deploy/pdserving/inference_to_serving.py --model_dir /PaddleOCR/inference/ch_ppocr_mobile_v1.1_rec_infer \
&&  rm /PaddleOCR/inference/ch_ppocr_mobile_v1.1_cls_infer/model && rm /PaddleOCR/inference/ch_ppocr_mobile_v1.1_cls_infer/params \
&&  rm /PaddleOCR/inference/ch_ppocr_mobile_v1.1_det_infer/model && rm /PaddleOCR/inference/ch_ppocr_mobile_v1.1_det_infer/params \
&&  rm /PaddleOCR/inference/ch_ppocr_mobile_v1.1_rec_infer/model && rm /PaddleOCR/inference/ch_ppocr_mobile_v1.1_rec_infer/params

EXPOSE 9292

WORKDIR /PaddleOCR

CMD ["/bin/bash","-c","cd /PaddleOCR && export PYTHONPATH=$PWD:$PYTHONPATH && cd /PaddleOCR/deploy/pdserving/ && python3 ocr_local_server.py"]
