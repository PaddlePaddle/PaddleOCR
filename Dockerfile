# FROM paddlecloud/paddleocr:2.6-gpu-cuda10.2-cudnn7-latest

# build from paddle
# FROM paddlepaddle/paddle:2.6.2-gpu-cuda12.0-cudnn8.9-trt8.6
FROM svtter/paddle:cuda11.2-paddle2.4.2

COPY requirements.txt .
RUN pip install jupyterlab && pip install -r requirements.txt

RUN jupyter lab --generate-config

RUN mkdir -p /root/.jupyter
EXPOSE 8888

WORKDIR /opt/PaddleOCR
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token='abcd'"]
