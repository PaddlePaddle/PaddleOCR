# FROM paddlecloud/paddleocr:2.6-gpu-cuda10.2-cudnn7-latest

# build from paddle
FROM paddlepaddle/paddle:2.6.2-gpu-cuda12.0-cudnn8.9-trt8.6

COPY requirements.txt .
RUN pip install jupyterlab && pip install -r requirements.txt

RUN jupyter lab --generate-config

RUN mkdir -p /root/.jupyter
RUN echo "c.NotebookApp.password = u'1234567890'" >> /root/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.token = u'33fab9690fd8fb67f5ce375c722f3c4204b06d31c654b289'" >> /root/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.port = 8888" >> /root/.jupyter/jupyter_notebook_config.py

EXPOSE 8888

WORKDIR /opt/PaddleOCR
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
