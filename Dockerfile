# FROM paddlepaddle/paddle:2.6.2-gpu-cuda12.0-cudnn8.9-trt8.6 # not working
FROM svtter/paddle:cuda11.8

RUN pip install jupyterlab

# Configure JupyterLab to listen on all network interfaces
RUN jupyter server --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_server_config.py


COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# build: docker build -t svtter/paddleocr:cuda11.8 -f Dockerfile .
