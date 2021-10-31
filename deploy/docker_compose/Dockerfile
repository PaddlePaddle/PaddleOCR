FROM python:3.7
ARG work_dir=/app
ENV FLASK_APP_PATH=${work_dir}/server.py
ENV FLASK_PORT=5000
SHELL ["/bin/bash", "--login", "-c"]
WORKDIR ${work_dir}
COPY ./ ${work_dir}
COPY ./entrypoint.sh /
RUN chmod +x /entrypoint.sh
RUN pip install -r requirements.txt -U
## Recommendation for those in China
# RUN pip install -r requirements.txt -U -i https://mirror.baidu.com/pypi/simple
ENTRYPOINT [ "/entrypoint.sh" ]

