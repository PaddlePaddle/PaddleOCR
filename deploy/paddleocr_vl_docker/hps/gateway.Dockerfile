FROM python:3.10-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgl1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY gateway .
RUN --mount=type=bind,source=paddlex_hps_PaddleOCR-VL_sdk/client,target=/tmp/sdk \
    python -m pip install -r requirements.txt \
    && python -m pip install -r /tmp/sdk/requirements.txt \
    && python -m pip install /tmp/sdk/paddlex_hps_client-*.whl
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "app:app"]
