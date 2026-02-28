#!/usr/bin/env bash

wget https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.4/paddlex_hps_PaddleOCR-VL_sdk.tar.gz
tar -xf paddlex_hps_PaddleOCR-VL_sdk.tar.gz
cp ../pipeline_config_vllm.yaml paddlex_hps_PaddleOCR-VL_sdk/server/pipeline_config.yaml
