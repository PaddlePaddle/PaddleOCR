
# 介绍

test.sh和params.txt文件配合使用，完成OCR轻量检测和识别模型从训练到预测的流程测试。

# 安装依赖
- 安装PaddlePaddle >= 2.0
- 安装PaddleOCR依赖
    ```
    pip3 install  -r ../requirements.txt
    ```
- 安装autolog
    ```
    git clone https://github.com/LDOUBLEV/AutoLog
    cd AutoLog
    pip3 install -r requirements.txt
    python3 setup.py bdist_wheel
    pip3 install ./dist/auto_log-1.0.0-py3-none-any.whl
    cd ../
    ```

# 目录介绍

```bash
tests/
├── ocr_det_params.txt   # 测试OCR检测模型的参数配置文件
├── ocr_rec_params.txt   # 测试OCR识别模型的参数配置文件
└── prepare.sh           # 完成test.sh运行所需要的数据和模型下载
└── test.sh              # 根据
```

# 使用方法
test.sh包含四种运行模式，每种模式的运行数据不同，分别用于测试速度和精度，分别是：
- 模式1 lite_train_infer，使用少量数据训练，用于快速验证训练到预测的走通流程，不验证精度和速度；
```
bash test/prepare.sh ./tests/ocr_det_params.txt 'lite_train_infer'
bash tests/test.sh ./tests/ocr_det_params.txt 'lite_train_infer'
```
- 模式2 whole_infer，使用少量数据训练，一定量数据预测，用于验证训练后的模型执行预测，预测速度是否合理；
```
bash tests/prepare.sh ./tests/ocr_det_params.txt 'whole_infer'
bash tests/test.sh ./tests/ocr_det_params.txt 'whole_infer'
```

- 模式3 infer 不训练，全量数据预测，走通开源模型评估、动转静，检查inference model预测时间和精度;
```
bash tests/prepare.sh ./tests/ocr_det_params.txt 'infer'
用法1:
bash tests/test.sh ./tests/ocr_det_params.txt 'infer'
用法2: 指定GPU卡预测，第三个传入参数为GPU卡号
bash tests/test.sh ./tests/ocr_det_params.txt 'infer' '1'
```

模式4: whole_train_infer , CE： 全量数据训练，全量数据预测，验证模型训练精度，预测精度，预测速度
```
bash tests/prepare.sh ./tests/ocr_det_params.txt 'whole_train_infer'
bash tests/test.sh ./tests/ocr_det_params.txt 'whole_train_infer'
```
