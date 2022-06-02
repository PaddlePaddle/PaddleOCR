# Distributed training

## Introduction

The high performance of distributed training is one of the core advantages of PaddlePaddle. In the classification task, distributed training can achieve almost linear speedup ratio. Generally, OCR training task need massive training data. Such as recognition, PP-OCR v2.0 model is trained based on 1800W dataset, which is very time-consuming if using single machine. Therefore, the distributed training is used in PaddleOCR to speedup the training task. For more information about distributed training, please refer to [distributed training quick start tutorial](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/ps_quick_start.html).

## Quick Start

### Training with single machine

Take recognition as an example. After the data is prepared locally, start the training task with the interface of `paddle.distributed.launch`. The start command as follows:

```shell
python3 -m paddle.distributed.launch \
    --log_dir=./log/ \
    --gpus "0,1,2,3,4,5,6,7" \
    tools/train.py \
    -c configs/rec/rec_mv3_none_bilstm_ctc.yml
```

### Training with multi machine

Compared with single machine, training with multi machine only needs to add the parameter `--ips` to start command, which represents the IP list of machines used for distributed training, and the IP of different machines are separated by commas. The start command as follows:

```shell
ip_list="192.168.0.1,192.168.0.2"
python3 -m paddle.distributed.launch \
    --log_dir=./log/ \
    --ips="${ip_list}" \
    --gpus="0,1,2,3,4,5,6,7" \
    tools/train.py \
    -c configs/rec/rec_mv3_none_bilstm_ctc.yml
```

**Notice:**
* The IP addresses of different machines need to be separated by commas, which can be queried through `ifconfig` or `ipconfig`.
* Different machines need to be set to be secret free and can `ping` success with others directly, otherwise communication cannot establish between them.
* The code, data and start command between different machines must be completely consistent and then all machines need to run start command. The first machine in the `ip_list` is set to `trainer0`, and so on.


## Performance comparison

* On two 8-card P40 graphics cards, the final time consumption and speedup ratio for public recognition dataset (LSVT, RCTW, MTWI) containing 260k images are as follows.


| Model   | Config file  | Recognition acc     | single 8-card training time | two 8-card training time | Speedup ratio |
|------|-----|--------|--------|--------|-----|
| CRNN | [rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml) | 67.0% | 2.50d   | 1.67d  | **1.5** |


* On four 8-card V100 graphics cards, the final time consumption and speedup ratio for full data are as follows.


| Model   | Config file  | Recognition acc     | single 8-card training time | four 8-card training time | Speedup ratio |
|------|-----|--------|--------|--------|-----|
| SVTR | [ch_PP-OCRv3_rec_distillation.yml](../../configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml) | 74.0% | 10d   | 2.84d  | **3.5** |
