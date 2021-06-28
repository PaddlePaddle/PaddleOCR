# Distributed training

## Introduction

The high performance of distributed training is one of the core advantages of PaddlePaddle. In the classification task, distributed training can achieve almost linear speedup ratio. Generally, OCR training task need massive training data. Such as recognition, ppocrv2.0 model is trained based on 1800W dataset, which is very time-consuming if using single machine. Therefore, the distributed training is used in paddleocr to speedup the training task. For more information about distributed training, please refer to [distributed training quick start tutorial](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/ps_quick_start.html).

## Quick Start

### Training with single machine

Take recognition as an example. After the data is prepared locally, start the training task with the interface of `paddle.distributed.launch`. The start command as follows:

```shell
python3 -m paddle.distributed.launch \
    --log_dir=./log/ \
    --gpus '0,1,2,3,4,5,6,7' \
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
* The code, data and start command betweent different machines must be completely consistent and then all machines need to run start command. The first machine in the `ip_list` is set to `trainer0`, and so on.


## Performance comparison

* Based on 26W public recognition dataset (LSVT, rctw, mtwi), training on single 8-card P40 and dual 8-card P40, the final time consumption is as follows.

|   Model   |   Config file  |  Number of machines |   Number of GPUs per machine   |   Training time      | Recognition acc  | Speedup ratio |
| :-------: | :------------: |  :----------------: | :----------------------------: | :------------------: | :--------------: | :-----------: |
|   CRNN    |   configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml   |   1          |  8  |  60h  |  66.7% | - |
|   CRNN    |   configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml   |   2          |  8  |  40h  |  67.0% | 150% |

It can be seen that the training time is shortened from 60h to 40h, the speedup ratio can reach 150% (60h / 40h), and the efficiency is 75% (60h / (40h * 2)).
