# 分布式训练

## 简介

* 分布式训练的高性能，是飞桨的核心优势技术之一，在分类任务上，分布式训练可以达到几乎线性的加速比。OCR训练任务中往往包含大量训练数据，以识别为例，ppocrv2.0模型在训练时使用了1800W数据，如果使用单机训练，会非常耗时。因此，PaddleOCR中使用分布式训练接口完成训练任务，同时支持单机训练与多机训练。更多关于分布式训练的方法与文档可以参考：[分布式训练快速开始教程](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/ps_quick_start.html)。

## 使用方法

### 单机训练

* 以识别为例，本地准备好数据之后，使用`paddle.distributed.launch`的接口启动训练任务即可。下面为运行代码示例。

```shell
python3 -m paddle.distributed.launch \
    --log_dir=./log/ \
    --gpus "0,1,2,3,4,5,6,7" \
    tools/train.py \
    -c configs/rec/rec_mv3_none_bilstm_ctc.yml
```

### 多机训练

* 相比单机训练，多机训练时，只需要添加`--ips`的参数，该参数表示需要参与分布式训练的机器的ip列表，不同机器的ip用逗号隔开。下面为运行代码示例。


```shell
ip_list="192.168.0.1,192.168.0.2"
python3 -m paddle.distributed.launch \
    --log_dir=./log/ \
    --ips="${ip_list}" \
    --gpus="0,1,2,3,4,5,6,7" \
    tools/train.py \
    -c configs/rec/rec_mv3_none_bilstm_ctc.yml
```

**注：**
* 不同机器的ip信息需要用逗号隔开，可以通过`ifconfig`或者`ipconfig`查看。
* 不同机器之间需要做免密设置，且可以直接ping通，否则无法完成通信。
* 不同机器之间的代码、数据与运行命令或脚本需要保持一致，且所有的机器上都需要运行设置好的训练命令或者脚本。最终`ip_list`中的第一台机器的第一块设备是trainer0，以此类推。


## 性能效果测试

* 基于单机8卡P40，和2机8卡P40，在26W公开识别数据集(LSVT, RCTW, MTWI)上进行训练，最终耗时如下。

|         模型             |     配置文件 |  机器数量    | 每台机器的GPU数量  |   训练时间    | 识别Acc    | 加速比 |
| :----------------------: | :------------: | :------------: | :---------------: | :----------: | :-----------: | :-----------: |
|          CRNN        |   configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml  | 1     |  8  |  60h  |  66.7% | - |
|          CRNN        |   configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml   | 2   |  8  |  40h  |  67.0% | 150% |

可以看出，精度没有下降的情况下，训练时间由60h缩短为了40h，加速比可以达到60h/40h=150%，效率为60h/(40h*2)=75%。
