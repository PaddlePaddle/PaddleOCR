[English](README.md) | 简体中文

# PaddleOCR 模型在SOPHGO上部署方案-FastDeploy

## 1. 说明
PaddleOCR支持通过FastDeploy在SOPHGO上部署相关模型.

## 2.支持模型列表

下表中的模型下载链接由PaddleOCR模型库提供, 详见[PP-OCR系列模型列表](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md)

| PaddleOCR版本 | 文本框检测 | 方向分类模型 | 文字识别 |字典文件| 说明 |
|:----|:----|:----|:----|:----|:--------|
| ch_PP-OCRv3[推荐] |[ch_PP-OCRv3_det](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_PP-OCRv3_rec](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) | [ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv3系列原始超轻量模型，支持中英文、多语种文本检测 |
| en_PP-OCRv3[推荐] |[en_PP-OCRv3_det](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [en_PP-OCRv3_rec](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) | [en_dict.txt](https://bj.bcebos.com/paddlehub/fastdeploy/en_dict.txt) | OCRv3系列原始超轻量模型，支持英文与数字识别，除检测模型和识别模型的训练数据与中文模型不同以外，无其他区别 |
| ch_PP-OCRv2 |[ch_PP-OCRv2_det](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_PP-OCRv2_rec](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar) | [ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv2系列原始超轻量模型，支持中英文、多语种文本检测 |
| ch_PP-OCRv2_mobile |[ch_ppocr_mobile_v2.0_det](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_ppocr_mobile_v2.0_rec](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) | [ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv2系列原始超轻量模型，支持中英文、多语种文本检测,比PPOCRv2更加轻量 |
| ch_PP-OCRv2_server |[ch_ppocr_server_v2.0_det](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_ppocr_server_v2.0_rec](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) |[ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv2服务器系列模型, 支持中英文、多语种文本检测，比超轻量模型更大，但效果更好|

## 3. 准备PP-OCR推理模型以及转换模型

PP-OCRv3包括文本检测模型（ch_PP-OCRv3_det）、方向分类模型（ch_ppocr_mobile_v2.0_cls）、文字识别模型（ch_PP-OCRv3_rec）
SOPHGO-TPU部署模型前需要将以上Paddle模型转换成bmodel模型，我们以ch_PP-OCRv3_det模型为例，具体步骤如下:
- 下载Paddle模型[ch_PP-OCRv3_det](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar)
- Pddle模型转换为ONNX模型，请参考[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)
- ONNX模型转换bmodel模型的过程，请参考[TPU-MLIR](https://github.com/sophgo/tpu-mlir)
下面我们提供一个example, 供用户参考，完成模型的转换.

### 3.1 下载ch_PP-OCRv3_det模型,并转换为ONNX模型
```shell
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar xvf ch_PP-OCRv3_det_infer.tar

# 修改ch_PP-OCRv3_det模型的输入shape，由动态输入变成固定输入
python paddle_infer_shape.py --model_dir ch_PP-OCRv3_det_infer \
                             --model_filename inference.pdmodel \
                             --params_filename inference.pdiparams \
                             --save_dir ch_PP-OCRv3_det_infer_fix \
                             --input_shape_dict="{'x':[1,3,960,608]}"

# 请用户自行安装最新发布版本的paddle2onnx, 转换模型到ONNX格式的模型
paddle2onnx --model_dir ch_PP-OCRv3_det_infer_fix \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ch_PP-OCRv3_det_infer_fix.onnx \
            --enable_dev_version True
```

### 3.2 导出bmodel模型

以转换BM1684x的bmodel模型为例子，我们需要下载[TPU-MLIR](https://github.com/sophgo/tpu-mlir)工程，安装过程具体参见[TPU-MLIR文档](https://github.com/sophgo/tpu-mlir/blob/master/README.md)。
#### 3.2.1    安装
``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234是一个示例，也可以设置其他名字
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest

source ./envsetup.sh
./build.sh
```

#### 3.2.2    ONNX模型转换为bmodel模型
``` shell
mkdir ch_PP-OCRv3_det && cd ch_PP-OCRv3_det

#在该文件中放入测试图片，同时将上一步转换的ch_PP-OCRv3_det_infer_fix.onnx放入该文件夹中
cp -rf ${REGRESSION_PATH}/dataset/COCO2017 .
cp -rf ${REGRESSION_PATH}/image .
#放入onnx模型文件ch_PP-OCRv3_det_infer_fix.onnx

mkdir workspace && cd workspace

#将ONNX模型转换为mlir模型，其中参数--output_names可以通过NETRON查看
model_transform.py \
    --model_name ch_PP-OCRv3_det \
    --model_def ../ch_PP-OCRv3_det_infer_fix.onnx \
    --input_shapes [[1,3,960,608]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --output_names sigmoid_0.tmp_0 \
    --test_input ../image/dog.jpg \
    --test_result ch_PP-OCRv3_det_top_outputs.npz \
    --mlir ch_PP-OCRv3_det.mlir

#将mlir模型转换为BM1684x的F32 bmodel模型
model_deploy.py \
  --mlir ch_PP-OCRv3_det.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input ch_PP-OCRv3_det_in_f32.npz \
  --test_reference ch_PP-OCRv3_det_top_outputs.npz \
  --model ch_PP-OCRv3_det_1684x_f32.bmodel
```
最终获得可以在BM1684x上能够运行的bmodel模型ch_PP-OCRv3_det_1684x_f32.bmodel。按照上面同样的方法，可以将ch_ppocr_mobile_v2.0_cls，ch_PP-OCRv3_rec转换为bmodel的格式。如果需要进一步对模型进行加速，可以将ONNX模型转换为INT8 bmodel，具体步骤参见[TPU-MLIR文档](https://github.com/sophgo/tpu-mlir/blob/master/README.md)。


## 4. 详细部署的部署示例
- [Python部署](python)
- [C++部署](cpp)
