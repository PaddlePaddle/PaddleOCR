# PPOCRv3 SOPHGO C++部署示例

## 支持模型列表

- PP-OCRv3部署模型实现来自[PP-OCR系列模型列表](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md)

## 准备PPOCRv3部署模型以及转换模型

PPOCRv3包括文本框检测模型（ch_PP-OCRv3_det）、方向分类模型（ch_ppocr_mobile_v2.0_cls）、文字识别模型（ch_PP-OCRv3_rec）  
SOPHGO-TPU部署模型前需要将以上Paddle模型转换成bmodel模型，我们以ch_PP-OCRv3_det模型为例，具体步骤如下:
- 下载Paddle模型[ch_PP-OCRv3_det](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar)
- Pddle模型转换为ONNX模型，请参考[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)
- ONNX模型转换bmodel模型的过程，请参考[TPU-MLIR](https://github.com/sophgo/tpu-mlir)

## 模型转换example

### 下载ch_PP-OCRv3_det模型,并转换为ONNX模型
```shell
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar xvf ch_PP-OCRv3_det_infer.tar

# 修改ch_PP-OCRv3_det模型的输入shape，由动态输入变成固定输入
python paddle_infer_shape.py --model_dir ch_PP-OCRv3_det_infer \
                             --model_filename inference.pdmodel \
                             --params_filename inference.pdiparams \
                             --save_dir ch_PP-OCRv3_det_infer_fix \
                             --input_shape_dict="{'x':[1,3,960,608]}"

#将固定输入的Paddle模型转换成ONNX模型
paddle2onnx --model_dir ch_PP-OCRv3_det_infer_fix \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ch_PP-OCRv3_det_infer_fix.onnx \
            --enable_dev_version True
```

### 导出bmodel模型

以转换BM1684x的bmodel模型为例子，我们需要下载[TPU-MLIR](https://github.com/sophgo/tpu-mlir)工程，安装过程具体参见[TPU-MLIR文档](https://github.com/sophgo/tpu-mlir/blob/master/README.md)。
### 1.	安装
``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234是一个示例，也可以设置其他名字
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest

source ./envsetup.sh
./build.sh
```

### 2.	ONNX模型转换为bmodel模型
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

## 其他链接
- [Cpp部署](./cpp)
