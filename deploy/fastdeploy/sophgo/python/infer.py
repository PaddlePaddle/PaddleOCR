import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--det_model", required=True, help="Path of Detection model of PPOCR.")
    parser.add_argument(
        "--cls_model",
        required=True,
        help="Path of Classification model of PPOCR.")
    parser.add_argument(
        "--rec_model",
        required=True,
        help="Path of Recognization model of PPOCR.")
    parser.add_argument(
        "--rec_label_file",
        required=True,
        help="Path of Recognization label of PPOCR.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")

    return parser.parse_args()


args = parse_arguments()

# 配置runtime，加载模型
runtime_option = fd.RuntimeOption()
runtime_option.use_sophgo()

# Detection模型, 检测文字框
det_model_file = args.det_model
det_params_file = ""
# Classification模型，方向分类，可选
cls_model_file = args.cls_model
cls_params_file = ""
# Recognition模型，文字识别模型
rec_model_file = args.rec_model
rec_params_file = ""
rec_label_file = args.rec_label_file

# PPOCR的cls和rec模型现在已经支持推理一个Batch的数据
# 定义下面两个变量后, 可用于设置trt输入shape, 并在PPOCR模型初始化后, 完成Batch推理设置
cls_batch_size = 1
rec_batch_size = 1

# 当使用TRT时，分别给三个模型的runtime设置动态shape,并完成模型的创建.
# 注意: 需要在检测模型创建完成后，再设置分类模型的动态输入并创建分类模型, 识别模型同理.
# 如果用户想要自己改动检测模型的输入shape, 我们建议用户把检测模型的长和高设置为32的倍数.
det_option = runtime_option
det_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640],
                               [1, 3, 960, 960])
# 用户可以把TRT引擎文件保存至本地
# det_option.set_trt_cache_file(args.det_model  + "/det_trt_cache.trt")
det_model = fd.vision.ocr.DBDetector(
    det_model_file,
    det_params_file,
    runtime_option=det_option,
    model_format=fd.ModelFormat.SOPHGO)

cls_option = runtime_option
cls_option.set_trt_input_shape("x", [1, 3, 48, 10],
                               [cls_batch_size, 3, 48, 320],
                               [cls_batch_size, 3, 48, 1024])
# 用户可以把TRT引擎文件保存至本地
# cls_option.set_trt_cache_file(args.cls_model  + "/cls_trt_cache.trt")
cls_model = fd.vision.ocr.Classifier(
    cls_model_file,
    cls_params_file,
    runtime_option=cls_option,
    model_format=fd.ModelFormat.SOPHGO)

rec_option = runtime_option
rec_option.set_trt_input_shape("x", [1, 3, 48, 10],
                               [rec_batch_size, 3, 48, 320],
                               [rec_batch_size, 3, 48, 2304])
# 用户可以把TRT引擎文件保存至本地
# rec_option.set_trt_cache_file(args.rec_model  + "/rec_trt_cache.trt")
rec_model = fd.vision.ocr.Recognizer(
    rec_model_file,
    rec_params_file,
    rec_label_file,
    runtime_option=rec_option,
    model_format=fd.ModelFormat.SOPHGO)

# 创建PP-OCR，串联3个模型，其中cls_model可选，如无需求，可设置为None
ppocr_v3 = fd.vision.ocr.PPOCRv3(
    det_model=det_model, cls_model=cls_model, rec_model=rec_model)

# 需要使用下行代码, 来启用rec模型的静态shape推理，这里rec模型的静态输入为[3, 48, 584]
rec_model.preprocessor.static_shape_infer = True
rec_model.preprocessor.rec_image_shape = [3, 48, 584]

# 给cls和rec模型设置推理时的batch size
# 此值能为-1, 和1到正无穷
# 当此值为-1时, cls和rec模型的batch size将默认和det模型检测出的框的数量相同
ppocr_v3.cls_batch_size = cls_batch_size
ppocr_v3.rec_batch_size = rec_batch_size

# 预测图片准备
im = cv2.imread(args.image)

#预测并打印结果
result = ppocr_v3.predict(im)

print(result)

# 可视化结果
vis_im = fd.vision.vis_ppocr(im, result)
cv2.imwrite("sophgo_result.jpg", vis_im)
print("Visualized result save in ./sophgo_result.jpg")
