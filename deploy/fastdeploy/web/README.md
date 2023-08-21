[English](README.md) | 简体中文
# PP-OCRv3 前端部署示例

本节介绍部署PaddleOCR的PP-OCRv3模型在浏览器中运行，以及@paddle-js-models/ocr npm包中的js接口。


## 1. 前端部署PP-OCRv3模型
PP-OCRv3模型web demo使用[**参考文档**](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/application/js/web_demo)

## 2. PP-OCRv3 js接口

```
import * as ocr from "@paddle-js-models/ocr";
await ocr.init(detConfig, recConfig);
const res = await ocr.recognize(img, option, postConfig);
```
ocr模型加载和初始化，其中模型为Paddle.js模型格式，js模型转换方式参考[文档](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/application/js/web_demo/README.md)

**init函数参数**

> * **detConfig**(dict): 文本检测模型配置参数，默认值为 {modelPath: 'https://js-models.bj.bcebos.com/PaddleOCR/PP-OCRv3/ch_PP-OCRv3_det_infer_js_960/model.json', fill: '#fff', mean: [0.485, 0.456, 0.406],std: [0.229, 0.224, 0.225]}; 其中，modelPath为文本检测模型路径，fill 为图像预处理padding的值，mean和std分别为预处理的均值和标准差
> * **recConfig**(dict)): 文本识别模型配置参数，默认值为 {modelPath: 'https://js-models.bj.bcebos.com/PaddleOCR/PP-OCRv3/ch_PP-OCRv3_rec_infer_js/model.json', fill: '#000', mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5]}; 其中，modelPath为文本检测模型路径，fill 为图像预处理padding的值，mean和std分别为预处理的均值和标准差


**recognize函数参数**

> * **img**(HTMLImageElement): 输入图像参数，类型为HTMLImageElement。
> * **option**(dict): 可视化文本检测框的canvas参数，可不用设置。
> * **postConfig**(dict): 文本检测后处理参数，默认值为：{shape: 960, thresh: 0.3, box_thresh: 0.6, unclip_ratio:1.5}; thresh是输出预测图的二值化阈值；box_thresh是输出框的阈值，低于此值的预测框会被丢弃，unclip_ratio是输出框扩大的比例。


## 其它文档
- [PP-OCRv3 微信小程序部署文档](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/application/js/mini_program)
