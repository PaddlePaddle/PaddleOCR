# Web 端基础预测功能测试

Web 端主要基于 Jest-Puppeteer 完成 e2e 测试，其中 Puppeteer 操作 Chrome 完成推理流程，Jest 完成测试流程。
>Puppeteer 是一个 Node 库，它提供了一个高级 API 来通过 DevTools 协议控制 Chromium 或 Chrome
>Jest 是一个 JavaScript 测试框架，旨在确保任何 JavaScript 代码的正确性。
#### 环境准备

* 安装 Node（包含 npm ） （https://nodejs.org/zh-cn/download/）
* 确认是否安装成功，在命令行执行
```sh
# 显示所安 node 版本号，即表示成功安装
node -v
```
* 确认 npm 是否安装成成
```sh
# npm 随着 node 一起安装，一般无需额外安装
# 显示所安 npm 版本号，即表示成功安装
npm -v
```

#### 使用
```sh
# web 测试环境准备
bash test_tipc/prepare_js.sh 'js_infer'
# web 推理测试
bash test_tipc/test_inference_js.sh
```

#### 流程设计

###### paddlejs prepare
 1. 判断 node, npm 是否安装
 2. 下载测试模型，当前检测模型是 ch_PP-OCRv2_det_infer ，识别模型是 ch_PP-OCRv2_rec_infer[1, 3, 32, 320]。如果需要替换模型，可直接将模型文件放在test_tipc/web/models/目录下。
  - 文本检测模型：https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar
  - 文本识别模型：https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar
  - 文本识别模型[1, 3, 32, 320]：https://paddlejs.bj.bcebos.com/models/ch_PP-OCRv2_rec_infer.tar
  - 保证较为准确的识别效果，需要将文本识别模型导出为输入shape是[1, 3, 32, 320]的静态模型
 3. 转换模型， model.pdmodel model.pdiparams 转换为 model.json chunk.dat（检测模型保存地址：test_tipc/web/models/ch_PP-OCRv2/det，识别模型保存地址：test_tipc/web/models/ch_PP-OCRv2/rec）
 4. 安装最新版本 ocr sdk  @paddlejs-models/ocr@latest
 5. 安装测试环境依赖 puppeteer、jest、jest-puppeteer，如果检查到已经安装，则不会进行二次安装

 ###### paddlejs infer test
 1. Jest 执行 server command：`python3 -m http.server 9811` 开启本地服务
 2. 启动 Jest 测试服务，通过 jest-puppeteer 插件完成 chrome 操作，加载 @paddlejs-models/ocr 脚本完成推理流程
 3. 测试用例为原图识别后的文本结果与预期文本结果（expect.json）进行对比，测试通过有两个标准：
    * 原图识别结果逐字符与预期结果对比，误差不超过 **10个字符**；
    * 原图识别结果每个文本框字符内容与预期结果进行相似度对比，相似度不小于 0.9（全部一致则相似度为1）。

    只有满足上述两个标准，视为测试通过。通过为如下显示：
 <img width="600" src="https://user-images.githubusercontent.com/43414102/146406599-80b30c66-f2f8-4f57-a68a-007c479ff0f7.png">
