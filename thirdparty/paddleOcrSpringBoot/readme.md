简体中文

- 使用本教程前请先基于PaddleHub Serving的部署.

# 基于PaddleHub Serving的Java SpringBoot调用

paddleOcrSpringBoot服务部署目录下包括全部SpringBoot代码。目录结构如下：
```
deploy/paddleOcrSpringBoot/
  └─  src     代码文件
       └─  main     主函数代码
            └─  java\com\paddelocr_springboot\demo
                                               └─ DemoApplication.java  SpringBoot启动代码
                                               └─ Controller  
                                                   └─ OCR.java    控制器代码
       └─  test     测试代码
```

- Hub Serving启动后的APi端口如下：
`http://[ip_address]:[port]/predict/[module_name]`  

## 返回结果格式说明
返回结果为列表（list），列表中的每一项为词典（dict），词典一共可能包含3种字段，信息如下：

|字段名称|数据类型|意义|
|-|-|-|
|text|str|文本内容|
|confidence|float| 文本识别置信度|
|text_region|list|文本位置坐标|

不同模块返回的字段不同，如，文本识别服务模块返回结果不含`text_region`字段，具体信息如下：

|字段名/模块名|ocr_det|ocr_rec|ocr_system|
|-|-|-|-|
|text||✔|✔|
|confidence||✔|✔|
|text_region|✔||✔|
