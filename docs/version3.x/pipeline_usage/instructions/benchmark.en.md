# Pipeline Benchmark

## Table of Contents

- [1. Instructions](#1.-instructions)
- [2. Usage Examples](#2.-usage-examples)

## 1. Instructions

The Benchmark feature calculates the average execution time of all operations during end-to-end pipeline inference and provides summary information. The time unit is milliseconds.

The benchmark feature needs to be enabled via environment variables as follows:

* `PADDLE_PDX_PIPELINE_BENCHMARK`: Set to `True` to enable the benchmark feature. The default is `False`.

The following table describes the methods and parameters related to pipeline inference benchmark:

<table border="1" width="100%" cellpadding="5">
    <tr>
        <th>Method Name</th>
        <th>Description</th>
    </tr>
    <tr>
        <td><code>start_warmup()</code></td>
        <td>Start the benchmark warmup.</td>
    </tr>
    <tr>
        <td><code>stop_warmup()</code></td>
        <td>Stop the warmup and clear all benchmark data generated during warmup.</td>
    </tr>
        <tr>
        <td><code>print_detail_data()</code></td>
        <td>Print detailed benchmark data, including the sequence (Step), name (Operation), and average time (Time) of each operation.</td>
    </tr>
    <tr>
        <td><code>print_summary_data()</code></td>
        <td>Print summary benchmark data, including the level (Level), name (Operation), and total average time (Time) of each operation. The Time for level 1 represents the total average time.</td>
    </tr>
    <tr>
        <td><code>print_operation_info()</code></td>
        <td>Print the source code location of each operation.</td>
    </tr>
    <tr>
        <td><code>print_pipeline_data()</code></td>
        <td>Print the detail data, summary data, and operation_info data of the benchmark to the console.</td>
    </tr>
    <tr>
        <td><code>save_pipeline_data(save_path)</code></td>
        <td><code>save_path</code>: string <br />Save the benchmark data to the specified file path, including detailed benchmark data <code>detail.csv</code> and summary benchmark data <code>summary.csv</code>.</td>
    </tr>
    <tr>
        <td><code>reset()</code></td>
        <td>Clear existing benchmark data.</td>
    </tr>
</table>


## 2. Usage Examples

Create a `test_infer.py` script:

```python
from paddleocr import PaddleOCR, benchmark

pipeline = PaddleOCR()
image = "general_ocr_002.png"

benchmark.start_warmup() # Start warmup
for _ in range(50):
    pipeline.predict(image)
benchmark.stop_warmup() # End warmup

for _ in range(100): # Start formal speed measurement
    pipeline.predict(image)

benchmark.print_pipeline_data()  # Print summary benchmark data
benchmark.save_pipeline_data("./benchmark") # Save benchmark data to the benchmark folder
```

Execute the script:

```bash
PADDLE_PDX_PIPELINE_BENCHMARK=True python test_infer.py
```

The benchmark results obtained from running the example program are as follows:

```
                                                             Operation Info
+-----------------------------------------------------+-------------------------------------------------------------------------------+
|                      Operation                      |                              Source Code Location                             |
+-----------------------------------------------------+-------------------------------------------------------------------------------+
|                      ReadImage                      |       /PaddleX/paddlex/inference/common/reader/image_reader.py:47       |
|                   DocTrPostProcess                  |    /PaddleX/paddlex/inference/models/image_unwarping/processors.py:51   |
|                   DetResizeForTest                  |    /PaddleX/paddlex/inference/models/text_detection/processors.py:58    |
|     _DocPreprocessorPipeline.get_model_settings     |  /PaddleX/paddlex/inference/pipelines/doc_preprocessor/pipeline.py:110  |
|                         Crop                        | /PaddleX/paddlex/inference/models/image_classification/processors.py:45 |
|                PaddleInferChainLegacy               |       /PaddleX/paddlex/inference/models/common/static_infer.py:248      |
|              _OCRPipeline.rotate_image              |         /PaddleX/paddlex/inference/pipelines/ocr/pipeline.py:140        |
|           _DocPreprocessorPipeline.predict          |  /PaddleX/paddlex/inference/pipelines/doc_preprocessor/pipeline.py:133  |
|                 ClasPredictor.apply                 |  /PaddleX/paddlex/inference/models/base/predictor/base_predictor.py:213 |
|                 WarpPredictor.apply                 |  /PaddleX/paddlex/inference/models/base/predictor/base_predictor.py:213 |
|                TextDetPredictor.apply               |  /PaddleX/paddlex/inference/models/base/predictor/base_predictor.py:213 |
|                    ResizeByShort                    |    /PaddleX/paddlex/inference/models/common/vision/processors.py:203    |
|                      Normalize                      |    /PaddleX/paddlex/inference/models/common/vision/processors.py:268    |
|                    DBPostProcess                    |    /PaddleX/paddlex/inference/models/text_detection/processors.py:487   |
|                TextRecPredictor.apply               |  /PaddleX/paddlex/inference/models/base/predictor/base_predictor.py:213 |
|                    NormalizeImage                   |    /PaddleX/paddlex/inference/models/text_detection/processors.py:252   |
| _DocPreprocessorPipeline.check_model_settings_valid |   /PaddleX/paddlex/inference/pipelines/doc_preprocessor/pipeline.py:82  |
|           _OCRPipeline.get_model_settings           |         /PaddleX/paddlex/inference/pipelines/ocr/pipeline.py:204        |
|           _OCRPipeline.get_text_det_params          |         /PaddleX/paddlex/inference/pipelines/ocr/pipeline.py:236        |
|                        Resize                       |    /PaddleX/paddlex/inference/models/common/vision/processors.py:117    |
|                    CTCLabelDecode                   |   /PaddleX/paddlex/inference/models/text_recognition/processors.py:189  |
|                         Topk                        | /PaddleX/paddlex/inference/models/image_classification/processors.py:83 |
|                  OCRReisizeNormImg                  |   /PaddleX/paddlex/inference/models/text_recognition/processors.py:65   |
|                       ToBatch                       |   /PaddleX/paddlex/inference/models/text_recognition/processors.py:235  |
|                      ToCHWImage                     |    /PaddleX/paddlex/inference/models/common/vision/processors.py:277    |
|                 _OCRPipeline.predict                |         /PaddleX/paddlex/inference/pipelines/ocr/pipeline.py:282        |
|                       ToBatch                       |    /PaddleX/paddlex/inference/models/common/vision/processors.py:284    |
|       _OCRPipeline.check_model_settings_valid       |         /PaddleX/paddlex/inference/pipelines/ocr/pipeline.py:176        |
+-----------------------------------------------------+-------------------------------------------------------------------------------+
                                           Detail Data
+------+----------------------------------------------------------------+-----------------------+
| Step | Operation                                                      | Time                  |
+------+----------------------------------------------------------------+-----------------------+
|  1   | _OCRPipeline.predict                                           | 375.11244628956774    |
|  2   |     -> _OCRPipeline.get_model_settings                         | 0.00428391998866573   |
|  3   |     -> _OCRPipeline.check_model_settings_valid                 | 0.0024828016466926783 |
|  4   |     -> _OCRPipeline.get_text_det_params                        | 0.005152080120751634  |
|  5   |     -> ReadImage                                               | 3.2029549301660154    |
|  6   |     -> _DocPreprocessorPipeline.predict                        | 27.310913350374904    |
|  7   |         -> _DocPreprocessorPipeline.get_model_settings         | 0.004107539862161502  |
|  8   |         -> _DocPreprocessorPipeline.check_model_settings_valid | 0.0016830896493047476 |
|  9   |         -> ReadImage                                           | 0.0029576495580840856 |
|  10  |         -> ClasPredictor.apply                                 | 4.701614730001893     |
|  11  |             -> ReadImage                                       | 0.13587839042884298   |
|  12  |             -> ResizeByShort                                   | 0.3281894406245556    |
|  13  |             -> Crop                                            | 0.01503000981756486   |
|  14  |             -> Normalize                                       | 0.3884544402535539    |
|  15  |             -> ToCHWImage                                      | 0.006330519245238975  |
|  16  |             -> ToBatch                                         | 0.14169737987685949   |
|  17  |             -> PaddleInferChainLegacy                          | 3.283889550511958     |
|  18  |             -> Topk                                            | 0.10010718091507442   |
|  19  |         -> WarpPredictor.apply                                 | 21.893062600429403    |
|  20  |             -> ReadImage                                       | 0.004573430051095784  |
|  21  |             -> Normalize                                       | 4.245691860560328     |
|  22  |             -> ToCHWImage                                      | 0.005895959911867976  |
|  23  |             -> ToBatch                                         | 1.7250755491841119    |
|  24  |             -> PaddleInferChainLegacy                          | 10.887994960212382    |
|  25  |             -> DocTrPostProcess                                | 1.4253830898087472    |
|  26  |     -> TextDetPredictor.apply                                  | 49.976056129235076    |
|  27  |         -> ReadImage                                           | 0.004843260976485908  |
|  28  |         -> DetResizeForTest                                    | 3.3269549095712136    |
|  29  |         -> NormalizeImage                                      | 2.9576204597833566    |
|  30  |         -> ToCHWImage                                          | 0.005182310123927891  |
|  31  |         -> ToBatch                                             | 1.046062790119322     |
|  32  |         -> PaddleInferChainLegacy                              | 34.70224040953326     |
|  33  |         -> DBPostProcess                                       | 5.826775671303039     |
|  34  |     -> ClasPredictor.apply                                     | 23.43678753997665     |
|  35  |         -> ReadImage                                           | 0.0633359991479665    |
|  36  |         -> Resize                                              | 0.24419097986537963   |
|  37  |         -> Normalize                                           | 0.480741420033155     |
|  38  |         -> ToCHWImage                                          | 0.0066608507768251    |
|  39  |         -> ToBatch                                             | 0.18536171046434902   |
|  40  |         -> PaddleInferChainLegacy                              | 3.3766339404974133    |
|  41  |         -> Topk                                                | 0.15909907990135252   |
|  42  |         -> ReadImage                                           | 0.0395357194065582    |
|  43  |         -> Resize                                              | 0.2085290702234488    |
|  44  |         -> Normalize                                           | 0.4068155895220116    |
|  45  |         -> ToCHWImage                                          | 0.005677459557773545  |
|  46  |         -> ToBatch                                             | 0.11155156986205839   |
|  47  |         -> PaddleInferChainLegacy                              | 2.7268862597702537    |
|  48  |         -> Topk                                                | 0.13428127014776692   |
|  49  |         -> ReadImage                                           | 0.032502070971531793  |
|  50  |         -> Resize                                              | 0.20152631899691187   |
|  51  |         -> Normalize                                           | 0.347195100330282     |
|  52  |         -> ToCHWImage                                          | 0.005517759709618986  |
|  53  |         -> ToBatch                                             | 0.10656953061698005   |
|  54  |         -> PaddleInferChainLegacy                              | 2.612808299745666     |
|  55  |         -> Topk                                                | 0.13188434022595175   |
|  56  |         -> ReadImage                                           | 0.03589507090509869   |
|  57  |         -> Resize                                              | 0.2076980892161373    |
|  58  |         -> Normalize                                           | 0.3592138692329172    |
|  59  |         -> ToCHWImage                                          | 0.005206359783187509  |
|  60  |         -> ToBatch                                             | 0.1359267797670327    |
|  61  |         -> PaddleInferChainLegacy                              | 2.619662079960108     |
|  62  |         -> Topk                                                | 0.130717080028262     |
|  63  |         -> ReadImage                                           | 0.038393009890569374  |
|  64  |         -> Resize                                              | 0.19743553988519125   |
|  65  |         -> Normalize                                           | 0.33197281998582184   |
|  66  |         -> ToCHWImage                                          | 0.00512515107402578   |
|  67  |         -> ToBatch                                             | 0.10293568033375777   |
|  68  |         -> PaddleInferChainLegacy                              | 2.5824282996472903    |
|  69  |         -> Topk                                                | 0.129485729848966     |
|  70  |         -> ReadImage                                           | 0.04028105002362281   |
|  71  |         -> Resize                                              | 0.10972122952807695   |
|  72  |         -> Normalize                                           | 0.1787920702190604    |
|  73  |         -> ToCHWImage                                          | 0.00408922991482541   |
|  74  |         -> ToBatch                                             | 0.05458273953991011   |
|  75  |         -> PaddleInferChainLegacy                              | 2.262636839877814     |
|  76  |         -> Topk                                                | 0.1055472502775956    |
|  77  |     -> _OCRPipeline.rotate_image                               | 0.05102259965497069   |
|  78  |     -> TextRecPredictor.apply                                  | 169.44437422047486    |
|  79  |         -> ReadImage                                           | 0.004737989947898313  |
|  80  |         -> OCRReisizeNormImg                                   | 0.46037410967983305   |
|  81  |         -> ToBatch                                             | 0.6405122207070235    |
|  82  |         -> PaddleInferChainLegacy                              | 15.439773340767715    |
|  83  |         -> CTCLabelDecode                                      | 10.742378439754248    |
|  84  |         -> ReadImage                                           | 0.006349970353767276  |
|  85  |         -> OCRReisizeNormImg                                   | 0.6252558408596087    |
|  86  |         -> ToBatch                                             | 0.7338531101413537    |
|  87  |         -> PaddleInferChainLegacy                              | 15.204189889482222    |
|  88  |         -> CTCLabelDecode                                      | 6.7516070799320005    |
|  89  |         -> ReadImage                                           | 0.006978959863772616  |
|  90  |         -> OCRReisizeNormImg                                   | 0.7167729703360237    |
|  91  |         -> ToBatch                                             | 0.6568272292497568    |
|  92  |         -> PaddleInferChainLegacy                              | 14.973864750063512    |
|  93  |         -> CTCLabelDecode                                      | 6.695752280211309     |
|  94  |         -> ReadImage                                           | 0.0070425499870907515 |
|  95  |         -> OCRReisizeNormImg                                   | 0.7757280093210284    |
|  96  |         -> ToBatch                                             | 0.6442721793428063    |
|  97  |         -> PaddleInferChainLegacy                              | 15.027350780292181    |
|  98  |         -> CTCLabelDecode                                      | 6.661591530573787     |
|  99  |         -> ReadImage                                           | 0.007066540565574542  |
| 100  |         -> OCRReisizeNormImg                                   | 0.9195591000025161    |
| 101  |         -> ToBatch                                             | 0.7951801503077149    |
| 102  |         -> PaddleInferChainLegacy                              | 15.379044259898365    |
| 103  |         -> CTCLabelDecode                                      | 9.372330370388227     |
| 104  |         -> ReadImage                                           | 0.006225309771252796  |
| 105  |         -> OCRReisizeNormImg                                   | 1.1437026296334807    |
| 106  |         -> ToBatch                                             | 1.091715270158602     |
| 107  |         -> PaddleInferChainLegacy                              | 23.505835609685164    |
| 108  |         -> CTCLabelDecode                                      | 17.118994210031815    |
+------+----------------------------------------------------------------+-----------------------+
                                      Summary Data
+-------+-----------------------------------------------------+-----------------------+
| Level | Operation                                           | Time                  |
+-------+-----------------------------------------------------+-----------------------+
|   1   | _OCRPipeline.predict                                | 375.11244628956774    |
|       |                                                     |                       |
|   2   | Layer                                               | 375.11244628956774    |
|       | Core                                                | 273.4340275716386     |
|       | Other                                               | 101.67841871792916    |
|       | _OCRPipeline.get_model_settings                     | 0.00428391998866573   |
|       | _OCRPipeline.check_model_settings_valid             | 0.0024828016466926783 |
|       | _OCRPipeline.get_text_det_params                    | 0.005152080120751634  |
|       | ReadImage                                           | 3.2029549301660154    |
|       | _DocPreprocessorPipeline.predict                    | 27.310913350374904    |
|       | TextDetPredictor.apply                              | 49.976056129235076    |
|       | ClasPredictor.apply                                 | 23.43678753997665     |
|       | _OCRPipeline.rotate_image                           | 0.05102259965497069   |
|       | TextRecPredictor.apply                              | 169.44437422047486    |
|       |                                                     |                       |
|   3   | Layer                                               | 270.1681312400615     |
|       | Core                                                | 261.8130224109336     |
|       | Other                                               | 8.355108829127857     |
|       | _DocPreprocessorPipeline.get_model_settings         | 0.004107539862161502  |
|       | _DocPreprocessorPipeline.check_model_settings_valid | 0.0016830896493047476 |
|       | ReadImage                                           | 0.29614515136927366   |
|       | ClasPredictor.apply                                 | 4.701614730001893     |
|       | WarpPredictor.apply                                 | 21.893062600429403    |
|       | DetResizeForTest                                    | 3.3269549095712136    |
|       | NormalizeImage                                      | 2.9576204597833566    |
|       | ToCHWImage                                          | 0.03745912094018422   |
|       | ToBatch                                             | 6.305350960610667     |
|       | PaddleInferChainLegacy                              | 150.41335475922097    |
|       | DBPostProcess                                       | 5.826775671303039     |
|       | Resize                                              | 1.1691012277151458    |
|       | Normalize                                           | 2.104730869323248     |
|       | Topk                                                | 0.7910147504298948    |
|       | OCRReisizeNormImg                                   | 4.641392659832491     |
|       | CTCLabelDecode                                      | 57.342653910891386    |
|       |                                                     |                       |
|   4   | Layer                                               | 26.594677330431296    |
|       | Core                                                | 22.69419176140218     |
|       | Other                                               | 3.900485569029115     |
|       | ReadImage                                           | 0.14045182047993876   |
|       | ResizeByShort                                       | 0.3281894406245556    |
|       | Crop                                                | 0.01503000981756486   |
|       | Normalize                                           | 4.634146300813882     |
|       | ToCHWImage                                          | 0.012226479157106951  |
|       | ToBatch                                             | 1.8667729290609714    |
|       | PaddleInferChainLegacy                              | 14.17188451072434     |
|       | Topk                                                | 0.10010718091507442   |
|       | DocTrPostProcess                                    | 1.4253830898087472    |
+-------+-----------------------------------------------------+-----------------------+
```

The results will be saved locally to: `./benchmark/detail.csv` and `./benchmark/summary.csv`:

The content of `detail.csv` is as follows:

```
Step,Operation,Time
1,_OCRPipeline.predict,375.11244628956774
2,    -> _OCRPipeline.get_model_settings,0.00428391998866573
3,    -> _OCRPipeline.check_model_settings_valid,0.0024828016466926783
4,    -> _OCRPipeline.get_text_det_params,0.005152080120751634
5,    -> ReadImage,3.2029549301660154
6,    -> _DocPreprocessorPipeline.predict,27.310913350374904
7,        -> _DocPreprocessorPipeline.get_model_settings,0.004107539862161502
8,        -> _DocPreprocessorPipeline.check_model_settings_valid,0.0016830896493047476
9,        -> ReadImage,0.0029576495580840856
10,        -> ClasPredictor.apply,4.701614730001893
11,            -> ReadImage,0.13587839042884298
12,            -> ResizeByShort,0.3281894406245556
13,            -> Crop,0.01503000981756486
14,            -> Normalize,0.3884544402535539
15,            -> ToCHWImage,0.006330519245238975
16,            -> ToBatch,0.14169737987685949
17,            -> PaddleInferChainLegacy,3.283889550511958
18,            -> Topk,0.10010718091507442
19,        -> WarpPredictor.apply,21.893062600429403
20,            -> ReadImage,0.004573430051095784
21,            -> Normalize,4.245691860560328
22,            -> ToCHWImage,0.005895959911867976
23,            -> ToBatch,1.7250755491841119
24,            -> PaddleInferChainLegacy,10.887994960212382
25,            -> DocTrPostProcess,1.4253830898087472
26,    -> TextDetPredictor.apply,49.976056129235076
27,        -> ReadImage,0.004843260976485908
28,        -> DetResizeForTest,3.3269549095712136
29,        -> NormalizeImage,2.9576204597833566
30,        -> ToCHWImage,0.005182310123927891
31,        -> ToBatch,1.046062790119322
32,        -> PaddleInferChainLegacy,34.70224040953326
33,        -> DBPostProcess,5.826775671303039
34,    -> ClasPredictor.apply,23.43678753997665
35,        -> ReadImage,0.0633359991479665
36,        -> Resize,0.24419097986537963
37,        -> Normalize,0.480741420033155
38,        -> ToCHWImage,0.0066608507768251
39,        -> ToBatch,0.18536171046434902
40,        -> PaddleInferChainLegacy,3.3766339404974133
41,        -> Topk,0.15909907990135252
42,        -> ReadImage,0.0395357194065582
43,        -> Resize,0.2085290702234488
44,        -> Normalize,0.4068155895220116
45,        -> ToCHWImage,0.005677459557773545
46,        -> ToBatch,0.11155156986205839
47,        -> PaddleInferChainLegacy,2.7268862597702537
48,        -> Topk,0.13428127014776692
49,        -> ReadImage,0.032502070971531793
50,        -> Resize,0.20152631899691187
51,        -> Normalize,0.347195100330282
52,        -> ToCHWImage,0.005517759709618986
53,        -> ToBatch,0.10656953061698005
54,        -> PaddleInferChainLegacy,2.612808299745666
55,        -> Topk,0.13188434022595175
56,        -> ReadImage,0.03589507090509869
57,        -> Resize,0.2076980892161373
58,        -> Normalize,0.3592138692329172
59,        -> ToCHWImage,0.005206359783187509
60,        -> ToBatch,0.1359267797670327
61,        -> PaddleInferChainLegacy,2.619662079960108
62,        -> Topk,0.130717080028262
63,        -> ReadImage,0.038393009890569374
64,        -> Resize,0.19743553988519125
65,        -> Normalize,0.33197281998582184
66,        -> ToCHWImage,0.00512515107402578
67,        -> ToBatch,0.10293568033375777
68,        -> PaddleInferChainLegacy,2.5824282996472903
69,        -> Topk,0.129485729848966
70,        -> ReadImage,0.04028105002362281
71,        -> Resize,0.10972122952807695
72,        -> Normalize,0.1787920702190604
73,        -> ToCHWImage,0.00408922991482541
74,        -> ToBatch,0.05458273953991011
75,        -> PaddleInferChainLegacy,2.262636839877814
76,        -> Topk,0.1055472502775956
77,    -> _OCRPipeline.rotate_image,0.05102259965497069
78,    -> TextRecPredictor.apply,169.44437422047486
79,        -> ReadImage,0.004737989947898313
80,        -> OCRReisizeNormImg,0.46037410967983305
81,        -> ToBatch,0.6405122207070235
82,        -> PaddleInferChainLegacy,15.439773340767715
83,        -> CTCLabelDecode,10.742378439754248
84,        -> ReadImage,0.006349970353767276
85,        -> OCRReisizeNormImg,0.6252558408596087
86,        -> ToBatch,0.7338531101413537
87,        -> PaddleInferChainLegacy,15.204189889482222
88,        -> CTCLabelDecode,6.7516070799320005
89,        -> ReadImage,0.006978959863772616
90,        -> OCRReisizeNormImg,0.7167729703360237
91,        -> ToBatch,0.6568272292497568
92,        -> PaddleInferChainLegacy,14.973864750063512
93,        -> CTCLabelDecode,6.695752280211309
94,        -> ReadImage,0.0070425499870907515
95,        -> OCRReisizeNormImg,0.7757280093210284
96,        -> ToBatch,0.6442721793428063
97,        -> PaddleInferChainLegacy,15.027350780292181
98,        -> CTCLabelDecode,6.661591530573787
99,        -> ReadImage,0.007066540565574542
100,        -> OCRReisizeNormImg,0.9195591000025161
101,        -> ToBatch,0.7951801503077149
102,        -> PaddleInferChainLegacy,15.379044259898365
103,        -> CTCLabelDecode,9.372330370388227
104,        -> ReadImage,0.006225309771252796
105,        -> OCRReisizeNormImg,1.1437026296334807
106,        -> ToBatch,1.091715270158602
107,        -> PaddleInferChainLegacy,23.505835609685164
108,        -> CTCLabelDecode,17.118994210031815
```

The content of `summary.csv` is as follows:

```
Level, Operation, Time
1, _OCRPipeline.predict, 375.11244628956774
,,
2, Layer, 375.11244628956774
, Core, 273.4340275716386
, Other, 101.67841871792916
, _OCRPipeline.get_model_settings, 0.00428391998866573
, _OCRPipeline.check_model_settings_valid, 0.0024828016466926783
, _OCRPipeline.get_text_det_params, 0.005152080120751634
, ReadImage, 3.2029549301660154
, _DocPreprocessorPipeline.predict, 27.310913350374904
, TextDetPredictor.apply, 49.976056129235076
, ClasPredictor.apply, 23.43678753997665
, _OCRPipeline.rotate_image, 0.05102259965497069
, TextRecPredictor.apply, 169.44437422047486
,,
3, Layer, 270.1681312400615
, Core, 261.8130224109336
, Other, 8.355108829127857
, _DocPreprocessorPipeline.get_model_settings, 0.004107539862161502
, _DocPreprocessorPipeline.check_model_settings_valid, 0.0016830896493047476
, ReadImage, 0.29614515136927366
, ClasPredictor.apply, 4.701614730001893
, WarpPredictor.apply, 21.893062600429403
, DetResizeForTest, 3.3269549095712136
, NormalizeImage, 2.9576204597833566
, ToCHWImage, 0.03745912094018422
, ToBatch, 6.305350960610667
, PaddleInferChainLegacy, 150.41335475922097
, DBPostProcess, 5.826775671303039
, Resize, 1.1691012277151458
, Normalize, 2.104730869323248
, Topk, 0.7910147504298948
, OCRReisizeNormImg, 4.641392659832491
, CTCLabelDecode, 57.342653910891386
,,
4, Layer, 26.594677330431296
, Core, 22.69419176140218
, Other, 3.900485569029115
, ReadImage, 0.14045182047993876
, ResizeByShort, 0.3281894406245556
, Crop, 0.01503000981756486
, Normalize, 4.634146300813882
, ToCHWImage, 0.012226479157106951
, ToBatch, 1.8667729290609714
, PaddleInferChainLegacy, 14.17188451072434
, Topk, 0.10010718091507442
, DocTrPostProcess, 1.4253830898087472
```
