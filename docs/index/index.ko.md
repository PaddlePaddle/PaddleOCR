---
comments: true
hide:
  - navigation
  - toc
---

<div align="center">
 <img src="https://github.com/PaddlePaddle/PaddleOCR/releases/download/v2.9.1/PaddleOCR_log.png" align="middle" width = "600"/>
  <p align="center">
      <a href="https://discord.gg/z9xaRVjdbD"><img src="https://img.shields.io/badge/Chat-on%20discord-7289da.svg?sanitize=true" alt="Chat"></a>
      <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/{{PADDLEOCR_GITHUB_REF}}/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
      <a href="https://github.com/PaddlePaddle/PaddleOCR/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleOCR?color=ffa"></a>
      <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
      <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
      <a href="https://pypi.org/project/PaddleOCR/"><img src="https://img.shields.io/pypi/dm/PaddleOCR?color=9cf"></a>
      <a href="https://github.com/PaddlePaddle/PaddleOCR/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf"></a>
  </p>
</div>

## 소개

PaddleOCR은 사용자들이 보다 나은 모델을 훈련하여 실전에 투입하는데 도움을 주는 다중 언어로 된 엄청나게 멋지고 주도적이며 실용적인 OCR 툴을 만드는데 목표를 두고 있습니다.

<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleOCR/releases/download/v2.8.0/demo.gif" width="800">
</div>

## 📣최근 업데이트

- **🔥2022년 8월 24일에 패들 OCR 출시 [출시/2.6](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6)**
  차이니즈 씬에 맞춘 완전 업그레이드 된 기능과 성능을 갖춘 ; [PP-Structurev2](../../ppstructure/) 출시, 그리고 레이아웃 [리커버리](../../ppstructure/recovery) 신규 지원 및 PDF 를 워드로 전환하는 원 라인 명령
    - [레이아웃 분석](../../ppstructure/layout)  최적화: 95% 감소된 모델 저장, while 반면 속도는 11배 증가하고, 평균 CPU 시간 비용은 41ms에 불과함;
    - [표 인식](../../ppstructure/table) 최적화: 3 최적화 전략이 디자인되고 모델 정확도는 비교 가능한 시간 소비 하에 6% 개선됨;
    - [핵심 정보 추출](../../ppstructure/kie)  최적화： 시각에 의존하지 않는 모델 구조가 디자인되고, 의미체 인식 정확도가 2.8% 증가되며 관계 추출 정확도는 9.1% 증가됨.

- **🔥2022년 7월 출시[OCR 씬 애플리케이션 컬렉션](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.8/applications/README_en.md)**
    디지털 튜브, LCD 스크린, 라이선스 플레이트, 수기 인식 모델, 고정밀 SVTR 모델 등등과 같은 “9수직 모델” 출시로, 일반적으로 주된 OCR 수직 애플리케이션, 제조, 금융 및 수송 산업 커버

- **🔥2022년 5월 9일에 패들 OCR 출시 [출시/2.5](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.5)**
    - [PP-OCRv3](../version2.x/ppocr/overview.en.md#pp-ocrv3)출시: 5%.비교 가능한 속도로, 차이니즈 씬의 효과는 PP-OCRv2와 비교해 볼 때 추가로 5% 정도 더 개선되고, 잉글리쉬 씬 효과는 11% 개선되었으며, 80개 언어 다중 언어 모델 평균 인식 정확도는 5% 이상 개선됨.
    - [PPOCRLabelv2](https://github.com/PFCCLab/PPOCRLabel)출시: 표 인식 업무, 핵심 정보 추출 업무 및 불규칙한 텍스트 이미지주석 기능 추가.
    - 쌍방향e-북 출시 [*"OCR 뛰어들기"*](../version2.x/ppocr/blog/ocr_book.en.md), 첨단 이론 및 OCR 정식 스택 기술 코드 연습 포함.

- [추가](../update/update.en.md)

## 🌟특징

패들OCR은 OCR 관련 다양한 첨단 알고리즘 지원  [PP-OCR](../version2.x/ppocr/overview.en.md) 및 [PP-Structure](../../ppstructure/README.md)  이를 기반으로, 그리고 전체 데이터 생산 처리, 모델 훈련, 압축, 추론 및 배치를 통해 획득.

<div align="center">
    <img src="https://user-images.githubusercontent.com/50011306/196963392-6cd1b251-109b-49c3-9b3d-ccf203dcec49.png">
</div>

## ⚡ 신속한 경험

```bash
pip3 install paddlepaddle # for gpu user please install paddlepaddle-gpu
pip3 install paddleocr
paddleocr --image_dir /your/test/image.jpg --lang=korean
```

>만일 당신이 파이톤 환경이 없다면 [환경 준비를 따르기 바람](../version2.x/ppocr/environment.en.md). 우리는 당신이[사용지침 프로그램]으로 시작할 것을 권장합니다.(#Tutorials).

<a name="북"></a>

## 📚 E-북: *OCR로 뛰어들기*

- [OCR로 뛰어들기](../version2.x/ppocr/blog/ocr_book.en.md)

<a name="커뮤니티"></a>

## 👫 커뮤니티로

국제 개발자들을 위해 우리는 [PaddleOCR 논의하기](https://github.com/PaddlePaddle/PaddleOCR/discussions) 를 우리의 국제 커뮤니티로 간주. 모든 아이디어와 질문은 여기서 영어로 논의 가능.

<a name="지원됨 – 차이니즈-모델-목록- "></a>

## PP-OCR 시리즈 모델 목록

| 모델 소개                                          |모델 명                   | 권장 씬| 감지 모델                                              |인식 모델                                            |
| ------------------------------------------------------------ | ---------------------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 한국어 초경량 PP-OCRv3 모델(14.8M) | korean_PP-OCRv3_xx | 모바일 & 서버 | [추론 모델](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar) / [훈련 모델](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_distill_train.tar) | [추론 모델](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar) / [훈련 모델](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_train.tar) |
| 영어 초경량 PP-OCRv3 모델（13.4M） | en_PP-OCRv3_xx | 모바일 & 서버 | [추론 모델](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) / [훈련 모델](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar) | [추론 모델](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv3_mobile_rec_infer.tar) / [훈련 모델](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar) |
| 중국어 및 영어 초경량 PP-OCRv3 model（16.2M）     | ch_PP-OCRv3_xx          | 모바일 & 서버 | [추론 모델](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_mobile_det_infer.tar) / [훈련 모델](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_det_pretrained.pdparams) | [추론 모델](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_mobile_rec_infer.tar) / [훈련 모델](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar) |

- (다중 언어를 포함하여)더 많은 모델을 다운로드 하려면, [PP-OCR 시리즈 모델 다운로드](../version2.x/ppocr/model_list.en.md)를 참조할 것.
- 신규 언어 요청에 대해서는, [신규 언어 요청 지침](#language_requests)을 참조할 것.
- 구조적 문서 분석 모델에 대해서는, [PP-Structure models](../version2.x/ppstructure/models_list.en.md).을 참조할 것.

<a name="사용 지침 프로그램"></a>

## 📖 사용 지침 프로그램

- [환경 준비](../version2.x/ppocr/environment.en.md)
- [PP-OCR 🔥](../version2.x/ppocr/overview.en.md)
    - [신속한 시작](../version2.x/ppocr/quick_start.en.md)
    - [동물원 모델](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.9/doc/doc_en/models_en.md)
    - [모델 훈련](../version2.x/ppocr/model_train/training.en.md)
        - [텍스트 감지](../version2.x/ppocr/model_train/detection.en.md)
        - [텍스트 인식](../version2.x/ppocr/model_train/recognition.en.md)
        - [텍스트 방향 분류](../version2.x/ppocr/model_train/angle_class.en.md)
    - 모델 압축
        - [모델 계량화](../../deploy/slim/quantization/README_en.md)
        - [모델 전지작업](../../deploy/slim/prune/README_en.md)
        - [지식 정제](../version2.x/ppocr/model_compress/knowledge_distillation.en.md)
    - [추론 및 배치](../../deploy/README.md)
        - [파이톤 추론](../version2.x/legacy/python_infer.en.md)
        - [C++ 추론](../version2.x/legacy/cpp_infer.en.md)
        - [서빙](../version2.x/legacy/paddle_server.en.md)
        - [모바일](../../deploy/lite/readme.md)
        - [Paddle2ONNX](../../deploy/paddle2onnx/readme.md)
        - [패들 클라우드](../../deploy/paddlecloud/README.md)
        - [Benchmark](../version2.x/legacy/benchmark.en.md)
- [PP-Structure 🔥](../../ppstructure/README.md)
    - [신속한 시작](../version2.x/ppstructure/quick_start.en.md)
    - [동물원 모델](../version2.x/ppstructure/models_list.en.md)
    - [모델 훈련](../version2.x/ppocr/model_train/training.en.md)
        - [레이아웃 분석](../../ppstructure/layout/README.md)
        - [표 인식](../../ppstructure/table/README.md)
        - [핵심 정보 추출](../../ppstructure/kie/README.md)
    - [추론 및 배치](../../deploy/README.md)
        - [파이톤 추론](../version2.x/ppstructure/infer_deploy/python_infer.en.md)
        - [C++ 추론](../version2.x/legacy/cpp_infer.en.md)
        - [서빙](../../deploy/hubserving/readme_en.md)
- [학문적 알고리즘](../version2.x/algorithm/overview.en.md)
    - [텍스트 감지](../version2.x/algorithm/overview.en.md)
    - [텍스트 인식](../version2.x/algorithm/overview.en.md)
    - [종단종OCR](../version2.x/algorithm/overview.en.md)
    - [표 인식](../version2.x/algorithm/overview.en.md)
    - [핵심 정보 추출](../version2.x/algorithm/overview.en.md)
    - [PaddleOCR에 신규 알고리즘 추가](../version2.x/algorithm/add_new_algorithm.en.md)
- 데이터 주석 및 합성
    - [반-자동 주석 툴: PPOCRLabel](https://github.com/PFCCLab/PPOCRLabel/blob/main/README.md)
    - [데이터 합성 툴: 스타일-텍스트](https://github.com/PFCCLab/StyleText/blob/main/README.md)
    - [기타 데이터 주석 툴](../data_anno_synth/data_annotation.en.md)
    - [기타 데이터 합성 툴](../data_anno_synth/data_synthesis.en.md)
- 데이터세트
    - [일반 OCR 데이터세트(중국어/영어)](../datasets/datasets.en.md)
    - [수기_OCR_데이터세트(중국어)](../datasets/handwritten_datasets.en.md)
    - [다양한 OCR 데이터세트(다중언어)](../datasets/vertical_and_multilingual_datasets.en.md)
    - [레이아웃 분석](../datasets/layout_datasets.en.md)
    - [표 인식](../datasets/table_datasets.en.md)
    - [핵심 정보 추출](../datasets/kie_datasets.en.md)
- [코드 구조](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.9/doc/doc_en/tree_en.md)
- [시각화](#Visualization)
- [커뮤니티](#Community)
- [신규 언어 요청](#language_requests)
- [자주 묻는 질문](../FAQ.en.md)
- [추론](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.9/doc/doc_en/reference_en.md)
- [라이선스](#LICENSE)

<a name="language_requests"></a>

## 신규 언어 요청에 대한 유엔 가이드라인

만일 신규 언어 모델을 요청하고자 한다면**, [다중 언어 모델 업그레이드 투표하기](https://github.com/PaddlePaddle/PaddleOCR/discussions/7253)에서 투표하기 바람. 우리는 결과에 따라 규칙적으로 모델을 업그레이드 시킬 것임**함께 투표하고자 당신의 친구들을 초대할 것!**
만일 당신이 시나리오 기반 “신규 언어 모델”을 훈련하고자 한다면, [다중 언어 모델 훈련 프로젝트](https://github.com/PaddlePaddle/PaddleOCR/discussions/7252) 를 통해 당신의 데이터세트를 작성하는데 도움이 되고 단계별로 전체 절차를 보여줄 것입니다.
원본[다중 언어 OCR 개발 계획](https://github.com/PaddlePaddle/PaddleOCR/issues/1048)은 여전히 수많은 유용한 말뭉치와 사전을 보여줍니다.

<a name="시각화"></a>

## 👀 시각화[추가](../version2.x/ppocr/visualization.en.md)

<details open>
<summary>PP-OCRv3 다중 언어 모델</summary>
<div align="center">
    <img src="../version2.x/ppocr/images/PP-OCRv3/multi_lang/japan_2.jpg" width="800">
    <img src="../version2.x/ppocr/images/PP-OCRv3/multi_lang/korean_1.jpg" width="800">
</div>
</details>

<details open>
<summary>PP-OCRv3 영어 모델</summary>
<div align="center">
    <img src="../version2.x/ppocr/images/PP-OCRv3/en/en_1.png" width="800">
    <img src="../version2.x/ppocr/images/PP-OCRv3/en/en_2.png" width="800">
</div>
</details>
<details open>
<summary>PP-OCRv3 중국어 모델</summary>
<div align="center">
    <img src="../version2.x/ppocr/images/PP-OCRv3/ch/PP-OCRv3-pic001.jpg" width="800">
    <img src="../version2.x/ppocr/images/PP-OCRv3/ch/PP-OCRv3-pic002.jpg" width="800">
    <img src="../version2.x/ppocr/images/PP-OCRv3/ch/PP-OCRv3-pic003.jpg" width="800">
</div>
</details>

<details open>
<summary>PP-Structurev2</summary>
1.  레이아웃 분석 + 표 인식
<div align="center">
    <img src="../version2.x/ppstructure/images/ppstructure.gif" width="800">
</div>
2. SER (의미체 인식)
<div align="center">
    <img src="https://user-images.githubusercontent.com/25809855/186094456-01a1dd11-1433-4437-9ab2-6480ac94ec0a.png" width="600">
</div>
<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185310636-6ce02f7c-790d-479f-b163-ea97a5a04808.jpg" width="600">
</div>
<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185539517-ccf2372a-f026-4a7c-ad28-c741c770f60a.png" width="600">
</div>
3. RE (관계 추출)
<div align="center">
    <img src="https://user-images.githubusercontent.com/25809855/186094813-3a8e16cc-42e5-4982-b9f4-0134dfb5688d.png" width="600">
</div>
<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185393805-c67ff571-cf7e-4217-a4b0-8b396c4f22bb.jpg" width="600">
</div>
<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185540080-0431e006-9235-4b6d-b63d-0b3c6e1de48f.jpg" width="600">
</div>
</details>
<a name="라이선스"></a>

## 📄 라이선스

본 프로젝트는 <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/{{PADDLEOCR_GITHUB_REF}}/LICENSE">Apache 2.0 license</a> 하에 출시됨.
