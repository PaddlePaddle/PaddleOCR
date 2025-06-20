<div align="center">
  <p>
      <img width="100%" src="./docs/images/Banner.png" alt="PaddleOCR 배너">
  </p>

<!-- language -->
[English](./README.md) | [简体中文](./README_cn.md) | [繁體中文](./README_tcn.md) | [日本語](./README_ja.md) | 한국어 | [Français](./README_fr.md) | [Русский](./README_ru.md) | [Español](./README_es.md) | [العربية](./README_ar.md)

<!-- icon -->

[![stars](https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf)](https://github.com/PaddlePaddle/PaddleOCR)
[![Downloads](https://img.shields.io/pypi/dm/paddleocr)](https://pypi.org/project/PaddleOCR/)
![python](https://img.shields.io/badge/python-3.8~3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
![hardware](https://img.shields.io/badge/hardware-cpu%2C%20gpu%2C%20xpu%2C%20npu-yellow.svg)


[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

</div>

## 🚀 소개
PaddleOCR은 출시 이후 최첨단 알고리즘(algorithm)과 실제 애플리케이션(application)에서의 입증된 성능 덕분에 학계, 산업계, 연구 커뮤니티에서 폭넓은 찬사를 받아왔습니다. Umi-OCR, OmniParser, MinerU, RAGFlow와 같은 유명 오픈소스 프로젝트에 이미 적용되어 전 세계 개발자(developer)들에게 필수 OCR 툴킷(toolkit)으로 자리 잡았습니다.

2025년 5월 20일, PaddlePaddle 팀은 **PaddlePaddle 3.0** 프레임워크의 공식 릴리스와 완전히 호환되는 PaddleOCR 3.0을 발표했습니다. 이 업데이트는 **텍스트 인식 정확도를 더욱 향상**시키고, **다중 텍스트 유형 인식** 및 **필기 인식**을 지원하며, 대규모 모델 애플리케이션의 **복잡한 문서의 고정밀 구문 분석**에 대한 증가하는 수요를 충족합니다. **ERNIE 4.5 Turbo**와 결합하면 주요 정보 추출 정확도가 크게 향상됩니다. 사용 설명서 전체는 [PaddleOCR 3.0 문서](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html)를 참조하십시오.

PaddleOCR 3.0의 세 가지 주요 신규 기능:
- 범용 장면 텍스트 인식 모델(Universal-Scene Text Recognition Model) [PP-OCRv5](./docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.en.md): 다섯 가지 다른 텍스트 유형과 복잡한 필기체를 처리하는 단일 모델입니다. 전체 인식 정확도는 이전 세대보다 13%p 향상되었습니다. [온라인 체험](https://aistudio.baidu.com/community/app/91660/webUI)

- 일반 문서 파싱(parsing) 솔루션 [PP-StructureV3](./docs/version3.x/algorithm/PP-StructureV3/PP-StructureV3.en.md): 다중 레이아웃(multi-layout), 다중 장면 PDF의 고정밀 파싱(parsing)을 제공하며, 공개 벤치마크(benchmark)에서 많은 오픈 소스 및 클로즈드 소스 솔루션을 능가합니다. [온라인 체험](https://aistudio.baidu.com/community/app/518494/webUI)

- 지능형 문서 이해 솔루션 [PP-ChatOCRv4](./docs/version3.x/algorithm/PP-ChatOCRv4/PP-ChatOCRv4.en.md): ERNIE 4.5 Turbo에 의해 네이티브로 구동되며, 이전 모델보다 15%p 높은 정확도를 달성합니다. [온라인 체험](https://aistudio.baidu.com/community/app/518493/webUI)

PaddleOCR 3.0은 뛰어난 모델 라이브러리(model library)를 제공할 뿐만 아니라 모델 훈련, 추론 및 서비스 배포를 포괄하는 사용하기 쉬운 도구를 제공하여 개발자가 AI 애플리케이션을 신속하게 상용화할 수 있도록 지원합니다.
<div align="center">
  <p>
      <img width="100%" src="./docs/images/Arch.png" alt="PaddleOCR 아키텍처">
  </p>
</div>



## 📣 최신 업데이트

#### **🔥🔥 2025.06.05: PaddleOCR 3.0.1 릴리스, 포함 내용:**

- **일부 모델 및 모델 구성 최적화:**
  - PP-OCRv5의 기본 모델 구성을 업데이트하여 탐지 및 인식을 모두 mobile에서 server 모델로 변경했습니다. 대부분의 시나리오에서 기본 성능을 향상시키기 위해 구성의 `limit_side_len` 파라미터(parameter)가 736에서 64로 변경되었습니다.
  - 99.42%의 정확도를 가진 새로운 텍스트 라인 방향 분류 모델 `PP-LCNet_x1_0_textline_ori`를 추가했습니다. OCR, PP-StructureV3, PP-ChatOCRv4 파이프라인의 기본 텍스트 라인 방향 분류기가 이 모델로 업데이트되었습니다.
  - 텍스트 라인 방향 분류 모델 `PP-LCNet_x0_25_textline_ori`를 최적화하여 정확도를 3.3%p 향상시켜 현재 정확도는 98.85%입니다.

- **버전 3.0.0의 일부 문제점에 대한 최적화 및 수정, [상세 정보](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)**

🔥🔥2025.05.20: **PaddleOCR v3.0** 정식 출시, 포함 내용:
- **PP-OCRv5**: 모든 시나리오를 위한 고정밀 텍스트 인식 모델 - 이미지/PDF에서 즉시 텍스트 추출.
   1. 🌐 단일 모델로 **다섯 가지** 텍스트 유형 지원 - **중국어 간체, 중국어 번체, 중국어 간체 병음, 영어, 일본어**를 단일 모델 내에서 원활하게 처리합니다.
   2. ✍️ 향상된 **필기체 인식**: 복잡한 흘림체 및 비표준 필기체에서 성능이 크게 향상되었습니다.
   3. 🎯 PP-OCRv4에 비해 **정확도 13%p 향상**, 다양한 실제 시나리오에서 SOTA(state-of-the-art) 성능을 달성했습니다.

- **PP-StructureV3**: 범용 문서 파싱(parsing) – 실제 시나리오를 위한 SOTA 이미지/PDF 파싱(parsing) 성능!
   1. 🧮 **고정밀 다중 장면 PDF 파싱(parsing)**, OmniDocBench 벤치마크(benchmark)에서 오픈 소스 및 클로즈드 소스 솔루션을 모두 능가합니다.
   2. 🧠 전문 기능에는 **도장 인식**, **차트-표 변환**, **중첩된 수식/이미지가 있는 표 인식**, **세로 텍스트 문서 파싱(parsing)**, **복잡한 표 구조 분석** 등이 포함됩니다.

- **PP-ChatOCRv4**: 지능형 문서 이해 – 이미지/PDF에서 단순한 텍스트가 아닌 핵심 정보 추출.
   1. 🔥 이전 세대에 비해 PDF/PNG/JPG 파일의 핵심 정보 추출에서 **정확도 15%p 향상**.
   2. 💻 **ERNIE 4.5 Turbo** 기본 지원, PaddleNLP, Ollama, vLLM 등을 통한 대규모 모델 배포와 호환됩니다.
   3. 🤝 [PP-DocBee2](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee2)와 통합되어 인쇄된 텍스트, 필기체, 도장, 표, 차트 등 복잡한 문서의 일반적인 요소 추출 및 이해를 지원합니다.

<details>
   <summary><strong>업데이트 내역</strong></summary>

- 🔥🔥2025.03.07: **PaddleOCR v2.10** 릴리스, 포함 내용:

  - **12개의 자체 개발 신규 모델:**
    - **[레이아웃 탐지 시리즈](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html)**(3개 모델): PP-DocLayout-L, M, S -- 영어와 중국어로 된 다양한 문서 형식(논문, 보고서, 시험, 책, 잡지, 계약서 등)에서 23개의 공통 레이아웃 유형을 탐지할 수 있습니다. 최대 **90.4% mAP@0.5**를 달성하고, 경량 기능으로 초당 100페이지 이상을 처리할 수 있습니다.
    - **[수식 인식 시리즈](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/formula_recognition.html)**(2개 모델): PP-FormulaNet-L, S -- 50,000개 이상의 LaTeX 표현식 인식을 지원하며 인쇄 및 필기 수식을 모두 처리합니다. PP-FormulaNet-L은 유사 모델보다 **6% 더 높은 정확도**를 제공하며, PP-FormulaNet-S는 유사한 정확도를 유지하면서 16배 더 빠릅니다.
    - **[표 구조 인식 시리즈](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_structure_recognition.html)**(2개 모델): SLANeXt_wired 및 SLANeXt_wireless -- 복잡한 표 인식에서 SLANet_plus보다 **6% 정확도가 향상**된 새로 개발된 모델입니다.
    - **[표 분류](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_classification.html)**(1개 모델): PP-LCNet_x1_0_table_cls -- 유선 및 무선 표를 위한 초경량 분류기입니다.

[더 알아보기](https://paddlepaddle.github.io/PaddleOCR/latest/en/update.html)

</details>

## ⚡ 빠른 시작
### 1. 온라인 데모 실행
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

### 2. 설치

[설치 가이드](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)를 참조하여 PaddlePaddle을 설치한 후, PaddleOCR 툴킷을 설치하십시오.

```bash
# paddleocr 설치
pip install paddleocr
```

### 3. CLI를 통한 추론 실행
```bash
# PP-OCRv5 추론 실행
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False  

# PP-StructureV3 추론 실행
paddleocr pp_structurev3 -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png --use_doc_orientation_classify False --use_doc_unwarping False

# 먼저 Qianfan API 키를 받고, PP-ChatOCRv4 추론 실행
paddleocr pp_chatocrv4_doc -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png -k 驾驶室准乘人数 --qianfan_api_key your_api_key --use_doc_orientation_classify False --use_doc_unwarping False 

# "paddleocr ocr"에 대한 추가 정보 얻기
paddleocr ocr --help
```

### 4. API를 통한 추론 실행
**4.1 PP-OCRv5 예제**
```python
from paddleocr import PaddleOCR
# PaddleOCR 인스턴스 초기화
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# 샘플 이미지에 대해 OCR 추론 실행
result = ocr.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png")

# 결과 시각화 및 JSON 결과 저장
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

<details>
    <summary><strong>4.2 PP-StructureV3 예제</strong></summary>

```python
from pathlib import Path
from paddleocr import PPStructureV3

pipeline = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
)

# 이미지용
output = pipeline.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png",
    )

# 결과 시각화 및 JSON 결과 저장
for res in output:
    res.print() 
    res.save_to_json(save_path="output") 
    res.save_to_markdown(save_path="output")           
```

</details>

<details>
   <summary><strong>4.3 PP-ChatOCRv4 예제</strong></summary>

```python
from paddleocr import PPChatOCRv4Doc

chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "ernie-3.5-8k",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key",  # your api_key
}

retriever_config = {
    "module_name": "retriever",
    "model_name": "embedding-v1",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "qianfan",
    "api_key": "api_key",  # your api_key
}

pipeline = PPChatOCRv4Doc(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
)

visual_predict_res = pipeline.visual_predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png",
    use_common_ocr=True,
    use_seal_recognition=True,
    use_table_recognition=True,
)

mllm_predict_info = None
use_mllm = False
# 다중 모드 대형 모델을 사용하는 경우 로컬 mllm 서비스를 시작해야 합니다. 문서: https://github.com/PaddlePaddle/PaddleX/blob/release/3.0/docs/pipeline_usage/tutorials/vlm_pipelines/doc_understanding.en.md를 참조하여 배포하고 mllm_chat_bot_config 구성을 업데이트할 수 있습니다.
if use_mllm:
    mllm_chat_bot_config = {
        "module_name": "chat_bot",
        "model_name": "PP-DocBee",
        "base_url": "http://127.0.0.1:8080/",  # your local mllm service url
        "api_type": "openai",
        "api_key": "api_key",  # your api_key
    }

    mllm_predict_res = pipeline.mllm_pred(
        input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png",
        key_list=["驾驶室准乘人数"],
        mllm_chat_bot_config=mllm_chat_bot_config,
    )
    mllm_predict_info = mllm_predict_res["mllm_res"]

visual_info_list = []
for res in visual_predict_res:
    visual_info_list.append(res["visual_info"])
    layout_parsing_result = res["layout_parsing_result"]

vector_info = pipeline.build_vector(
    visual_info_list, flag_save_bytes_vector=True, retriever_config=retriever_config
)
chat_result = pipeline.chat(
    key_list=["驾驶室准乘人数"],
    visual_info=visual_info_list,
    vector_info=vector_info,
    mllm_predict_info=mllm_predict_info,
    chat_bot_config=chat_bot_config,
    retriever_config=retriever_config,
)
print(chat_result)
```

</details>

## ⛰️ 고급 튜토리얼
- [PP-OCRv5 튜토리얼](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/OCR.html)
- [PP-StructureV3 튜토리얼](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-StructureV3.html)
- [PP-ChatOCRv4 튜토리얼](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-ChatOCRv4.html)

## 🔄 실행 결과 빠른 개요

<div align="center">
  <p>
     <img width="100%" src="./docs/images/demo.gif" alt="PP-OCRv5 데모">
  </p>
</div>

<div align="center">
  <p>
     <img width="100%" src="./docs/images/blue_v3.gif" alt="PP-StructureV3 데모">
  </p>
</div>

## 👩‍👩‍👧‍👦 커뮤니티

| PaddlePaddle 위챗(WeChat) 공식 계정 | 기술 토론 그룹 가입 |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |


## 🏆 PaddleOCR을 활용하는 우수 프로젝트
PaddleOCR의 발전은 커뮤니티 없이는 불가능합니다! 💗 오랜 파트너, 새로운 협력자, 그리고 이름을 언급했든 안 했든 PaddleOCR에 열정을 쏟아부은 모든 분들께 진심으로 감사드립니다. 여러분의 지원이 우리의 원동력입니다!

| 프로젝트 이름 | 설명 |
| ------------ | ----------- |
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|심층 문서 이해 기반의 RAG 엔진.|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|다중 유형 문서를 마크다운(Markdown)으로 변환하는 도구|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|무료, 오픈 소스, 배치 오프라인 OCR 소프트웨어.|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |순수 비전 기반 GUI 에이전트를 위한 화면 파싱(parsing) 도구.|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |무엇이든 기반으로 한 질의응답 시스템.|
| [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) <a href="https://github.com/opendatalab/PDF-Extract-Kit"><img src="https://img.shields.io/github/stars/opendatalab/PDF-Extract-Kit"></a>|복잡하고 다양한 PDF 문서에서 고품질 콘텐츠를 효율적으로 추출하도록 설계된 강력한 오픈 소스 툴킷.|
| [Dango-Translator](https://github.com/PantsuDango/Dango-Translator)<a href="https://github.com/PantsuDango/Dango-Translator"><img src="https://img.shields.io/github/stars/PantsuDango/Dango-Translator"></a> |화면의 텍스트를 인식하여 번역하고 번역 결과를 실시간으로 표시합니다.|
| [Learn more projects](./awesome_projects.md) | [More projects based on PaddleOCR](./awesome_projects.md)|

## 👩‍👩‍👧‍👦 기여자

<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>


## 🌟 Star

[![Star History Chart](https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date)](https://star-history.com/#PaddlePaddle/PaddleOCR&Date)


## 📄 라이선스
이 프로젝트는 [Apache 2.0 license](LICENSE)에 따라 배포됩니다.

## 🎓 인용

```
@misc{paddleocr2020,
title={PaddleOCR, Awesome multilingual OCR toolkits based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleOCR}},
year={2020}
}
```
