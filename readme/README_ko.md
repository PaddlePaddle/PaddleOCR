<div align="center">
  <p>
      <img width="100%" src="../docs/images/Banner.png" alt="PaddleOCR 배너">
  </p>

<!-- language -->
[English](../README.md) | [简体中文](./README_cn.md) | [繁體中文](./README_tcn.md) | [日本語](./README_ja.md) | 한국어 | [Français](./README_fr.md) | [Русский](./README_ru.md) | [Español](./README_es.md) | [العربية](./README_ar.md)

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

2025년 5월 20일, PaddlePaddle 팀은 **PaddlePaddle 3.0** 프레임워크의 공식 릴리스와 완전히 호환되는 PaddleOCR 3.0을 발표했습니다. 이 업데이트는 **텍스트 인식 정확도를 더욱 향상**시키고, **다중 텍스트 유형 인식** 및 **필기 인식**을 지원하며, 대규모 모델 애플리케이션의 **복잡한 문서의 고정밀 구문 분석**에 대한 증가하는 수요를 충족합니다. **ERNIE 4.5**와 결합하면 주요 정보 추출 정확도가 크게 향상됩니다. 사용 설명서 전체는 [PaddleOCR 3.0 문서](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html)를 참조하십시오.

PaddleOCR 3.0의 세 가지 주요 신규 기능:
- 범용 장면 텍스트 인식 모델(Universal-Scene Text Recognition Model) [PP-OCRv5](../docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.en.md): 다섯 가지 다른 텍스트 유형과 복잡한 필기체를 처리하는 단일 모델입니다. 전체 인식 정확도는 이전 세대보다 13%p 향상되었습니다. [온라인 체험](https://aistudio.baidu.com/community/app/91660/webUI)

- 일반 문서 파싱(parsing) 솔루션 [PP-StructureV3](../docs/version3.x/algorithm/PP-StructureV3/PP-StructureV3.en.md): 다중 레이아웃(multi-layout), 다중 장면 PDF의 고정밀 파싱(parsing)을 제공하며, 공개 벤치마크(benchmark)에서 많은 오픈 소스 및 클로즈드 소스 솔루션을 능가합니다. [온라인 체험](https://aistudio.baidu.com/community/app/518494/webUI)

- 지능형 문서 이해 솔루션 [PP-ChatOCRv4](../docs/version3.x/algorithm/PP-ChatOCRv4/PP-ChatOCRv4.en.md): ERNIE 4.5에 의해 네이티브로 구동되며, 이전 모델보다 15%p 높은 정확도를 달성합니다. [온라인 체험](https://aistudio.baidu.com/community/app/518493/webUI)

PaddleOCR 3.0은 뛰어난 모델 라이브러리(model library)를 제공할 뿐만 아니라 모델 훈련, 추론 및 서비스 배포를 포괄하는 사용하기 쉬운 도구를 제공하여 개발자가 AI 애플리케이션을 신속하게 상용화할 수 있도록 지원합니다.
<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Arch.jpg" alt="PaddleOCR Architecture">
  </p>
</div>

**특별 참고**: PaddleOCR 3.x는 여러 가지 중요한 인터페이스 변경 사항을 도입했습니다. **PaddleOCR 2.x를 기반으로 작성된 기존 코드는 PaddleOCR 3.x와 호환되지 않을 수 있습니다**. 사용 중인 PaddleOCR 버전에 맞는 문서를 참고하고 있는지 확인해 주세요. [이 문서](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/upgrade_notes.html)에서는 업그레이드 이유와 PaddleOCR 2.x에서 3.x로의 주요 변경 사항을 설명하고 있습니다.

## 📣 최신 업데이트

#### **🔥🔥2025.08.21: PaddleOCR 3.2.0 출시**, 주요 내용:

- **주요 모델 추가:**
    - PP-OCRv5 영어, 태국어, 그리스어 인식 모델의 학습, 추론, 배포 기능 추가. **PP-OCRv5 영어 모델은 영어 환경에서 기존 PP-OCRv5 모델 대비 11% 성능 향상이 있었으며, 태국어 인식 모델 정확도는 82.68%, 그리스어 인식 모델 정확도는 89.28%에 달함.**

- **배포 기능 업그레이드:**
    - **PaddlePaddle 프레임워크 3.1.0 및 3.1.1 완전 지원.**
    - **PP-OCRv5 C++ 로컬 배포 솔루션 전체 업그레이드, Linux 및 Windows 지원, Python 솔루션과 동일한 기능과 정확도 제공.**
    - **고성능 추론을 위해 CUDA 12 지원, Paddle Inference 및 ONNX Runtime 백엔드 사용 가능.**
    - **고안정성 서비스형 배포 솔루션 전체 오픈소스화, 사용자가 Docker 이미지 및 SDK를 필요에 따라 커스터마이징 가능.**
    - 고안정성 서비스형 배포 솔루션은 수동 HTTP 요청을 통한 호출도 지원, 클라이언트 프로그램은 임의의 언어로 개발 가능.

- **벤치마크 지원:**
    - **모든 프로덕션 환경에서 세분화된 벤치마크 지원, 엔드투엔드 추론 시간 및 계층별·모듈별 소요 시간 측정 가능, 성능 분석에 도움.[여기](../docs/version3.x/pipeline_usage/instructions/benchmark.en.md)는 벤치마크 기능의 설정 및 사용 방법입니다**
    - **문서에 주요 하드웨어에서의 일반적인 설정 기준 지표(추론 시간, 메모리 점유 등) 추가, 사용자 배포에 참고 정보 제공.**

- **버그 수정:**
    - 모델 학습 시 학습 로그가 저장되지 않는 문제 수정.
    - 수식 모델의 데이터 증강 부분을 새 albumentations 버전에 맞게 호환성 업그레이드, tokenizers 패키지 다중 프로세스 사용 시 데드락 경고 문제 해결.
    - PP-StructureV3 설정 파일에서 `use_chart_parsing` 등 스위치 동작이 다른 프로덕션과 일치하지 않는 문제 수정.

- **기타 업그레이드:**
    - **필수 종속성과 선택 종속성 구분, 기본 문자 인식 기능에는 최소 핵심 종속성만 설치 필요, 문서 파싱이나 정보 추출 등 추가 기능은 필요에 따라 종속성 추가 설치 가능.**
    - **Windows 사용자의 NVIDIA 50 시리즈 그래픽카드 지원, [설치 문서](../docs/version3.x/installation.en.md) 참조하여 Paddle 프레임워크 버전 선택 가능.**
    - **PP-OCR 시리즈 모델, 글자별 좌표 반환 가능.**
    - 모델 다운로드 소스로 AIStudio, ModelScope 등 추가, 원하는 소스 지정 가능.
    - 차트-테이블 변환 모듈(PP-Chart2Table) 추론 기능 지원.
    - 일부 문서 설명 최적화로 사용성 향상.


#### **2025.08.15: PaddleOCR 3.1.1 출시**, 주요 내용:

- **버그 수정:**
  - `PP-ChatOCRv4` 클래스에 누락된 `save_vector`, `save_visual_info_list`, `load_vector`, `load_visual_info_list` 메서드 추가.
  - `PPDocTranslation` 클래스의 `translate` 메서드에 누락된 `glossary`와 `llm_request_interval` 파라미터 추가.

- **문서 최적화:**
  - MCP 문서에 데모(demo) 추가.
  - 성능 지표 테스트에 사용된 PaddlePaddle 프레임워크 및 PaddleOCR 버전 명시 추가.
  - 문서 번역 생산 라인 문서의 오류 및 누락 수정.

- **기타:**
  - MCP 서버 의존성 변경: 설치 문제를 줄이기 위해 `python-magic` 대신 pure Python 라이브러리인 `puremagic` 사용.
  - PaddleOCR 3.1.0 버전으로 PP-OCRv5 성능 지표를 재테스트하고 문서 업데이트.

#### **2025.06.29: PaddleOCR 3.1.0 출시**, 주요 내용:

- **주요 모델 및 파이프라인:**
  - **PP-OCRv5 다국어 텍스트 인식 모델 추가**, 프랑스어, 스페인어, 포르투갈어, 러시아어, 한국어 등 37개 언어의 텍스트 인식 모델 학습 및 추론 지원. **평균 정확도 30% 이상 향상.** [자세히 보기](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)
  - PP-StructureV3의 **PP-Chart2Table 모델 업그레이드**, 차트에서 표로 변환하는 기능이 더욱 향상됨. 내부 커스텀 평가 세트에서 지표(RMS-F1)가 **9.36%p 상승(71.24% → 80.60%)**.
  - PP-StructureV3 및 ERNIE 4.5 기반 **문서 번역 파이프라인 PP-DocTranslation 신규 출시**, Markdown 형식 문서, 다양한 복잡 레이아웃의 PDF 문서, 문서 이미지를 번역 지원, 결과는 Markdown 형식으로 저장 가능. [자세히 보기](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/PP-DocTranslation.html)

- **새로운 MCP 서버:** [Details](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/mcp_server.html)
  - **OCR 및 PP-StructureV3 파이프라인을 모두 지원합니다.**
  - 로컬 Python 라이브러리, AIStudio 커뮤니티 클라우드 서비스, 자체 호스팅 서비스의 세 가지 작업 모드를 지원합니다.
  - stdio를 통해 로컬 서비스를 호출하고, Streamable HTTP를 통해 원격 서비스를 호출할 수 있습니다.

- **문서 최적화:** 일부 사용자 가이드 설명 개선으로 읽기 경험 향상.

#### **2025.06.26: PaddleOCR 3.0.3 릴리스, 포함 내용:**

- 버그 수정: `enable_mkldnn` 매개변수가 작동하지 않는 문제를 해결하고, CPU가 기본적으로 MKL-DNN 추론을 사용하는 동작을 복원했습니다.

#### **🔥🔥 2025.06.19: PaddleOCR 3.0.2 릴리스, 포함 내용:**

- **새로운 기능:**
  - 모델 기본 다운로드 소스가 `BOS`에서 `HuggingFace`로 변경되었습니다. 사용자는 환경 변수 `PADDLE_PDX_MODEL_SOURCE`를 `BOS`로 설정하여 모델 다운로드 소스를 Baidu Object Storage(BOS)로 되돌릴 수 있습니다.
  - PP-OCRv5, PP-StructureV3, PP-ChatOCRv4 파이프라인에 대해 C++, Java, Go, C#, Node.js, PHP 6개 언어의 서비스 호출 예제가 추가되었습니다.
  - PP-StructureV3 파이프라인의 레이아웃 파티션 정렬 알고리즘을 개선하여 복잡한 세로 레이아웃의 정렬 논리를 향상했습니다.
  - 언어(`lang`)만 지정하고 모델 버전을 명시하지 않은 경우, 해당 언어를 지원하는 최신 모델 버전을 자동으로 선택하도록 모델 선택 로직을 강화했습니다.
  - MKL-DNN 캐시 크기에 기본 상한을 설정하여 무한 확장을 방지하고, 사용자 정의 캐시 용량 설정을 지원합니다.
  - 고성능 추론의 기본 구성을 업데이트하여 Paddle MKL-DNN 가속을 지원하고, 자동 구성 선택 로직을 최적화했습니다.
  - 설치된 Paddle 프레임워크가 지원하는 실제 디바이스를 고려하도록 기본 디바이스 선택 로직을 조정했습니다.
  - PP-OCRv5의 Android 예제가 추가되었습니다. [Details](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/on_device_deployment.html).

- **버그 수정:**
  - PP-StructureV3 일부 CLI 파라미터가 적용되지 않던 문제를 수정했습니다.
  - `export_paddlex_config_to_yaml`가 특정 상황에서 정상 동작하지 않던 문제를 해결했습니다.
  - `save_path`의 실제 동작과 문서 설명이 일치하지 않던 문제를 수정했습니다.
  - 기본 서비스화 배포에서 MKL-DNN을 사용할 때 발생할 수 있는 다중 스레딩 오류를 수정했습니다.
  - Latex-OCR 모델의 이미지 전처리 과정에서 채널 순서 오류를 수정했습니다.
  - 텍스트 인식 모듈에서 시각화 이미지를 저장할 때 발생하던 채널 순서 오류를 수정했습니다.
  - PP-StructureV3 파이프라인의 표 시각화 결과에 발생하던 채널 순서 오류를 수정했습니다.
  - PP-StructureV3 파이프라인에서 특수한 상황에서 `overlap_ratio` 계산 시 발생하던 오버플로 문제를 수정했습니다.

- **문서 개선:**
  - 문서의 `enable_mkldnn` 파라미터 설명을 프로그램의 실제 동작에 맞게 업데이트했습니다.
  - `lang` 및 `ocr_version` 파라미터에 대한 문서 오류를 수정했습니다.
  - CLI를 통해 생산 라인 설정 파일을 내보내는 방법을 문서에 추가했습니다.
  - PP-OCRv5 성능 데이터 표에서 누락된 열을 복원했습니다.
  - 다양한 구성에서 PP-StructureV3의 벤치마크 지표를 개선했습니다.

- **기타:**
  - numpy, pandas 등 의존성 버전 제한을 완화하여 Python 3.12 지원을 복원했습니다.

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
   2. 💻 **ERNIE 4.5** 기본 지원, PaddleNLP, Ollama, vLLM 등을 통한 대규모 모델 배포와 호환됩니다.
   3. 🤝 [PP-DocBee2](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee2)와 통합되어 인쇄된 텍스트, 필기체, 도장, 표, 차트 등 복잡한 문서의 일반적인 요소 추출 및 이해를 지원합니다.

[히스토리 로그](https://paddlepaddle.github.io/PaddleOCR/latest/en/update.html)

</details>

## ⚡ 빠른 시작
### 1. 온라인 데모 실행
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

### 2. 설치

[설치 가이드](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)를 참조하여 PaddlePaddle을 설치한 후, PaddleOCR 툴킷을 설치하십시오.

```bash
# 기본 텍스트 인식 기능(텍스트 위치 좌표와 내용을 반환)만 사용하려면(PP-OCR 시리즈 포함)
python -m pip install paddleocr
# 문서 분석, 문서 이해, 문서 번역, 주요 정보 추출 등 모든 기능을 사용하려면
# python -m pip install "paddleocr[all]"
```

버전 3.2.0부터 위의 `all` 의존성 그룹 외에도, PaddleOCR는 다른 의존성 그룹을 지정하여 일부 선택적 기능의 설치도 지원합니다. PaddleOCR에서 제공하는 모든 의존성 그룹은 아래와 같습니다.

| 의존성 그룹 이름 | 해당 기능 |
| - | - |
| `doc-parser` | 문서 분석: 표, 수식, 도장, 이미지 등 문서에서 레이아웃 요소를 추출할 수 있습니다. PP-StructureV3 등 모델 포함 |
| `ie` | 정보 추출: 이름, 날짜, 주소, 금액 등 문서에서 주요 정보를 추출할 수 있습니다. PP-ChatOCRv4 등 모델 포함 |
| `trans` | 문서 번역: 문서를 다른 언어로 번역할 수 있습니다. PP-DocTranslation 등 모델 포함 |
| `all` | 전체 기능 |

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

## 🧩 추가 기능

- 모델을 ONNX 형식으로 변환: [ONNX 모델 얻기](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/obtaining_onnx_models.html)
- OpenVINO, ONNX Runtime, TensorRT 같은 엔진을 사용하거나 ONNX 형식 모델로 추론 가속: [고성능 추론](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/high_performance_inference.html)
- 다중 GPU 및 다중 프로세스를 사용한 추론 가속: [파이프라인 병렬 추론](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/instructions/parallel_inference.html)
- C++, C#, Java 등으로 작성된 애플리케이션에 PaddleOCR 통합: [서빙](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/serving.html).

## ⛰️ 고급 튜토리얼
- [PP-OCRv5 튜토리얼](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/OCR.html)
- [PP-StructureV3 튜토리얼](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-StructureV3.html)
- [PP-ChatOCRv4 튜토리얼](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-ChatOCRv4.html)

## 🔄 실행 결과 빠른 개요

<div align="center">
  <p>
     <img width="100%" src="../docs/images/demo.gif" alt="PP-OCRv5 데모">
  </p>
</div>

<div align="center">
  <p>
     <img width="100%" src="../docs/images/blue_v3.gif" alt="PP-StructureV3 데모">
  </p>
</div>

## 🌟 최신 소식을 놓치지 마세요

⭐ **이 저장소에 스타를 추가하여 강력한 OCR 및 문서 분석 기능을 포함한 흥미로운 업데이트와 새로운 릴리스를 받아보세요!** ⭐

<div align="center">
  <p>
       <img width="1200" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/star_paddleocr.en.gif" alt="Star-Project">
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
| [Learn more projects](../awesome_projects.md) | [More projects based on PaddleOCR](../awesome_projects.md)|

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
