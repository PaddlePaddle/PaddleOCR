
<div align="center">
  <p>
      <img width="800" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Banner.png" alt="스타 히스토리">
  </p>



<h3>세계 최고의 OCR 툴킷 & 문서 AI 엔진</h3>

[English](../README.md) | [简体中文](./README_cn.md) | [繁體中文](./README_tcn.md) | [日本語](./README_ja.md) | 한국어 | [Français](./README_fr.md) | [Русский](./README_ru.md) | [Español](./README_es.md) | [العربية](./README_ar.md)

<!-- icon -->

[![PyPI Downloads](https://static.pepy.tech/badge/paddleocr)](https://pepy.tech/projects/paddleocr)
[![Used by](https://img.shields.io/badge/Used%20by-6k%2B%20repositories-blue)](https://github.com/PaddlePaddle/PaddleOCR/network/dependents)
![python](https://img.shields.io/badge/python-3.8~3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
![hardware](https://img.shields.io/badge/hardware-cpu%2C%20gpu%2C%20xpu%2C%20npu-yellow.svg)

[![AI Studio](https://img.shields.io/badge/PaddleOCR-_Offiical_Website-1927BA?logo=data:image/png;base64,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&labelColor=white)](https://www.paddleocr.com)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PaddlePaddle/PaddleOCR)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](../LICENSE)

</div>






**PaddleOCR는 문서와 이미지를 업계 최고 수준의 정확도로 구조화된 LLM 지원 데이터(JSON/Markdown)로 변환합니다. 70,000개 이상의 Star와 Dify, RAGFlow, Cherry Studio 등 최상위 프로젝트의 신뢰를 받는 PaddleOCR는 지능형 RAG 및 에이전트 기반 애플리케이션 구축의 핵심 기반입니다.**


## 🚀 주요 기능

### 📄 지능형 문서 파싱 (LLM 지원)
> *복잡한 시각 자료를 LLM 시대에 맞는 구조화된 데이터로 변환합니다.*

* **최첨단 문서 VLM**: 문서 파싱을 위한 업계 최고의 경량 비전-언어 모델인 **PaddleOCR-VL-1.5 (0.9B)**를 탑재하였습니다. **왜곡, 스캔, 화면 촬영, 조명, 기울어진 문서**라는 5대 "실제 환경" 난제에서 복잡한 문서 파싱에 탁월하며, **Markdown** 및 **JSON** 형식의 구조화된 출력을 지원합니다.
* **구조 인식 변환**: **PP-StructureV3**를 기반으로 복잡한 PDF와 이미지를 **Markdown** 또는 **JSON**으로 원활하게 변환합니다. PaddleOCR-VL 시리즈 모델과 달리 표 셀 좌표, 텍스트 좌표 등 더욱 세밀한 좌표 정보를 제공합니다.
* **상용 수준의 효율성**: 초소형 모델로 상용 등급의 정확도를 달성합니다. 공개 벤치마크에서 다수의 비공개 솔루션을 능가하면서도 엣지/클라우드 배포에 적합한 자원 효율성을 유지합니다.

### 🔍 범용 텍스트 인식 (장면 OCR)
> *고속 다국어 텍스트 탐지의 글로벌 표준.*

* **100개 이상의 언어 지원**: 방대한 글로벌 언어 라이브러리를 기본 지원합니다. **PP-OCRv5** 단일 모델 솔루션은 다국어 혼합 문서(중국어, 영어, 일본어, 병음 등)를 원활하게 처리합니다.
* **복잡한 요소 처리**: 표준 텍스트 인식을 넘어 신분증, 거리 풍경, 도서, 산업 부품 등 다양한 환경에서의 **자연 장면 텍스트 탐지**를 지원합니다.
* **성능 도약**: PP-OCRv5는 이전 버전 대비 **13% 정확도 향상**을 달성하면서도 PaddleOCR의 상징인 "극한 효율성"을 유지합니다.

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Arch.jpg" alt="PaddleOCR 아키텍처">
  </p>
</div>

### 🛠️ 개발자 중심 생태계
* **원활한 통합**: AI 에이전트 생태계의 최고 선택 - **Dify, RAGFlow, Pathway, Cherry Studio**와 깊이 통합되어 있습니다.
* **LLM 데이터 플라이휠**: 고품질 데이터셋 구축을 위한 완전한 파이프라인으로, 대규모 언어 모델 파인튜닝을 위한 지속 가능한 "데이터 엔진"을 제공합니다.
* **원클릭 배포**: 다양한 하드웨어 백엔드(NVIDIA GPU, Intel CPU, Kunlunxin XPU 및 다양한 AI 가속기)를 지원합니다.


## 📣 최근 업데이트

### 🔥 PaddleOCR v3.5.0 출시: 더 유연한 추론 백엔드와 더 풍부한 문서 출력
* **유연한 추론 백엔드 전환**: Paddle 정적 그래프, Paddle 동적 그래프, Transformers 사이를 원활하게 전환할 수 있습니다. Hugging Face 생태계에 깊이 통합되었으며, 주요 20개 모델이 Transformers를 추론 백엔드로 지원합니다.
* **Office 문서를 Markdown으로 변환**: Word, Excel, PowerPoint 등 일반적인 문서 형식을 Markdown으로 변환할 수 있습니다.
* **분석 결과 DOCX 내보내기**: `PaddleOCR-VL` 시리즈, `PP-StructureV3`, `PP-DocTranslation`이 이제 분석 결과를 DOCX로 내보내 Microsoft Word에서 편리하게 확인하고 편집할 수 있습니다.
* **공식 브라우저 추론 SDK**: 공식 브라우저 추론 SDK `PaddleOCR.js`를 출시하여 브라우저에서 `PP-OCRv5`를 실행할 수 있습니다.

<details>
<summary><strong>2026.01.29: PaddleOCR 3.4.0 출시</strong></summary>
* **PaddleOCR-VL-1.5 (최첨단 0.9B VLM)**: 문서 파싱을 위한 최신 플래그십 모델이 출시되었습니다!
    * **OmniDocBench에서 94.5% 정확도**: 최상위 범용 대규모 모델 및 전문 문서 파서를 능가합니다.
    * **실제 환경 강건성**: 비정형 형태 위치 지정을 위한 **PP-DocLayoutV3** 알고리즘을 최초 도입하여 *기울기, 왜곡, 스캔, 조명, 화면 촬영*이라는 5가지 까다로운 시나리오를 정복합니다.
    * **기능 확장**: **인감 인식**, **텍스트 탐지**를 지원하며, **111개 언어**(중국 티베트 문자 및 벵골어 포함)로 확장되었습니다.
    * **장문서 처리**: 자동 교차 페이지 표 병합 및 계층적 제목 식별을 지원합니다.
    * **지금 사용해 보세요**: [HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) 또는 [공식 웹사이트](https://www.paddleocr.com)에서 이용 가능합니다.

</details>

<details>
<summary><strong>2025.10.16: PaddleOCR 3.3.0 출시</strong></summary>

- PaddleOCR-VL 출시:
    - **모델 소개**:
        - **PaddleOCR-VL**은 문서 파싱에 특화된 최첨단 자원 효율적 모델입니다. 핵심 구성 요소인 PaddleOCR-VL-0.9B는 NaViT 스타일의 동적 해상도 비주얼 인코더와 ERNIE-4.5-0.3B 언어 모델을 통합하여 정확한 요소 인식을 가능하게 하는 컴팩트하면서도 강력한 비전-언어 모델(VLM)입니다. **이 혁신적인 모델은 109개 언어를 효율적으로 지원하며, 최소한의 자원 소비를 유지하면서 복잡한 요소(텍스트, 표, 수식, 차트 등) 인식에 탁월합니다**. 널리 사용되는 공개 벤치마크와 자체 벤치마크를 통한 종합 평가에서 PaddleOCR-VL은 페이지 수준 문서 파싱과 요소 수준 인식 모두에서 최첨단 성능을 달성합니다. 기존 솔루션을 크게 능가하고, 최상위 VLM과의 경쟁에서도 강한 경쟁력을 보이며, 빠른 추론 속도를 제공합니다. 이러한 강점은 실제 시나리오에서의 실용적 배포에 매우 적합합니다. 모델은 [HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)에 공개되었습니다. 누구나 다운로드하여 사용할 수 있습니다! 더 자세한 소개는 [PaddleOCR-VL](https://www.paddleocr.ai/latest/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.html)에서 확인할 수 있습니다.

    - **핵심 기능**:
        - **컴팩트하면서도 강력한 VLM 아키텍처**: 자원 효율적 추론에 특화 설계된 새로운 비전-언어 모델을 제시하며, 요소 인식에서 탁월한 성능을 달성합니다. NaViT 스타일의 동적 고해상도 비주얼 인코더와 경량 ERNIE-4.5-0.3B 언어 모델을 통합하여 모델의 인식 능력과 디코딩 효율성을 크게 향상시킵니다. 이러한 통합은 높은 정확도를 유지하면서 연산 요구량을 줄여 효율적이고 실용적인 문서 처리 애플리케이션에 적합합니다.
        - **문서 파싱 최첨단 성능**: PaddleOCR-VL은 페이지 수준 문서 파싱과 요소 수준 인식 모두에서 최첨단 성능을 달성합니다. 기존 파이프라인 기반 솔루션을 크게 능가하며, 문서 파싱에서 선도적인 비전-언어 모델(VLM)과 강한 경쟁력을 보입니다. 또한 텍스트, 표, 수식, 차트 등 복잡한 문서 요소 인식에 탁월하여 필기체 텍스트 및 역사 문서를 포함한 다양하고 까다로운 콘텐츠 유형에 적합합니다. 이로 인해 다양한 문서 유형과 시나리오에 폭넓게 활용할 수 있습니다.
        - **다국어 지원**: PaddleOCR-VL은 109개 언어를 지원하며, 중국어, 영어, 일본어, 라틴어, 한국어를 포함한(이에 국한되지 않는) 주요 글로벌 언어와 러시아어(키릴 문자), 아랍어, 힌디어(데바나가리 문자), 태국어 등 다양한 문자 체계와 구조를 가진 언어를 포괄합니다. 이러한 광범위한 언어 지원은 다국어 및 글로벌 문서 처리 시나리오에서의 시스템 적용성을 크게 향상시킵니다.

- PP-OCRv5 다국어 인식 모델 출시:
    - 라틴 문자 인식의 정확도와 범위를 개선하였습니다. 키릴 문자, 아랍 문자, 데바나가리 문자, 텔루구 문자, 타밀 문자 및 기타 언어 체계 지원을 추가하여 109개 언어 인식을 지원합니다. 모델 파라미터는 2M에 불과하며, 일부 모델의 정확도는 이전 세대 대비 40% 이상 향상되었습니다.

</details>


<details>
<summary><strong>2025.08.21: PaddleOCR 3.2.0 출시</strong></summary>

- **주요 모델 추가:**
    - PP-OCRv5 인식 모델의 영어, 태국어, 그리스어 학습, 추론 및 배포를 도입하였습니다. **PP-OCRv5 영어 모델은 기존 PP-OCRv5 메인 모델 대비 영어 시나리오에서 11% 향상을 달성했으며, 태국어 및 그리스어 인식 모델은 각각 82.68%와 89.28%의 정확도를 기록하였습니다.**

- **배포 기능 업그레이드:**
    - **PaddlePaddle 프레임워크 버전 3.1.0 및 3.1.1 완전 지원.**
    - **PP-OCRv5 C++ 로컬 배포 솔루션 종합 업그레이드, Linux 및 Windows 모두 지원하며 Python 구현과 동일한 기능 및 정확도를 제공합니다.**
    - **고성능 추론이 CUDA 12를 지원하며, Paddle Inference 또는 ONNX Runtime 백엔드를 사용하여 추론을 수행할 수 있습니다.**
    - **고안정성 서비스 지향 배포 솔루션이 완전히 오픈소스화되어, 사용자가 필요에 따라 Docker 이미지와 SDK를 커스터마이징할 수 있습니다.**
    - 고안정성 서비스 지향 배포 솔루션은 수동으로 구성한 HTTP 요청을 통한 호출도 지원하여, 모든 프로그래밍 언어에서 클라이언트 측 코드 개발이 가능합니다.

- **벤치마크 지원:**
    - **모든 프로덕션 라인이 세분화된 벤치마킹을 지원하여, 엔드투엔드 추론 시간은 물론 레이어별 및 모듈별 지연 시간 데이터를 측정하여 성능 분석을 지원합니다. 벤치마크 기능의 설정 및 사용 방법은 [여기](docs/version3.x/pipeline_usage/instructions/benchmark.en.md)를 참조하세요.**
    - **문서에 주류 하드웨어에서의 일반 구성에 대한 주요 지표(추론 지연 시간 및 메모리 사용량 등)가 업데이트되어, 사용자에게 배포 참고 자료를 제공합니다.**

- **버그 수정:**
    - 모델 학습 중 로그 저장 실패 문제를 해결하였습니다.
    - 수식 모델의 데이터 증강 구성 요소를 업그레이드하여 최신 버전의 albumentations 종속성과의 호환성을 확보하고, 멀티프로세스 시나리오에서 tokenizers 패키지 사용 시 발생하는 데드락 경고를 수정하였습니다.
    - PP-StructureV3 구성 파일에서 스위치 동작(예: `use_chart_parsing`)이 다른 파이프라인과 불일치하는 문제를 수정하였습니다.

- **기타 개선 사항:**
    - **핵심 종속성과 선택적 종속성을 분리하였습니다. 기본 텍스트 인식에는 최소한의 핵심 종속성만 필요하며, 문서 파싱 및 정보 추출을 위한 추가 종속성은 필요에 따라 설치할 수 있습니다.**
    - **Windows에서 NVIDIA RTX 50 시리즈 그래픽 카드 지원을 활성화하였습니다. 해당 PaddlePaddle 프레임워크 버전은 [설치 가이드](docs/version3.x/installation.en.md)를 참조하세요.**
    - **PP-OCR 시리즈 모델이 단일 문자 좌표 반환을 지원합니다.**
    - AIStudio, ModelScope 등 모델 다운로드 소스를 추가하여, 사용자가 모델 다운로드 소스를 지정할 수 있습니다.
    - PP-Chart2Table 모듈을 통한 차트-표 변환 지원을 추가하였습니다.
    - 사용성 향상을 위해 문서 설명을 최적화하였습니다.
</details>


[변경 이력](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)

</details>


## 🚀 빠른 시작

### 1단계: 온라인 체험
PaddleOCR 공식 웹사이트에서는 별도 설정 없이 클릭 한 번으로 체험할 수 있는 인터랙티브 **체험 센터**와 **API**를 제공합니다.

👉 [공식 웹사이트 방문](https://www.paddleocr.com)

### 2단계: 로컬 배포
로컬 사용을 위해 필요에 따라 다음 문서를 참조하세요:

- **PP-OCR 시리즈**: [PP-OCR 문서](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html) 참조
- **PaddleOCR-VL 시리즈**: [PaddleOCR-VL 문서](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html) 참조
- **PP-StructureV3**: [PP-StructureV3 문서](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PP-StructureV3.html) 참조
- **추가 기능**: [추가 기능 문서](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/pipeline_overview.html) 참조


## 🧩 추가 기능

- 모델을 ONNX 형식으로 변환: [ONNX 모델 획득](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/obtaining_onnx_models.html).
- OpenVINO, ONNX Runtime, TensorRT 등의 엔진을 사용한 추론 가속 또는 ONNX 형식 모델을 사용한 추론: [고성능 추론](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/high_performance_inference.html).
- 다중 GPU 및 다중 프로세스를 사용한 추론 가속: [파이프라인 병렬 추론](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/instructions/parallel_inference.html).
- PaddleOCR를 C++, C#, Java 등으로 작성된 애플리케이션에 통합: [서빙](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/serving.html).

## 🔄 실행 결과 빠른 미리보기

### PP-OCRv5

<div align="center">
  <p>
       <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PP-OCRv5_demo.gif" alt="PP-OCRv5 데모">
  </p>
</div>



### PP-StructureV3

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PP-StructureV3_demo.gif" alt="PP-StructureV3 데모">
  </p>
</div>

### PaddleOCR-VL

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PaddleOCR-VL_demo.gif" alt="PaddleOCR-VL 데모">
  </p>
</div>


## ✨ 소식을 받아보세요

⭐ **이 저장소에 Star를 눌러 강력한 OCR 및 문서 파싱 기능을 포함한 흥미로운 업데이트와 새로운 릴리스를 확인하세요!** ⭐

<div align="center">
  <p>
       <img width="1200" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/star_paddleocr2.en.gif" alt="프로젝트에 Star 누르기">
  </p>
</div>


## 👩‍👩‍👧‍👦 커뮤니티

<div align="center">

| PaddlePaddle 위챗 공식 계정 | 기술 토론 그룹 참여 |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |
</div>


## 😃 PaddleOCR를 활용한 멋진 프로젝트
PaddleOCR가 오늘날의 모습을 갖추기까지 놀라운 커뮤니티의 힘이 있었습니다! 💗 오랜 파트너, 새로운 협력자, 그리고 PaddleOCR에 열정을 쏟아주신 모든 분들께 진심으로 감사드립니다 - 여기에 이름이 언급되지 않은 분들까지 포함하여. 여러분의 지원이 우리의 원동력입니다!

<div align="center">

| 프로젝트 이름 | 설명 |
| ------------ | ----------- |
| [Dify](https://github.com/langgenius/dify) <a href="https://github.com/langgenius/dify"><img src="https://img.shields.io/github/stars/langgenius/dify"></a>|에이전트 워크플로우 개발을 위한 프로덕션 지원 플랫폼.|
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|심층 문서 이해 기반 RAG 엔진.|
| [pathway](https://github.com/pathwaycom/pathway) <a href="https://github.com/pathwaycom/pathway"><img src="https://img.shields.io/github/stars/pathwaycom/pathway"></a>|스트림 처리, 실시간 분석, LLM 파이프라인 및 RAG를 위한 Python ETL 프레임워크.|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|다양한 유형의 문서를 Markdown으로 변환하는 도구.|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|무료, 오픈소스, 일괄 오프라인 OCR 소프트웨어.|
| [cherry-studio](https://github.com/CherryHQ/cherry-studio) <a href="https://github.com/CherryHQ/cherry-studio"><img src="https://img.shields.io/github/stars/CherryHQ/cherry-studio"></a>|여러 LLM 제공업체를 지원하는 데스크톱 클라이언트.|
| [haystack](https://github.com/deepset-ai/haystack)<a href="https://github.com/deepset-ai/haystack"><img src="https://img.shields.io/github/stars/deepset-ai/haystack"></a> |커스터마이징 가능한 프로덕션 지원 LLM 애플리케이션 구축을 위한 AI 오케스트레이션 프레임워크.|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |순수 비전 기반 GUI 에이전트를 위한 화면 파싱 도구.|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |모든 것에 기반한 질의응답.|
| [더 많은 프로젝트 보기](./awesome_projects.md) | [PaddleOCR 기반 추가 프로젝트](./awesome_projects.md)|
</div>

## 👩‍👩‍👧‍👦 기여자

<div align="center">
<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>
</div>

## 🌟 Star

<div align="center">
  <p>
      <img width="800" src="https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date" alt="스타 히스토리">
  </p>
</div>


## 📄 라이선스
이 프로젝트는 [Apache 2.0 라이선스](LICENSE)로 배포됩니다.

## 🎓 인용

```bibtex
@misc{cui2025paddleocr30technicalreport,
      title={PaddleOCR 3.0 Technical Report},
      author={Cheng Cui and Ting Sun and Manhui Lin and Tingquan Gao and Yubo Zhang and Jiaxuan Liu and Xueqing Wang and Zelun Zhang and Changda Zhou and Hongen Liu and Yue Zhang and Wenyu Lv and Kui Huang and Yichao Zhang and Jing Zhang and Jun Zhang and Yi Liu and Dianhai Yu and Yanjun Ma},
      year={2025},
      eprint={2507.05595},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.05595},
}

@misc{cui2025paddleocrvlboostingmultilingualdocument,
      title={PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model},
      author={Cheng Cui and Ting Sun and Suyin Liang and Tingquan Gao and Zelun Zhang and Jiaxuan Liu and Xueqing Wang and Changda Zhou and Hongen Liu and Manhui Lin and Yue Zhang and Yubo Zhang and Handong Zheng and Jing Zhang and Jun Zhang and Yi Liu and Dianhai Yu and Yanjun Ma},
      year={2025},
      eprint={2510.14528},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.14528},
}

@misc{cui2026paddleocrvl15multitask09bvlm,
      title={PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM for Robust In-the-Wild Document Parsing},
      author={Cheng Cui and Ting Sun and Suyin Liang and Tingquan Gao and Zelun Zhang and Jiaxuan Liu and Xueqing Wang and Changda Zhou and Hongen Liu and Manhui Lin and Yue Zhang and Yubo Zhang and Yi Liu and Dianhai Yu and Yanjun Ma},
      year={2026},
      eprint={2601.21957},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.21957},
}
```
