<div align="center">
  <p>
      <img width="100%" src="./docs/images/Banner.png" alt="Баннер PaddleOCR">
  </p>

<!-- language -->
[English](./README.md) | [简体中文](./README_cn.md) | [繁體中文](./README_tcn.md) | [日本語](./README_ja.md) | [한국어](./README_ko.md) | [Français](./README_fr.md) | Русский | [Español](./README_es.md) | [العربية](./README_ar.md)

<!-- icon -->

[![stars](https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf)](https://github.com/PaddlePaddle/PaddleOCR)
[![Downloads](https://img.shields.io/pypi/dm/paddleocr)](https://pypi.org/project/PaddleOCR/)
![python](https://img.shields.io/badge/python-3.8～3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
![hardware](https://img.shields.io/badge/hardware-cpu%2C%20gpu%2C%20xpu%2C%20npu-yellow.svg)


[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

</div>

## 🚀 Введение
С момента своего первого выпуска PaddleOCR получил широкое признание в академических, промышленных и исследовательских кругах благодаря своим передовым алгоритмам и доказанной производительности в реальных приложениях. Он уже используется в таких популярных проектах с открытым исходным кодом, как Umi-OCR, OmniParser, MinerU и RAGFlow, что делает его предпочтительным инструментарием OCR для разработчиков по всему миру.

20 мая 2025 года команда PaddlePaddle представила PaddleOCR 3.0, полностью совместимый с официальным выпуском фреймворка **PaddlePaddle 3.0**. Это обновление еще больше **повышает точность распознавания текста**, добавляет поддержку **распознавания нескольких типов текста** и **распознавания рукописного текста**, а также удовлетворяет растущий спрос на приложения с большими моделями для **высокоточного анализа сложных документов**. В сочетании с **ERNIE 4.5 Turbo** он значительно улучшает точность извлечения ключевой информации. PaddleOCR 3.0 также вводит поддержку китайских гетерогенных AI ускорителей, таких как **KUNLUNXIN** и **Ascend**. Для получения полной документации по использованию, пожалуйста, обратитесь к [Документации PaddleOCR 3.0](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html).

Три новые ключевые функции в PaddleOCR 3.0:
- Универсальная модель распознавания текста в любых сценах [PP-OCRv5](./docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.en.md): Одна модель обрабатывает пять различных типов текста и сложный рукописный ввод. Общая точность распознавания увеличилась на 13 процентных пунктов по сравнению с предыдущим поколением. [Онлайн-демо](https://aistudio.baidu.com/community/app/91660/webUI)

- Общее решение для парсинга документов [PP-StructureV3](./docs/version3.x/algorithm/PP-StructureV3/PP-StructureV3.en.md): Обеспечивает высокоточный парсинг PDF-файлов с различными макетами и сценариями, превосходя многие решения с открытым и закрытым исходным кодом по результатам публичных тестов. [Онлайн-демо](https://aistudio.baidu.com/community/app/518494/webUI)

- Интеллектуальное решение для понимания документов [PP-ChatOCRv4](./docs/version3.x/algorithm/PP-ChatOCRv4/PP-ChatOCRv4.en.md): Нативно поддерживается большим моделью ERNIE 4.5 Turbo, достигая на 15 процентных пунктов более высокой точности, чем его предшественник. [Онлайн-демо](https://aistudio.baidu.com/community/app/518493/webUI)

Помимо предоставления выдающейся библиотеки моделей, PaddleOCR 3.0 также предлагает удобные инструменты, охватывающие обучение моделей, инференс и развертывание сервисов, чтобы разработчики могли быстро внедрять ИИ-приложения в производство.
<div align="center">
  <p>
      <img width="100%" src="./docs/images/Arch.png" alt="Архитектура PaddleOCR">
  </p>
</div>



## 📣 Последние обновления

#### **🔥🔥 2025.06.05: Релиз PaddleOCR 3.0.1, включает:**

- **Оптимизация некоторых моделей и их конфигураций:**
  - Обновлена конфигурация модели по умолчанию для PP-OCRv5: модели обнаружения и распознавания изменены с `mobile` на `server`. Для улучшения производительности по умолчанию в большинстве сценариев параметр `limit_side_len` в конфигурации изменен с 736 на 64.
  - Добавлена новая модель классификации ориентации строк текста `PP-LCNet_x1_0_textline_ori` с точностью 99.42%. Классификатор ориентации строк текста по умолчанию для пайплайнов OCR, PP-StructureV3 и PP-ChatOCRv4 обновлен до этой модели.
  - Оптимизирована модель классификации ориентации строк текста `PP-LCNet_x0_25_textline_ori`, точность улучшена на 3.3 процентных пункта до текущего значения 98.85%.

- **Оптимизации и исправления некоторых проблем версии 3.0.0, [подробности](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)**

🔥🔥2025.05.20: Официальный релиз **PaddleOCR v3.0**, включающий:
- **PP-OCRv5**: Высокоточная модель распознавания текста для всех сценариев - Мгновенное извлечение текста из изображений/PDF.
   1. 🌐 Поддержка **пяти** типов текста в одной модели - Бесшовная обработка **упрощенного китайского, традиционного китайского, пиньиня, английского** и **японского** в рамках одной модели.
   2. ✍️ Улучшенное **распознавание рукописного текста**: Значительно лучше справляется со сложными слитными и нестандартными почерками.
   3. 🎯 **Прирост точности на 13 процентных пунктов** по сравнению с PP-OCRv4, достижение самых современных результатов в различных реальных сценариях.

- **PP-StructureV3**: Универсальный парсинг документов – Используйте SOTA парсинг изображений/PDF для реальных сценариев! 
   1. 🧮 **Высокоточный парсинг PDF в различных сценариях**, опережающий как открытые, так и закрытые решения на бенчмарке OmniDocBench.
   2. 🧠 Специализированные возможности включают **распознавание печатей**, **преобразование диаграмм в таблицы**, **распознавание таблиц с вложенными формулами/изображениями**, **парсинг документов с вертикальным текстом** и **анализ сложных структур таблиц**.

- **PP-ChatOCRv4**: Интеллектуальное понимание документов – Извлекайте ключевую информацию, а не просто текст из изображений/PDF.
   1. 🔥 **Прирост точности на 15 процентных пунктов** в извлечении ключевой информации из файлов PDF/PNG/JPG по сравнению с предыдущим поколением.
   2. 💻 Нативная поддержка **ERNIE 4.5 Turbo**, с совместимостью для развертывания больших моделей через PaddleNLP, Ollama, vLLM и другие.
   3. 🤝 Интегрирован [PP-DocBee2](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee2), обеспечивающий извлечение и понимание печатного текста, рукописного текста, печатей, таблиц, диаграмм и других общих элементов в сложных документах.

<details>
   <summary><strong>История обновлений</strong></summary>


- 🔥🔥2025.03.07: Релиз **PaddleOCR v2.10**, включающий:

  - **12 новых самостоятельно разработанных моделей:**
    - **[Серия для обнаружения макетов](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html)** (3 модели): PP-DocLayout-L, M и S -- способны обнаруживать 23 распространенных типа макетов в различных форматах документов (статьи, отчеты, экзамены, книги, журналы, контракты и т.д.) на английском и китайском языках. Достигает до **90.4% mAP@0.5**, а легковесные версии могут обрабатывать более 100 страниц в секунду.
    - **[Серия для распознавания формул](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/formula_recognition.html)** (2 модели): PP-FormulaNet-L и S -- поддерживают распознавание более 50 000 выражений LaTeX, обрабатывая как печатные, так и рукописные формулы. PP-FormulaNet-L предлагает **на 6% более высокую точность**, чем сравнимые модели; PP-FormulaNet-S в 16 раз быстрее при сохранении аналогичной точности.
    - **[Серия для распознавания структуры таблиц](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_structure_recognition.html)** (2 модели): SLANeXt_wired и SLANeXt_wireless -- недавно разработанные модели с **улучшением точности на 6%** по сравнению с SLANet_plus в распознавании сложных таблиц.
    - **[Классификация таблиц](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_classification.html)** (1 модель): 
PP-LCNet_x1_0_table_cls -- сверхлегкий классификатор для таблиц с видимыми и невидимыми границами.

[Узнать больше](https://paddlepaddle.github.io/PaddleOCR/latest/en/update.html)

</details>

## ⚡ Быстрый старт
### 1. Запустить онлайн-демо
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

### 2. Установка

Установите PaddlePaddle, следуя [Руководству по установке](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html), после чего установите инструментарий PaddleOCR.

```bash
# Установить paddleocr
pip install paddleocr
```

### 3. Запуск инференса через CLI
```bash
# Запустить инференс PP-OCRv5
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False  

# Запустить инференс PP-StructureV3
paddleocr pp_structurev3 -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png --use_doc_orientation_classify False --use_doc_unwarping False

# Сначала получите Qianfan API Key, а затем запустите инференс PP-ChatOCRv4
paddleocr pp_chatocrv4_doc -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png -k 驾驶室准乘人数 --qianfan_api_key your_api_key --use_doc_orientation_classify False --use_doc_unwarping False 

# Получить больше информации о "paddleocr ocr"
paddleocr ocr --help
```

### 4. Запуск инференса через API
**4.1 Пример для PP-OCRv5**
```python
# Инициализация экземпляра PaddleOCR
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# Запуск инференса OCR на примере изображения
result = ocr.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png")

# Визуализация результатов и сохранение в формате JSON
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

<details>
    <summary><strong>4.2 Пример для PP-StructureV3</strong></summary>

```python
from pathlib import Path
from paddleocr import PPStructureV3

pipeline = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
)

# Для изображений
output = pipeline.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png",
)

# Визуализация результатов и сохранение в формате JSON
for res in output:
    res.print() 
    res.save_to_json(save_path="output") 
    res.save_to_markdown(save_path="output")           
```

</details>

<details>
   <summary><strong>4.3 Пример для PP-ChatOCRv4</strong></summary>

```python
from paddleocr import PPChatOCRv4Doc

chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "ernie-3.5-8k",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key",  # ваш api_key
}

retriever_config = {
    "module_name": "retriever",
    "model_name": "embedding-v1",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "qianfan",
    "api_key": "api_key",  # ваш api_key
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
# Если используется мультимодальная большая модель, необходимо запустить локальный сервис mllm. Вы можете обратиться к документации: https://github.com/PaddlePaddle/PaddleX/blob/release/3.0/docs/pipeline_usage/tutorials/vlm_pipelines/doc_understanding.en.md для выполнения развертывания и обновления конфигурации mllm_chat_bot_config.
if use_mllm:
    mllm_chat_bot_config = {
        "module_name": "chat_bot",
        "model_name": "PP-DocBee",
        "base_url": "http://127.0.0.1:8080/",  # URL вашего локального сервиса mllm
        "api_type": "openai",
        "api_key": "api_key",  # ваш api_key
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

### 5. Китайские гетерогенные ИИ-ускорители
- [Huawei Ascend](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/other_devices_support/paddlepaddle_install_NPU.html)
- [KUNLUNXIN](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/other_devices_support/paddlepaddle_install_XPU.html)

## ⛰️ Продвинутые руководства
- [Руководство по PP-OCRv5](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/OCR.html)
- [Руководство по PP-StructureV3](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-StructureV3.html)
- [Руководство по PP-ChatOCRv4](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-ChatOCRv4.html)

## 🔄 Краткий обзор результатов выполнения

<div align="center">
  <p>
     <img width="100%" src="./docs/images/demo.gif" alt="Демо PP-OCRv5">
  </p>
</div>

<div align="center">
  <p>
     <img width="100%" src="./docs/images/blue_v3.gif" alt="Демо PP-StructureV3">
  </p>
</div>

## 👩‍👩‍👧‍👦 Сообщество

| Официальный аккаунт PaddlePaddle в WeChat | Присоединяйтесь к группе для технических обсуждений |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |


## 😃 Потрясающие проекты, использующие PaddleOCR
PaddleOCR не был бы там, где он есть сегодня, без своего невероятного сообщества! 💗 Огромное спасибо всем нашим давним партнерам, новым сотрудникам и всем, кто вложил свою страсть в PaddleOCR — независимо от того, назвали мы вас или нет. Ваша поддержка разжигает наш огонь!

| Название проекта | Описание |
| ------------ | ----------- |
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|RAG-движок, основанный на глубоком понимании документов.|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|Инструмент для преобразования документов различных типов в Markdown|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|Бесплатное офлайн-программное обеспечение для пакетного OCR с открытым исходным кодом.|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |Инструмент парсинга экрана для GUI-агента, основанного исключительно на компьютерном зрении.|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |Система вопросов и ответов на основе любого контента.|
| [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) <a href="https://github.com/opendatalab/PDF-Extract-Kit"><img src="https://img.shields.io/github/stars/opendatalab/PDF-Extract-Kit"></a>|Мощный инструментарий с открытым исходным кодом, предназначенный для эффективного извлечения высококачественного контента из сложных и разнообразных PDF-документов.|
| [Dango-Translator](https://github.com/PantsuDango/Dango-Translator)<a href="https://github.com/PantsuDango/Dango-Translator"><img src="https://img.shields.io/github/stars/PantsuDango/Dango-Translator"></a> |Распознает текст на экране, переводит его и отображает результаты перевода в режиме реального времени.|
| [Узнать больше о проектах](./awesome_projects.md) | [Больше проектов на основе PaddleOCR](./awesome_projects.md)|

## 👩‍👩‍👧‍👦 Контрибьюторы

<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>


## 🌟 Star

[![Star History Chart](https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date)](https://star-history.com/#PaddlePaddle/PaddleOCR&Date)


## 📄 Лицензия
Этот проект выпущен под [лицензией Apache 2.0](LICENSE).

## 🎓 Цитирование

```
@misc{paddleocr2020,
title={PaddleOCR, Awesome multilingual OCR toolkits based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleOCR}},
year={2020}
}
``` 
