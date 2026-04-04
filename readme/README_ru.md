<div align="center">
  <p>
      <img width="800" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Banner.png" alt="Star-history">
  </p>



<h3>Ведущий в мире инструментарий OCR и движок Document AI</h3>

[English](../README.md) | [简体中文](./README_cn.md) | [繁體中文](./README_tcn.md) | [日本語](./README_ja.md) | [한국어](./README_ko.md) | [Français](./README_fr.md) | Русский | [Español](./README_es.md) | [العربية](./README_ar.md)

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






**PaddleOCR преобразует документы и изображения в структурированные данные, готовые для использования с LLM (JSON/Markdown), с точностью мирового уровня. Имея более 70 тысяч звёзд и доверие таких ведущих проектов, как Dify, RAGFlow и Cherry Studio, PaddleOCR является основой для создания интеллектуальных приложений RAG и Agentic.**


## 🚀 Ключевые возможности

### 📄 Интеллектуальный разбор документов (готово для LLM)
> *Преобразование сложных визуальных данных в структурированные данные для эпохи LLM.*

* **SOTA Document VLM**: В основе — **PaddleOCR-VL-1.5 (0.9B)**, ведущая в отрасли лёгкая визуально-языковая модель для разбора документов. Она превосходно справляется с разбором сложных документов в 5 основных «реальных» сценариях: **деформация, сканирование, фотосъёмка экрана, неравномерное освещение и перекошенные документы**, формируя структурированный вывод в форматах **Markdown** и **JSON**.
* **Конвертация с учётом структуры**: На основе **PP-StructureV3** — бесшовное преобразование сложных PDF-файлов и изображений в **Markdown** или **JSON**. В отличие от моделей серии PaddleOCR-VL, предоставляет более детальную координатную информацию, включая координаты ячеек таблиц, координаты текста и многое другое.
* **Эффективность промышленного уровня**: Коммерческая точность при минимальном объёме ресурсов. Превосходит многочисленные закрытые решения в публичных тестах, оставаясь ресурсоэффективным для развёртывания на периферийных устройствах и в облаке.

### 🔍 Универсальное распознавание текста (Scene OCR)
> *Мировой золотой стандарт высокоскоростного многоязычного обнаружения текста.*

* **Поддержка 100+ языков**: Нативное распознавание обширной глобальной библиотеки. Наше единое решение **PP-OCRv5** элегантно обрабатывает многоязычные смешанные документы (китайский, английский, японский, пиньинь и др.).
* **Мастерство работы со сложными элементами**: Помимо стандартного распознавания текста, поддерживается **обнаружение текста в естественных сценах** в широком диапазоне условий, включая удостоверения личности, уличные виды, книги и промышленные компоненты.
* **Скачок производительности**: PP-OCRv5 обеспечивает **повышение точности на 13%** по сравнению с предыдущими версиями, сохраняя «экстремальную эффективность», за которую PaddleOCR получил широкую известность.

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Arch.jpg" alt="PaddleOCR Architecture">
  </p>
</div>

### 🛠️ Экосистема, ориентированная на разработчиков
* **Бесшовная интеграция**: Первый выбор для экосистемы AI Agent — глубокая интеграция с **Dify, RAGFlow, Pathway и Cherry Studio**.
* **Маховик данных для LLM**: Полный конвейер для создания высококачественных наборов данных, обеспечивающий устойчивый «Data Engine» для тонкой настройки больших языковых моделей.
* **Развёртывание в один клик**: Поддержка различных аппаратных бэкендов (NVIDIA GPU, Intel CPU, Kunlunxin XPU и разнообразные AI-ускорители).


## 📣 Последние обновления

### 🔥 [2026.01.29] Выпуск PaddleOCR v3.4.0: Эра разбора нестандартных документов
* **PaddleOCR-VL-1.5 (SOTA 0.9B VLM)**: Наша новейшая флагманская модель для разбора документов уже доступна!
    * **94,5% точность на OmniDocBench**: Превосходит ведущие универсальные большие модели и специализированные парсеры документов.
    * **Устойчивость к реальным условиям**: Первая реализация алгоритма **PP-DocLayoutV3** для позиционирования нестандартных форм, освоившая 5 сложных сценариев: *перекос, деформация, сканирование, неравномерное освещение и фотосъёмка экрана*.
    * **Расширение возможностей**: Теперь поддерживается **распознавание печатей**, **обнаружение текста** и расширение до **111 языков** (включая тибетское письмо Китая и бенгальский язык).
    * **Работа с длинными документами**: Поддержка автоматического объединения таблиц на нескольких страницах и иерархической идентификации заголовков.
    * **Попробуйте сейчас**: Доступно на [HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) или на нашем [официальном сайте](https://www.paddleocr.com).

<details>
<summary><strong>2025.10.16: Выпуск PaddleOCR 3.3.0</strong></summary>

- Выпуск PaddleOCR-VL:
    - **Описание модели**:
        - **PaddleOCR-VL** — это SOTA-модель с эффективным использованием ресурсов, разработанная специально для разбора документов. Её ключевым компонентом является PaddleOCR-VL-0.9B — компактная, но мощная визуально-языковая модель (VLM), объединяющая динамический визуальный энкодер с переменным разрешением в стиле NaViT с языковой моделью ERNIE-4.5-0.3B для точного распознавания элементов. **Эта инновационная модель эффективно поддерживает 109 языков и превосходно справляется с распознаванием сложных элементов (например, текста, таблиц, формул и диаграмм), сохраняя минимальное потребление ресурсов**. По результатам комплексных оценок на широко используемых публичных тестах и внутренних тестах PaddleOCR-VL достигает SOTA-производительности как в разборе документов на уровне страниц, так и в распознавании элементов. Она значительно превосходит существующие решения, демонстрирует высокую конкурентоспособность по сравнению с ведущими VLM и обеспечивает высокую скорость вывода. Эти преимущества делают её высокопригодной для практического развёртывания в реальных сценариях. Модель опубликована на [HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL). Приглашаем всех скачать и использовать! Дополнительная информация доступна в разделе [PaddleOCR-VL](https://www.paddleocr.ai/latest/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.html).

    - **Основные возможности**:
        - **Компактная, но мощная архитектура VLM**: Представлена новая визуально-языковая модель, специально разработанная для ресурсоэффективного вывода, достигающая выдающейся производительности в распознавании элементов. Благодаря интеграции динамического высокоразрешающего визуального энкодера в стиле NaViT с лёгкой языковой моделью ERNIE-4.5-0.3B мы значительно повысили возможности распознавания и эффективность декодирования модели. Эта интеграция сохраняет высокую точность при снижении вычислительных требований, что делает её хорошо подходящей для эффективной и практической обработки документов.
        - **SOTA-производительность в разборе документов**: PaddleOCR-VL достигает передовой производительности как в разборе документов на уровне страниц, так и в распознавании элементов. Она значительно превосходит существующие конвейерные решения и демонстрирует высокую конкурентоспособность по сравнению с ведущими визуально-языковыми моделями (VLM) в разборе документов. Кроме того, она превосходно справляется с распознаванием сложных элементов документов, таких как текст, таблицы, формулы и диаграммы, что делает её пригодной для широкого спектра сложных типов контента, включая рукописный текст и исторические документы. Это делает её высоко универсальной и подходящей для широкого спектра типов и сценариев документов.
        - **Многоязычная поддержка**: PaddleOCR-VL поддерживает 109 языков, охватывая основные мировые языки, включая, но не ограничиваясь китайским, английским, японским, латинским и корейским, а также языки с различными системами письма и структурами, такие как русский (кириллица), арабский, хинди (письмо деванагари) и тайский. Широкий охват языков существенно повышает применимость нашей системы к многоязычным и глобализированным сценариям обработки документов.

- Выпуск PP-OCRv5 — многоязычной модели распознавания:
    - Улучшена точность и охват распознавания латинского письма; добавлена поддержка кириллицы, арабского, деванагари, телугу, тамильского и других языковых систем, охватывающих распознавание 109 языков. Модель имеет всего 2 МБ параметров, а точность некоторых моделей выросла более чем на 40% по сравнению с предыдущим поколением.

</details>


<details>
<summary><strong>2025.08.21: Выпуск PaddleOCR 3.2.0</strong></summary>

- **Значительные дополнения моделей:**
    - Введены обучение, вывод и развёртывание моделей распознавания PP-OCRv5 для английского, тайского и греческого языков. **Модель PP-OCRv5 для английского языка обеспечивает улучшение на 11% в английских сценариях по сравнению с основной моделью PP-OCRv5, при этом модели распознавания тайского и греческого языков достигают точности 82,68% и 89,28% соответственно.**

- **Улучшения возможностей развёртывания:**
    - **Полная поддержка версий фреймворка PaddlePaddle 3.1.0 и 3.1.1.**
    - **Комплексное обновление решения для локального развёртывания PP-OCRv5 на C++, теперь поддерживающего как Linux, так и Windows, с полным соответствием функций и идентичной точностью реализации на Python.**
    - **Высокопроизводительный вывод теперь поддерживает CUDA 12, а вывод может выполняться с использованием бэкендов Paddle Inference или ONNX Runtime.**
    - **Решение для высоконадёжного сервисного развёртывания теперь полностью открыто, позволяя пользователям при необходимости настраивать образы Docker и SDK.**
    - Решение для высоконадёжного сервисного развёртывания также поддерживает вызов через вручную сформированные HTTP-запросы, что позволяет разрабатывать клиентский код на любом языке программирования.

- **Поддержка бенчмарков:**
    - **Все производственные конвейеры теперь поддерживают детализированное бенчмаркирование, позволяя измерять сквозное время вывода, а также задержки на уровне отдельных слоёв и модулей для анализа производительности. [Здесь](docs/version3.x/pipeline_usage/instructions/benchmark.en.md) описано, как настроить и использовать функцию бенчмарка.**
    - **Документация обновлена и включает ключевые метрики для часто используемых конфигураций на основном оборудовании, такие как задержка вывода и использование памяти, предоставляя справочные данные для развёртывания.**

- **Исправления ошибок:**
    - Устранена проблема с неудачным сохранением журналов во время обучения модели.
    - Обновлён компонент аугментации данных для моделей формул для совместимости с более новыми версиями зависимости albumentations, а также исправлены предупреждения о взаимоблокировке при использовании пакета tokenizers в многопроцессорных сценариях.
    - Исправлены несоответствия в поведении переключателей (например, `use_chart_parsing`) в файлах конфигурации PP-StructureV3 по сравнению с другими конвейерами.

- **Прочие улучшения:**
    - **Разделены основные и дополнительные зависимости. Для базового распознавания текста требуются только минимальные основные зависимости; дополнительные зависимости для разбора документов и извлечения информации могут быть установлены по мере необходимости.**
    - **Включена поддержка видеокарт NVIDIA RTX серии 50 на Windows; пользователи могут обратиться к [руководству по установке](docs/version3.x/installation.en.md) для получения информации о соответствующих версиях фреймворка PaddlePaddle.**
    - **Модели серии PP-OCR теперь поддерживают возврат координат отдельных символов.**
    - Добавлены источники загрузки моделей AIStudio, ModelScope и другие, позволяющие пользователям указывать источник для загрузки моделей.
    - Добавлена поддержка преобразования диаграмм в таблицы через модуль PP-Chart2Table.
    - Оптимизированы описания в документации для улучшения удобства использования.
</details>


[История изменений](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)

</details>


## 🚀 Быстрый старт

### Шаг 1: Попробуйте онлайн
Официальный сайт PaddleOCR предоставляет интерактивный **Центр опыта** и **API** — без необходимости настройки, просто один клик для ознакомления.

👉 [Посетить официальный сайт](https://www.paddleocr.com)

### Шаг 2: Локальное развёртывание
Для локального использования обратитесь к следующей документации в соответствии с вашими потребностями:

- **Серия PP-OCR**: См. [Документацию PP-OCR](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html)
- **Серия PaddleOCR-VL**: См. [Документацию PaddleOCR-VL](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- **PP-StructureV3**: См. [Документацию PP-StructureV3](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PP-StructureV3.html)
- **Дополнительные возможности**: См. [Документацию по дополнительным возможностям](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/pipeline_overview.html)


## 🧩 Дополнительные возможности

- Конвертация моделей в формат ONNX: [Получение моделей ONNX](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/obtaining_onnx_models.html).
- Ускорение вывода с использованием движков OpenVINO, ONNX Runtime, TensorRT или выполнение вывода с использованием моделей в формате ONNX: [Высокопроизводительный вывод](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/high_performance_inference.html).
- Ускорение вывода с использованием нескольких GPU и многопроцессорной обработки: [Параллельный вывод для конвейеров](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/instructions/parallel_inference.html).
- Интеграция PaddleOCR в приложения, написанные на C++, C#, Java и др.: [Сервисное развёртывание](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/serving.html).

## 🔄 Краткий обзор результатов выполнения

### PP-OCRv5

<div align="center">
  <p>
       <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PP-OCRv5_demo.gif" alt="PP-OCRv5 Demo">
  </p>
</div>



### PP-StructureV3

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PP-StructureV3_demo.gif" alt="PP-StructureV3 Demo">
  </p>
</div>

### PaddleOCR-VL

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PaddleOCR-VL_demo.gif" alt="PP-StructureV3 Demo">
  </p>
</div>


## ✨ Следите за обновлениями

⭐ **Добавьте этот репозиторий в избранное, чтобы быть в курсе захватывающих обновлений и новых выпусков, включая мощные возможности OCR и разбора документов!** ⭐

<div align="center">
  <p>
       <img width="1200" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/star_paddleocr2.en.gif" alt="Star-Project">
  </p>
</div>


## 👩‍👩‍👧‍👦 Сообщество

<div align="center">

| Официальный аккаунт PaddlePaddle в WeChat | Присоединиться к группе технических обсуждений |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |
</div>


## 😃 Замечательные проекты, использующие PaddleOCR
PaddleOCR не достиг бы своего нынешнего уровня без своего невероятного сообщества! 💗 Огромная благодарность всем нашим давним партнёрам, новым соавторам и всем, кто вложил свою душу в PaddleOCR — независимо от того, упомянуты вы здесь или нет. Ваша поддержка питает наш огонь!

<div align="center">

| Название проекта | Описание |
| ------------ | ----------- |
| [Dify](https://github.com/langgenius/dify) <a href="https://github.com/langgenius/dify"><img src="https://img.shields.io/github/stars/langgenius/dify"></a>|Готовая к производству платформа для разработки агентных рабочих процессов.|
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|RAG-движок на основе глубокого понимания документов.|
| [pathway](https://github.com/pathwaycom/pathway) <a href="https://github.com/pathwaycom/pathway"><img src="https://img.shields.io/github/stars/pathwaycom/pathway"></a>|Python ETL-фреймворк для потоковой обработки, аналитики в реальном времени, конвейеров LLM и RAG.|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|Инструмент для конвертации документов различных типов в Markdown.|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|Бесплатное программное обеспечение для пакетного офлайн-OCR с открытым исходным кодом.|
| [cherry-studio](https://github.com/CherryHQ/cherry-studio) <a href="https://github.com/CherryHQ/cherry-studio"><img src="https://img.shields.io/github/stars/CherryHQ/cherry-studio"></a>|Настольный клиент с поддержкой нескольких провайдеров LLM.|
| [haystack](https://github.com/deepset-ai/haystack)<a href="https://github.com/deepset-ai/haystack"><img src="https://img.shields.io/github/stars/deepset-ai/haystack"></a> |Фреймворк оркестрации AI для создания настраиваемых, готовых к производству приложений LLM.|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |OmniParser: инструмент разбора экрана для агента GUI на основе чистого зрения.|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |Вопросы и ответы на основе чего угодно.|
| [Узнать о других проектах](./awesome_projects.md) | [Другие проекты на основе PaddleOCR](./awesome_projects.md)|
</div>

## 👩‍👩‍👧‍👦 Участники

<div align="center">
<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>
</div>

## 🌟 Звёзды

<div align="center">
  <p>
      <img width="800" src="https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date" alt="Star-history">
  </p>
</div>


## 📄 Лицензия
Этот проект выпущен под лицензией [Apache 2.0](LICENSE).

## 🎓 Цитирование

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
