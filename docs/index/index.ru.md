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

## Введение

PaddleOCR стремится создавать многоязычные, потрясающие, передовые и практичные инструменты OCR, которые помогают пользователям обучать лучшие модели и применять их на практике

<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleOCR/releases/download/v2.8.0/demo.gif" width="800">
</div>

## 📣 Последние обновления

- **🔥2022.8.24 Выпуск PaddleOCR [Выпуск /2.6](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6)**
    - Выпускать [PP-Structurev2](../../ppstructure/)，с полностью обновленными функциями и производительностью, адаптированными для китайских сцен и новой поддержкой pаспознавание таблиц
     [Восстановление макета](../../ppstructure/recovery) и **однострочная команда для преобразования PDF в Word**;
    - [Анализ макета](../../ppstructure/layout) оптимизация: память модели уменьшена на 95%, а скорость увеличена в 11 раз, а среднее время процессорного времени составляет всего 41 мс;
    - [Распознавание таблиц](../../ppstructure/table) оптимизация: разработано 3 стратегии оптимизации, а точность модели улучшена на 6% при сопоставимых затратах времени;
    - [Извлечение ключевой информации](../../ppstructure/kie) оптимизация: разработана визуально независимая структура модели, точность распознавания семантической сущности увеличена на 2,8%, а точность извлечения отношения увеличена на 9,1%.
- **🔥2022.7 Выпуск [Коллекция приложений сцены OCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.8/applications/README_en.md)**
- Выпуск **9 вертикальных моделей**, таких как цифровая трубка, ЖК-экран, номерной знак, модель распознавания рукописного ввода, высокоточная модель SVTR и т. д., охватывающих основные вертикальные приложения OCR в целом, производственной, финансовой и транспортной отраслях.
- **🔥2022.5.9 Выпуск PaddleOCR [Выпуск /2.5](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.5)**
- Выпускать [PP-OCRv3](../version2.x/ppocr/overview.en.md#pp-ocrv3): При сопоставимой скорости эффект китайской сцены улучшен на 5% по сравнению с ПП-OCRRv2, эффект английской сцены улучшен на 11%, а средняя точность распознавания 80 языковых многоязычных моделей улучшена более чем на 5%.
- Выпускать [PPOCRLabelv2](https://github.com/PFCCLab/PPOCRLabel/blob/main/README.md): Добавьте функцию аннотации для задачи распознавания таблиц, задачи извлечения ключевой информации и изображения неправильного текста.
    - Выпустить интерактивную электронную книгу [*"Погружение в OCR"*](../version2.x/ppocr/blog/ocr_book.en.md), охватывает передовую теорию и практику кодирования технологии полного стека OCR.
- [подробнее](../update/update.en.md)

## 🌟 Функции

PaddleOCR поддерживает множество передовых алгоритмов, связанных с распознаванием текста, и разработала промышленные модели/решения. [PP-OCR](../version2.x/ppocr/overview.en.md) и [PP-Structure](../../ppstructure/README.md) на этой основе и пройти весь процесс производства данных, обучения модели, сжатия, логического вывода и развертывания.

<div align="center">
    <img src="https://user-images.githubusercontent.com/50011306/196963669-f53b0ee5-3cb4-481c-b73c-97c4b3e2efb8.png">
</div>

## ⚡ Быстрый опыт

```bash
pip3 install paddlepaddle # for gpu user please install paddlepaddle-gpu
pip3 install paddleocr
paddleocr --image_dir /your/test/image.jpg --lang=ru
```

> Если у вас нет среды Python, выполните [Подготовка среды](../version2.x/ppocr/environment.en.md). Мы рекомендуем вам начать с [Учебники](#Tutorials).

<a name="книга"></a>

## 📚 Электронная книга: *Погружение в OCR*

- [Погружение в распознавание символов](../version2.x/ppocr/blog/ocr_book.en.md)

<a name="Сообщество"></a>

## 👫 Сообщество

Что касается международных разработчиков, мы рассматриваем [Обсуждения PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/discussions) как нашу платформу для международного сообщества. Все идеи и вOCRосы можно обсудить здесь на английском языке.

<a name="Список-поддерживаемых-китайских-моделей"></a>

## 🛠️ Список моделей серии ПП -OCR

| Введение модели | Название модели | Рекомендуемая сцена | Модель обнаружения | Модель распознавания |
| ------------------------------------------------------------ | ---------------------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Ру́сский язы́к：Ру́сский язы́к Сверхлегкая модель PP-OCRv3 (13.4M) | cyrillic_PP-OCRv3_xx | Мобильный и сервер |[модель вывода](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar)/[обученный модель](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_distill_train.tar) | [модель вывода](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar)/[обученный модель](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_train.tar)  |
| Английский сверхлегкая модель PP-OCRv3 (13,4 Мб) | en_PP-OCRv3_xx |Мобильный и сервер | [модель вывода](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) / [обученный модель](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar) | [вывод модель](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv3_mobile_rec_infer.tar) / [обученный модель](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar) |
| Сверхлегкая китайская и английская модель PP-OCRv3 (16,2M) | ch_PP-OCRv3_xx | Мобильный и сервер | [вывод модель](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_mobile_det_infer.tar) / [обученный модель](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_det_pretrained.pdparams) | [вывод модель](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_mobile_rec_infer.tar) / [обученный модель](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar) |

- Для получения дополнительных загрузок моделей (включая несколько языков) см. [Загрузки моделей серии ПП-OCR](../version2.x/ppocr/model_list.en.md).
- Для запроса нового языка см [Руководство для новых языковых_запросов](#language_requests).
- Модели структурного анализа документов см [PP-Structure модельs](../version2.x/ppstructure/models_list.en.md).

<a name=" Учебники "></a>

## 📖 Учебники

- [Подготовка окружающей среды](../version2.x/ppocr/environment.en.md)
- [PP-OCR 🔥](../version2.x/ppocr/overview.en.md)

    - [Быстрый старт](../version2.x/ppocr/quick_start.en.md)
    - [Модель Zoo](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.9/doc/doc_en/models_en.md)
    - [Модель тренировки](../version2.x/ppocr/model_train/training.en.md)
    - [Обнаружение текста](../version2.x/ppocr/model_train/detection.en.md)
        - [Распознавание текста](../version2.x/ppocr/model_train/recognition.en.md)
        - [Классификация направления текста](../version2.x/ppocr/model_train/angle_class.en.md)
    - Модель Сжатие
        - [Модель квантования](../../deploy/slim/quantization/README_en.md)
        - [Модель Обрезка](../../deploy/slim/prune/README_en.md)
        - [Дистилляция знаний](../version2.x/ppocr/model_compress/knowledge_distillation.en.md)
    - [Вывод и развертывание](../../deploy/README.md)
        - [Python Вывод](../version2.x/legacy/python_infer.en.md)
        - [Вывод C++](../version2.x/legacy/cpp_infer.en.md)
        - [Подача](../version2.x/legacy/paddle_server.en.md)
        - [Мобильный](../../deploy/lite/readme.md)
        - [Paddle2ONNX](../../deploy/paddle2onnx/readme.md)
        - [ВеслоОблако](../../deploy/paddlecloud/README.md)
        - [Benchmark](../version2.x/legacy/benchmark.en.md)
- [PP-Structure 🔥](../../ppstructure/README.md)
    - [Быстрый старт](../version2.x/ppstructure/quick_start.en.md)
        - [Модель Zoo](../version2.x/ppstructure/models_list.en.md)
        - [Модель тренировки](../version2.x/ppocr/model_train/training.en.md)
    - [Анализ макета](../../ppstructure/layout/README.md)
        - [Распознавание таблиц](../../ppstructure/table/README.md)
        - [Извлечение ключевой информации](../../ppstructure/kie/README.md)
    - [Вывод и развертывание](../../deploy/README.md)
        - [Вывод Python](../version2.x/ppstructure/infer_deploy/python_infer.en.md)
        - [Вывод С++](../version2.x/legacy/cpp_infer.en.md)
        - [Обслуживание](../../deploy/hubserving/readme_en.md)
- [Академические алгоритмы](../version2.x/algorithm/overview.en.md)
    - [Обнаружение текста](../version2.x/algorithm/overview.en.md)
- [Распознавание текста](../version2.x/algorithm/overview.en.md)
    - [Непрерывной цепью OCR](../version2.x/algorithm/overview.en.md)
    - [Распознавание таблиц](../version2.x/algorithm/overview.en.md)
    - [Извлечение ключевой информации](../version2.x/algorithm/overview.en.md)
    - [Добавьте новые алгоритмы в PaddleOCR](../version2.x/algorithm/add_new_algorithm.en.md)
- Аннотации и синтез данных
    - [Полуавтоматический инструмент аннотации данных: метка ППOCRR](https://github.com/PFCCLab/PPOCRLabel/blob/main/README.md)
    - [Инструмент синтеза данных: Стиль-текст](https://github.com/PFCCLab/StyleText/blob/main/README.md)
    - [Другие инструменты аннотирования данных](../data_anno_synth/data_annotation.en.md)
    - [Другие инструменты синтеза данных](../data_anno_synth/data_synthesis.en.md)
- Наборы данных
    - [Общие наборы данных OCR (китайский/английский)](../datasets/datasets.en.md)
    - [Наборы данных Рукописный/*OCR* наборы данных (китайский)](../datasets/handwritten_datasets.en.md)
    - [Различные наборы данных OCR (многоязычные)](../datasets/vertical_and_multilingual_datasets.en.md)
    - [Анализ макета](../datasets/layout_datasets.en.md)
    - [Распознавание таблиц](../datasets/table_datasets.en.md)
    - [Извлечение ключевой информации](../datasets/kie_datasets.en.md)
- [Структура кода](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.9/doc/doc_en/tree_en.md)
- [Визуализация](#Visualization)
- [Сообщество](#Community)
- [Новые языковые запросы](#language_requests)
- [ЧАСТО ЗАДАВАЕМЫЕ ВOCRОСЫ](../FAQ.en.md)
- [Использованная литература](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.9/doc/doc_en/reference_en.md)
- [ЛИЦЕНЗИЯ](#LICENSE)

<a name="language_requests"></a>

## 🇺🇳 Руководство по запросам на новый язык

Если вы хотите **запросить новую языковую модель**, проголосуйте в [Голосуйте за обновление многоязычной модели](https://github.com/PaddlePaddle/PaddleOCR/discussions/7253). Мы будем регулярно обновлять модель по результату. **Пригласите друзей проголосовать вместе!**

Если вам нужно **обучить новую языковую модель** на основе вашего сценария, учебное пособие в [Проекте обучения многоязычной модели](https://github.com/PaddlePaddle/PaddleOCR/discussions/7252) поможет вам подготовить набор данных и показать вам весь процесс шаг за шагом.

Оригинальный [Многоязычный план разработки OCR](https://github.com/PaddlePaddle/PaddleOCR/issues/1048) по-прежнему показывает вам много полезных корпусов и словарей.

<a name=" Визуализация "></a>

## 👀 Визуализация [больше](../version2.x/ppocr/visualization.en.md)

<details open>
<summary>PP-OCRv3 Многоязычная модель </summary>
<div align="center">
    <img src="../version2.x/ppocr/images/PP-OCRv3/multi_lang/japan_2.jpg" width="800">
    <img src="../version2.x/ppocr/images/PP-OCRv3/multi_lang/korean_1.jpg" width="800">
</div>
</details>

<details open>
<summary>PP-OCRv3 Aнглийская модель </summary>
<div align="center">
    <img src="../version2.x/ppocr/images/PP-OCRv3/en/en_1.png" width="800">
    <img src="../version2.x/ppocr/images/PP-OCRv3/en/en_2.png" width="800">
</div>
</details>
<details open>
<summary>PP-OCRv3 Kитайская модель </summary>
<div align="center">
    <img src="../version2.x/ppocr/images/PP-OCRv3/ch/PP-OCRv3-pic001.jpg" width="800">
    <img src="../version2.x/ppocr/images/PP-OCRv3/ch/PP-OCRv3-pic002.jpg" width="800">
    <img src="../version2.x/ppocr/images/PP-OCRv3/ch/PP-OCRv3-pic003.jpg" width="800">
</div>
</details>

<details open>
<summary>PP-Structurev2</summary>
1. анализ макета + распознавание таблиц
<div align="center">
    <img src="../version2.x/ppstructure/images/ppstructure.gif" width="800">
</div>
2. SER (Семантическое распознавание объектов)
<div align="center">
    <img src="https://user-images.githubusercontent.com/25809855/186094456-01a1dd11-1433-4437-9ab2-6480ac94ec0a.png" width="600">
</div>
<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185310636-6ce02f7c-790d-479f-b163-ea97a5a04808.jpg" width="600">
</div>
<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185539517-ccf2372a-f026-4a7c-ad28-c741c770f60a.png" width="600">
</div>
3. RE (Извлечение отношений)
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

<a name="ЛИЦЕНЗИЯ"></a>

## 📄 Лицензия

Этот проект выпущен под <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/{{PADDLEOCR_GITHUB_REF}}/LICENSE">Apache 2.0 license</a>
