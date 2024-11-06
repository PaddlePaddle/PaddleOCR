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
      <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
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
    - Выпускать [PP-Structurev2](./ppstructure/)，с полностью обновленными функциями и производительностью, адаптированными для китайских сцен и новой поддержкой pаспознавание таблиц
     [Восстановление макета](./ppstructure/recovery) и **однострочная команда для преобразования PDF в Word**;
    - [Анализ макета](./ppstructure/layout) оптимизация: память модели уменьшена на 95%, а скорость увеличена в 11 раз, а среднее время процессорного времени составляет всего 41 мс;
    - [Распознавание таблиц](./ppstructure/table) оптимизация: разработано 3 стратегии оптимизации, а точность модели улучшена на 6% при сопоставимых затратах времени;
    - [Извлечение ключевой информации](./ppstructure/kie) оптимизация: разработана визуально независимая структура модели, точность распознавания семантической сущности увеличена на 2,8%, а точность извлечения отношения увеличена на 9,1%.
- **🔥2022.7 Выпуск [Коллекция приложений сцены OCR](../../applications/README_en.md)**
- Выпуск **9 вертикальных моделей**, таких как цифровая трубка, ЖК-экран, номерной знак, модель распознавания рукописного ввода, высокоточная модель SVTR и т. д., охватывающих основные вертикальные приложения OCR в целом, производственной, финансовой и транспортной отраслях.
- **🔥2022.5.9 Выпуск PaddleOCR [Выпуск /2.5](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.5)**
- Выпускать [PP-OCRv3](../doc_en/ppocr_introduction_en.md#pp-ocrv3): При сопоставимой скорости эффект китайской сцены улучшен на 5% по сравнению с ПП-OCRRv2, эффект английской сцены улучшен на 11%, а средняя точность распознавания 80 языковых многоязычных моделей улучшена более чем на 5%.
- Выпускать [PPOCRLabelv2](https://github.com/PFCCLab/PPOCRLabel/blob/main/README.md): Добавьте функцию аннотации для задачи распознавания таблиц, задачи извлечения ключевой информации и изображения неправильного текста.
    - Выпустить интерактивную электронную книгу [*"Погружение в OCR"*](../doc_en/ocr_book_en.md), охватывает передовую теорию и практику кодирования технологии полного стека OCR.
- [подробнее](../doc_en/update_en.md)

## 🌟 Функции

PaddleOCR поддерживает множество передовых алгоритмов, связанных с распознаванием текста, и разработала промышленные модели/решения. [PP-OCR](../doc_en/ppocr_introduction_en.md) и [PP-Structure](./ppstructure/README.md) на этой основе и пройти весь процесс производства данных, обучения модели, сжатия, логического вывода и развертывания.

<div align="center">
    <img src="https://user-images.githubusercontent.com/50011306/196963669-f53b0ee5-3cb4-481c-b73c-97c4b3e2efb8.png">
</div>

## ⚡ Быстрый опыт

```bash
pip3 install paddlepaddle # for gpu user please install paddlepaddle-gpu
pip3 install paddleocr
paddleocr --image_dir /your/test/image.jpg --lang=ru
```

> Если у вас нет среды Python, выполните [Подготовка среды](../doc_en/environment_en.md). Мы рекомендуем вам начать с [Учебники](#Tutorials).

<a name="книга"></a>

## 📚 Электронная книга: *Погружение в OCR*

- [Погружение в распознавание символов](../doc_en/ocr_book_en.md)

<a name="Сообщество"></a>

## 👫 Сообщество

Что касается международных разработчиков, мы рассматриваем [Обсуждения PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/discussions) как нашу платформу для международного сообщества. Все идеи и вOCRосы можно обсудить здесь на английском языке.

<a name="Список-поддерживаемых-китайских-моделей"></a>

## 🛠️ Список моделей серии ПП -OCR

| Введение модели | Название модели | Рекомендуемая сцена | Модель обнаружения | Модель распознавания |
| ------------------------------------------------------------ | ---------------------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Ру́сский язы́к：Ру́сский язы́к Сверхлегкая модель ПП-OCRv3 (13.4M) | cyrillic_PP-OCRv3_xx | Мобильный и сервер |[модель вывода](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar)/[обученный модель](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_distill_train.tar) | [модель вывода](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar)/[обученный модель](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_train.tar)  |
| Английский сверхлегкая модель ПП-OCRv3 (13,4 Мб) | en\_ПП-OCRv3_xx |Мобильный и сервер | [модель вывода](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) / [обученный модель](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar) | [вывод модель](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) / [обученный модель](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar) |
| Сверхлегкая китайская и английская модель ПП-OCRv3 (16,2M) | ch\_ПП-OCRv3_xx | Мобильный и сервер | [вывод модель](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) / [обученный модель](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar) | [вывод модель](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) / [обученный модель](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar) |

- Для получения дополнительных загрузок моделей (включая несколько языков) см. [Загрузки моделей серии ПП-OCR](../doc_en/models_list_en.md).
- Для запроса нового языка см [Руководство для новых языковых_запросов](#language_requests).
- Модели структурного анализа документов см [PP-Structure модельs](./ppstructure/docs/модельs_list_en.md).

<a name=" Учебники "></a>

## 📖 Учебники

- [Подготовка окружающей среды](../doc_en/environment_en.md)
- [PP-OCR 🔥](../doc_en/ppocr_introduction_en.md)

    - [Быстрый старт](../doc_en/quickstart_en.md)
    - [Модель Zoo](../doc_en/модельs_en.md)
    - [Модель тренировки](../doc_en/training_en.md)
    - [Обнаружение текста](../doc_en/detection_en.md)
        - [Распознавание текста](../doc_en/recognition_en.md)
        - [Классификация направления текста](../doc_en/angle_class_en.md)
    - Модель Сжатие
        - [Модель квантования](./deploy/slim/quantization/README_en.md)
        - [Модель Обрезка](./deploy/slim/prune/README_en.md)
        - [Дистилляция знаний](../doc_en/knowledge_distillation_en.md)
    - [Вывод и развертывание](./deploy/README.md)
        - [Python Вывод](../doc_en/ inference_ppocr_en.md)
        - [Вывод C++](./deploy/cpp_infer/readme.md)
        -[Подача](./deploy/pdserving/README.md)
        - [Мобильный](./deploy/lite/readme.md)
        - [Paddle2ONNX](./deploy/paddle2onnx/readme.md)
        -[ВеслоОблако](./deploy/paddlecloud/README.md)
        - [Benchmark](../doc_en/benchmark_en.md)
- [PP-Structure 🔥](../../ppstructure/README.md)
    - [Быстрый старт](../../ppstructure/docs/quickstart_en.md)
        - [Модель Zoo](../../ppstructure/docs/models_list_en.md)
        - [Модель тренировки](../doc_en/training_en.md)
    - [Анализ макета](../../ppstructure/layout/README.md)
        - [Распознавание таблиц](../../ppstructure/table/README.md)
        - [Извлечение ключевой информации](../../ppstructure/kie/README.md)
    - [Вывод и развертывание](./deploy/README.md)
        - [Вывод Python](../../ppstructure/docs/inference_en.md)
        - [Вывод С++](../../deploy/cpp_infer/readme.md)
        - [Обслуживание](../../deploy/hubserving/readme_en.md)
- [Академические алгоритмы](../doc_en/algorithm_overview_en.md)
    - [Обнаружение текста](../doc_en/algorithm_overview_en.md)
- [Распознавание текста](../doc_en/algorithm_overview_en.md)
    - [Непрерывной цепью OCR](../doc_en/algorithm_overview_en.md)
    - [Распознавание таблиц](../doc_en/algorithm_overview_en.md)
    - [Извлечение ключевой информации](../doc_en/algorithm_overview_en.md)
    - [Добавьте новые алгоритмы в PaddleOCR](../doc_en/add_new_algorithm_en.md)
- Аннотации и синтез данных
    - [Полуавтоматический инструмент аннотации данных: метка ППOCRR](https://github.com/PFCCLab/PPOCRLabel/blob/main/README.md)
    - [Инструмент синтеза данных: Стиль-текст](https://github.com/PFCCLab/StyleText/blob/main/README.md)
    - [Другие инструменты аннотирования данных](../doc_en/data_annotation_en.md)
    - [Другие инструменты синтеза данных](../doc_en/data_synthesis_en.md)
- Наборы данных
    - [Общие наборы данных OCR (китайский/английский)](../doc_en/dataset/datasets_en.md)
    - [Наборы данных Рукописный/*OCR* наборы данных (китайский)](../doc_en/dataset/handwritten_datasets_en.md)
    - [Различные наборы данных OCR (многоязычные)](../doc_en/dataset/vertical_and_multilingual_datasets_en.md)
    - [Анализ макета](../doc_en/dataset/layout_datasets_en.md)
    - [Распознавание таблиц](../doc_en/dataset/table_datasets_en.md)
    - [Извлечение ключевой информации](../doc_en/dataset/kie_datasets_en.md)
- [Структура кода](../doc_en/tree_en.md)
- [Визуализация](#Visualization)
- [Сообщество](#Community)
- [Новые языковые запросы](#language_requests)
- [ЧАСТО ЗАДАВАЕМЫЕ ВOCRОСЫ](../doc_en/FAQ_en.md)
- [Использованная литература](../doc_en/reference_en.md)
- [ЛИЦЕНЗИЯ](#LICENSE)

<a name="language_requests"></a>

## 🇺🇳 Руководство по запросам на новый язык

Если вы хотите **запросить новую языковую модель**, проголосуйте в [Голосуйте за обновление многоязычной модели](https://github.com/PaddlePaddle/PaddleOCR/discussions/7253). Мы будем регулярно обновлять модель по результату. **Пригласите друзей проголосовать вместе!**

Если вам нужно **обучить новую языковую модель** на основе вашего сценария, учебное пособие в [Проекте обучения многоязычной модели](https://github.com/PaddlePaddle/PaddleOCR/discussions/7252) поможет вам подготовить набор данных и показать вам весь процесс шаг за шагом.

Оригинальный [Многоязычный план разработки OCR](https://github.com/PaddlePaddle/PaddleOCR/issues/1048) по-прежнему показывает вам много полезных корпусов и словарей.

<a name=" Визуализация "></a>

## 👀 Визуализация [больше](../doc_en/visualization_en.md)

<details open>
<summary>PP-OCRv3 Многоязычная модель </summary>
<div align="center">
    <img src="../imgs_results/PP-OCRv3/multi_lang/japan_2.jpg" width="800">
    <img src="../imgs_results/PP-OCRv3/multi_lang/korean_1.jpg" width="800">
</div>
</details>

<details open>
<summary>PP-OCRv3 Aнглийская модель </summary>
<div align="center">
    <img src="../imgs_results/PP-OCRv3/en/en_1.png" width="800">
    <img src="../imgs_results/PP-OCRv3/en/en_2.png" width="800">
</div>
</details>
<details open>
<summary>PP-OCRv3 Kитайская модель </summary>
<div align="center">
    <img src="../imgs_results/PP-OCRv3/ch/PP-OCRv3-pic001.jpg" width="800">
    <img src="../imgs_results/PP-OCRv3/ch/PP-OCRv3-pic002.jpg" width="800">
    <img src="../imgs_results/PP-OCRv3/ch/PP-OCRv3-pic003.jpg" width="800">
</div>
</details>

<details open>
<summary>PP-Structurev2</summary>
1. анализ макета + распознавание таблиц
<div align="center">
    <img src="../../ppstructure/docs/table/ppstructure.GIF" width="800">
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

Этот проект выпущен под <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>
