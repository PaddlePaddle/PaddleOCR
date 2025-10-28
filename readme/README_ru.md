<div align="center">
  <p>
      <img width="100%" src="../docs/images/Banner.png" alt="Баннер PaddleOCR">
  </p>

<!-- language -->
[English](../README.md) | [简体中文](./README_cn.md) | [繁體中文](./README_tcn.md) | [日本語](./README_ja.md) | [한국어](./README_ko.md) | [Français](./README_fr.md) | Русский | [Español](./README_es.md) | [العربية](./README_ar.md)

<!-- icon -->

[![stars](https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf)](https://github.com/PaddlePaddle/PaddleOCR)
[![Downloads](https://img.shields.io/pypi/dm/paddleocr)](https://pypi.org/project/PaddleOCR/)
![python](https://img.shields.io/badge/python-3.8～3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
![hardware](https://img.shields.io/badge/hardware-cpu%2C%20gpu%2C%20xpu%2C%20npu-yellow.svg)


[![HuggingFace](https://img.shields.io/badge/PaddleOCR--VL-_Demo_on_HuggingFace-yellow?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAF8AAABYCAMAAACkl9t/AAAAk1BMVEVHcEz/nQv/nQv/nQr/nQv/nQr/nQv/nQv/nQr/wRf/txT/pg7/yRr/rBD/zRz/ngv/oAz/zhz/nwv/txT/ngv/0B3+zBz/nQv/0h7/wxn/vRb/thXkuiT/rxH/pxD/ogzcqyf/nQvTlSz/czCxky7/SjifdjT/Mj3+Mj3wMj15aTnDNz+DSD9RTUBsP0FRO0Q6O0WyIxEIAAAAGHRSTlMADB8zSWF3krDDw8TJ1NbX5efv8ff9/fxKDJ9uAAAGKklEQVR42u2Z63qjOAyGC4RwCOfB2JAGqrSb2WnTw/1f3UaWcSGYNKTdf/P+mOkTrE+yJBulvfvLT2A5ruenaVHyIks33npl/6C4s/ZLAM45SOi/1FtZPyFur1OYofBX3w7d54Bxm+E8db+nDr12ttmESZ4zludJEG5S7TO72YPlKZFyE+YCYUJTBZsMiNS5Sd7NlDmKM2Eg2JQg8awbglfqgbhArjxkS7dgp2RH6hc9AMLdZYUtZN5DJr4molC8BfKrEkPKEnEVjLbgW1fLy77ZVOJagoIcLIl+IxaQZGjiX597HopF5CkaXVMDO9Pyix3AFV3kw4lQLCbHuMovz8FallbcQIJ5Ta0vks9RnolbCK84BtjKRS5uA43hYoZcOBGIG2Epbv6CvFVQ8m8loh66WNySsnN7htL58LNp+NXT8/PhXiBXPMjLSxtwp8W9f/1AngRierBkA+kk/IpUSOeKByzn8y3kAAAfh//0oXgV4roHm/kz4E2z//zRc3/lgwBzbM2mJxQEa5pqgX7d1L0htrhx7LKxOZlKbwcAWyEOWqYSI8YPtgDQVjpB5nvaHaSnBaQSD6hweDi8PosxD6/PT09YY3xQA7LTCTKfYX+QHpA0GCcqmEHvr/cyfKQTEuwgbs2kPxJEB0iNjfJcCTPyocx+A0griHSmADiC91oNGVwJ69RudYe65vJmoqfpul0lrqXadW0jFKH5BKwAeCq+Den7s+3zfRJzA61/Uj/9H/VzLKTx9jFPPdXeeP+L7WEvDLAKAIoF8bPTKT0+TM7W8ePj3Rz/Yn3kOAp2f1Kf0Weony7pn/cPydvhQYV+eFOfmOu7VB/ViPe34/EN3RFHY/yRuT8ddCtMPH/McBAT5s+vRde/gf2c/sPsjLK+m5IBQF5tO+h2tTlBGnP6693JdsvofjOPnnEHkh2TnV/X1fBl9S5zrwuwF8NFrAVJVwCAPTe8gaJlomqlp0pv4Pjn98tJ/t/fL++6unpR1YGC2n/KCoa0tTLoKiEeUPDl94nj+5/Tv3/eT5vBQ60X1S0oZr+IWRR8Ldhu7AlLjPISlJcO9vrFotky9SpzDequlwEir5beYAc0R7D9KS1DXva0jhYRDXoExPdc6yw5GShkZXe9QdO/uOvHofxjrV/TNS6iMJS+4TcSTgk9n5agJdBQbB//IfF/HpvPt3Tbi7b6I6K0R72p6ajryEJrENW2bbeVUGjfgoals4L443c7BEE4mJO2SpbRngxQrAKRudRzGQ8jVOL2qDVjjI8K1gc3TIJ5KiFZ1q+gdsARPB4NQS4AjwVSt72DSoXNyOWUrU5mQ9nRYyjp89Xo7oRI6Bga9QNT1mQ/ptaJq5T/7WcgAZywR/XlPGAUDdet3LE+qS0TI+g+aJU8MIqjo0Kx8Ly+maxLjJmjQ18rA0YCkxLQbUZP1WqdmyQGJLUm7VnQFqodmXSqmRrdVpqdzk5LvmvgtEcW8PMGdaS23EOWyDVbACZzUJPaqMbjDxpA3Qrgl0AikimGDbqmyT8P8NOYiqrldF8rX+YN7TopX4UoHuSCYY7cgX4gHwclQKl1zhx0THf+tCAUValzjI7Wg9EhptrkIcfIJjA94evOn8B2eHaVzvBrnl2ig0So6hvPaz0IGcOvTHvUIlE2+prqAxLSQxZlU2stql1NqCCLdIiIN/i1DBEHUoElM9dBravbiAnKqgpi4IBkw+utSPIoBijDXJipSVV7MpOEJUAc5Qmm3BnUN+w3hteEieYKfRZSIUcXKMVf0u5wD4EwsUNVvZOtUT7A2GkffHjByWpHqvRBYrTV72a6j8zZ6W0DTE86Hn04bmyWX3Ri9WH7ZU6Q7h+ZHo0nHUAcsQvVhXRDZHChwiyi/hnPuOsSEF6Exk3o6Y9DT1eZ+6cASXk2Y9k+6EOQMDGm6WBK10wOQJCBwren86cPPWUcRAnTVjGcU1LBgs9FURiX/e6479yZcLwCBmTxiawEwrOcleuu12t3tbLv/N4RLYIBhYexm7Fcn4OJcn0+zc+s8/VfPeddZHAGN6TT8eGczHdR/Gts1/MzDkThr23zqrVfAMFT33Nx1RJsx1k5zuWILLnG/vsH+Fv5D4NTVcp1Gzo8AAAAAElFTkSuQmCC&labelColor=white)](https://huggingface.co/spaces/PaddlePaddle/PaddleOCR-VL_Online_Demo)
[![AI Studio](https://img.shields.io/badge/PaddleOCR--VL-_Demo_on_AI_Studio-1927BA?logo=data:image/png;base64,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&labelColor=white)](https://aistudio.baidu.com/application/detail/98365)
[![ModelScope](https://img.shields.io/badge/PaddleOCR--VL-_Demo_on_ModelScope-purple?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIzIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCiA8Zz4KICA8dGl0bGU+TGF5ZXIgMTwvdGl0bGU+CiAgPHBhdGggaWQ9InN2Z18xNCIgZmlsbD0iIzYyNGFmZiIgZD0ibTAsODkuODRsMjUuNjUsMGwwLDI1LjY0OTk5bC0yNS42NSwwbDAsLTI1LjY0OTk5eiIvPgogIDxwYXRoIGlkPSJzdmdfMTUiIGZpbGw9IiM2MjRhZmYiIGQ9Im05OS4xNCwxMTUuNDlsMjUuNjUsMGwwLDI1LjY1bC0yNS42NSwwbDAsLTI1LjY1eiIvPgogIDxwYXRoIGlkPSJzdmdfMTYiIGZpbGw9IiM2MjRhZmYiIGQ9Im0xNzYuMDksMTQxLjE0bC0yNS42NDk5OSwwbDAsMjIuMTlsNDcuODQsMGwwLC00Ny44NGwtMjIuMTksMGwwLDI1LjY1eiIvPgogIDxwYXRoIGlkPSJzdmdfMTciIGZpbGw9IiMzNmNmZDEiIGQ9Im0xMjQuNzksODkuODRsMjUuNjUsMGwwLDI1LjY0OTk5bC0yNS42NSwwbDAsLTI1LjY0OTk5eiIvPgogIDxwYXRoIGlkPSJzdmdfMTgiIGZpbGw9IiMzNmNmZDEiIGQ9Im0wLDY0LjE5bDI1LjY1LDBsMCwyNS42NWwtMjUuNjUsMGwwLC0yNS42NXoiLz4KICA8cGF0aCBpZD0ic3ZnXzE5IiBmaWxsPSIjNjI0YWZmIiBkPSJtMTk4LjI4LDg5Ljg0bDI1LjY0OTk5LDBsMCwyNS42NDk5OWwtMjUuNjQ5OTksMGwwLC0yNS42NDk5OXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIwIiBmaWxsPSIjMzZjZmQxIiBkPSJtMTk4LjI4LDY0LjE5bDI1LjY0OTk5LDBsMCwyNS42NWwtMjUuNjQ5OTksMGwwLC0yNS42NXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIxIiBmaWxsPSIjNjI0YWZmIiBkPSJtMTUwLjQ0LDQybDAsMjIuMTlsMjUuNjQ5OTksMGwwLDI1LjY1bDIyLjE5LDBsMCwtNDcuODRsLTQ3Ljg0LDB6Ii8+CiAgPHBhdGggaWQ9InN2Z18yMiIgZmlsbD0iIzM2Y2ZkMSIgZD0ibTczLjQ5LDg5Ljg0bDI1LjY1LDBsMCwyNS42NDk5OWwtMjUuNjUsMGwwLC0yNS42NDk5OXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIzIiBmaWxsPSIjNjI0YWZmIiBkPSJtNDcuODQsNjQuMTlsMjUuNjUsMGwwLC0yMi4xOWwtNDcuODQsMGwwLDQ3Ljg0bDIyLjE5LDBsMCwtMjUuNjV6Ii8+CiAgPHBhdGggaWQ9InN2Z18yNCIgZmlsbD0iIzYyNGFmZiIgZD0ibTQ3Ljg0LDExNS40OWwtMjIuMTksMGwwLDQ3Ljg0bDQ3Ljg0LDBsMCwtMjIuMTlsLTI1LjY1LDBsMCwtMjUuNjV6Ii8+CiA8L2c+Cjwvc3ZnPg==&labelColor=white)](https://www.modelscope.cn/studios/PaddlePaddle/PaddleOCR-VL_Online_Demo)

[![AI Studio](https://img.shields.io/badge/PP--OCRv5-Demo_on_AI_Studio-1927BA?logo=data:image/png;base64,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&labelColor=white)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP--StructureV3-Demo_on_AI_Studio-1927BA?logo=data:image/png;base64,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&labelColor=white)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP--ChatOCRv4-Demo_on_AI_Studio-1927BA?logo=data:image/png;base64,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&labelColor=white)](https://aistudio.baidu.com/community/app/518493/webUI)

</div>

## 🚀 Введение
С момента своего первого выпуска PaddleOCR получил широкое признание в академических, промышленных и исследовательских кругах благодаря своим передовым алгоритмам и доказанной производительности в реальных приложениях. Он уже используется в таких популярных проектах с открытым исходным кодом, как Umi-OCR, OmniParser, MinerU и RAGFlow, что делает его предпочтительным инструментарием OCR для разработчиков по всему миру.

20 мая 2025 года команда PaddlePaddle представила PaddleOCR 3.0, полностью совместимый с официальным выпуском фреймворка **PaddlePaddle 3.0**. Это обновление еще больше **повышает точность распознавания текста**, добавляет поддержку **распознавания нескольких типов текста** и **распознавания рукописного текста**, а также удовлетворяет растущий спрос на приложения с большими моделями для **высокоточного анализа сложных документов**. В сочетании с **ERNIE 4.5** он значительно улучшает точность извлечения ключевой информации. PaddleOCR 3.0 также вводит поддержку китайских гетерогенных AI ускорителей, таких как **KUNLUNXIN** и **Ascend**. Для получения полной документации по использованию, пожалуйста, обратитесь к [Документации PaddleOCR 3.0](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html).

Три новые ключевые функции в PaddleOCR 3.0:
- Универсальная модель распознавания текста в любых сценах [PP-OCRv5](../docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.en.md): Одна модель обрабатывает пять различных типов текста и сложный рукописный ввод. Общая точность распознавания увеличилась на 13 процентных пунктов по сравнению с предыдущим поколением. [Онлайн-демо](https://aistudio.baidu.com/community/app/91660/webUI)

- Общее решение для парсинга документов [PP-StructureV3](../docs/version3.x/algorithm/PP-StructureV3/PP-StructureV3.en.md): Обеспечивает высокоточный парсинг PDF-файлов с различными макетами и сценариями, превосходя многие решения с открытым и закрытым исходным кодом по результатам публичных тестов. [Онлайн-демо](https://aistudio.baidu.com/community/app/518494/webUI)

- Интеллектуальное решение для понимания документов [PP-ChatOCRv4](../docs/version3.x/algorithm/PP-ChatOCRv4/PP-ChatOCRv4.en.md): Нативно поддерживается большим моделью ERNIE 4.5, достигая на 15 процентных пунктов более высокой точности, чем его предшественник. [Онлайн-демо](https://aistudio.baidu.com/community/app/518493/webUI)

Помимо предоставления выдающейся библиотеки моделей, PaddleOCR 3.0 также предлагает удобные инструменты, охватывающие обучение моделей, инференс и развертывание сервисов, чтобы разработчики могли быстро внедрять ИИ-приложения в производство.
<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Arch.jpg" alt="PaddleOCR Architecture">
  </p>
</div>

**Особое примечание**: PaddleOCR 3.x вводит ряд значительных изменений в интерфейс. **Старый код, написанный на основе PaddleOCR 2.x, скорее всего, несовместим с PaddleOCR 3.x**. Убедитесь, что документация, которую вы читаете, соответствует версии PaddleOCR, которую вы используете. [В этом документе](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/upgrade_notes.html) объясняются причины обновления и основные изменения при переходе с PaddleOCR 2.x на 3.x.

## 📣 Последние обновления

#### **🔥🔥 16.10.2025: Выпущен PaddleOCR 3.3.0**, включает:

**Выпуск PaddleOCR-VL:**
- **Введение в модель:**
    - PaddleOCR-VL — это современная, эффективная с точки зрения ресурсов модель для анализа документов. В ее основе лежит PaddleOCR-VL-0.9B — компактная и мощная мульти-модальная (визуально-языковая, VLM) модель, которая интегрирует визуальный энкодер с динамическим разрешением по типу NaViT и языковую модель ERNIE-4.5-0.3B, обеспечивая точное распознавание элементов. Эта инновационная модель поддерживает 109 языков и превосходно справляется с распознаванием сложных элементов (текст, таблицы, формулы, графики и др.), минимизируя потребление ресурсов. Комплексные оценки на публичных и внутренних тестах показывают, что PaddleOCR-VL достигает лучших результатов (SOTA) как в анализе документов на уровне страницы, так и на уровне отдельных элементов, значительно превосходя существующие решения. Она также конкурирует с ведущими VLM и обеспечивает быструю скорость инференса, что делает ее очень пригодной для практического применения. Модель доступна для скачивания и использования на [HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL).

- **Основные характеристики:**
    - Компактная и мощная визуально-языковая архитектура: предлагает новую модель, эффективную по ресурсам и специально оптимизированную для быстрых инференсов и отличного распознавания элементов. Совмещает визуальный энкодер высокой динамической четкости по типу NaViT и легкую языковую модель ERNIE-4.5-0.3B, значительно повышая возможности распознавания и эффективность декодирования. Такая интеграция снижает вычислительные затраты при сохранении высокой точности, что идеально подходит для практических и эффективных задач обработки документов.
    - SOTA-результаты в анализе документов: PaddleOCR-VL достигает передовых результатов как на уровне страницы, так и на уровне элементов. Значительно превосходит традиционные пайплайновые решения и эффективно конкурирует с ведущими визуально-языковыми моделями (VLM). Отлично справляется с распознаванием сложных элементов — текста, таблиц, формул, графиков, а также способен обрабатывать сложный контент, например, рукописный текст или исторические документы. Это обеспечивает высокую универсальность модели для различных типов и сценариев документов.
    - Многоязычная поддержка: PaddleOCR-VL поддерживает 109 языков, включая основные мировые языки (китайский, английский, японский, латинский, корейский и т.д.), а также русский (кириллица), арабский, хинди (деванагари), тайский и другие системы письма и языковые структуры. Такая широкая языковая поддержка особенно ценна для глобальной и многоязычной обработки документов.

**Выпуск многоязычной модели PP-OCRv5:**
- Улучшена точность и расширен диапазон распознавания латинских символов, добавлена поддержка кириллицы, арабского, деванагари, телугу, тамильского и других языков. Всего поддерживается 109 языков; размер модели всего 2 миллиона параметров, а в некоторых случаях точность увеличена более чем на 40% по сравнению с предыдущим поколением.

#### **🔥🔥2025.08.21: Выпуск PaddleOCR 3.2.0**, включает:

- **Ключевые обновления моделей:**
    - Добавлены обучение, инференс и развертывание моделей распознавания PP-OCRv5 для английского, тайского и греческого языков. **Английская модель PP-OCRv5 обеспечивает прирост точности на 11% в англоязычных сценариях по сравнению с основной моделью PP-OCRv5; точность тайской модели составляет 82,68%, греческой — 89,28%.**

- **Улучшения возможностей развертывания:**
    - **Полная поддержка PaddlePaddle 3.1.0 и 3.1.1.**
    - **Полное обновление решения для локального развертывания на C++: поддержка Linux и Windows, идентичная функциональность и точность Python-версии.**
    - **Высокопроизводительный инференс с поддержкой CUDA 12, возможностью использования Paddle Inference или ONNX Runtime как backend.**
    - **Полностью открытый исходный код для высокостабильного сервисного развертывания, позволяющий пользователям настраивать Docker-образы и SDK под свои нужды.**
    - Сервисное решение также поддерживает вызовы через вручную сформированные HTTP-запросы, что позволяет реализовывать клиент на любом языке программирования.

- **Поддержка бенчмаркинга:**
    - **Все производственные цепочки теперь поддерживают тонкую бенчмаркинг-аналитику: измерение времени инференса end-to-end, а также времени работы по слоям и модулям для анализа производительности.[Вот](../docs/version3.x/pipeline_usage/instructions/benchmark.en.md), как настроить и использовать функцию бенчмарка**
    - **В документации приведены ключевые показатели (время инференса, использование памяти и др.) для различных конфигураций на популярных аппаратных платформах, чтобы помочь пользователям с развертыванием.**

- **Исправления ошибок:**
    - Исправлена проблема, из-за которой не сохранялись логи при обучении моделей.
    - Обновлены методы аугментации данных для формульных моделей для совместимости с новыми версиями зависимости albumentations, а также устранено предупреждение о deadlock при использовании библиотеки tokenizers в многопроцессном режиме.
    - Исправлено неконсистентное поведение некоторых флагов (например, `use_chart_parsing`) в конфигурационных файлах PP-StructureV3 в сравнении с другими производственными цепочками.

- **Другие улучшения:**
    - **Разделены основные и дополнительные зависимости: для базового распознавания достаточно минимального набора зависимостей; для расширенных функций (разбор документов, извлечение информации и др.) дополнительные зависимости можно устанавливать по мере необходимости.**
    - **Добавлена поддержка видеокарт серии NVIDIA 50 под Windows. Пользователи могут обратиться к [инструкции по установке](../docs/version3.x/installation.en.md) для выбора подходящей версии Paddle.**
    - **Модели серии PP-OCR теперь могут возвращать координаты каждого отдельного символа.**
    - Добавлены новые источники загрузки моделей, такие как AIStudio и ModelScope, с возможностью выбора нужного источника.
    - Добавлена поддержка модуля преобразования графиков в таблицы (PP-Chart2Table).
    - Оптимизированы отдельные описания в документации для повышения удобства использования.


#### **2025.08.15: Выпуск PaddleOCR 3.1.1**, включает:

- **Исправления ошибок:**
  - Добавлены недостающие методы `save_vector`, `save_visual_info_list`, `load_vector`, `load_visual_info_list` в класс `PP-ChatOCRv4`.
  - Добавлены недостающие параметры `glossary` и `llm_request_interval` в метод `translate` класса `PPDocTranslation`.

- **Улучшение документации:**
  - Добавлены демонстрационные примеры (demo) в документацию MCP.
  - Указаны используемые версии фреймворка PaddlePaddle и PaddleOCR для замера производительности в документации.
  - Исправлены ошибки и недочёты в переводе документации.

- **Другое:**
  - Изменены зависимости сервера MCP: вместо `python-magic` теперь используется чисто питоновская библиотека `puremagic`, чтобы уменьшить проблемы с установкой.
  - Производительность PP-OCRv5 была заново протестирована с PaddleOCR версии 3.1.0, документация обновлена.

#### **2025.06.29: Выпуск PaddleOCR 3.1.0**, включает:

- **Основные модели и пайплайны:**
  - **Добавлена многоязычная модель распознавания текста PP-OCRv5**, поддерживающая обучение и инференс для 37 языков, включая французский, испанский, португальский, русский, корейский и др. **Средняя точность увеличилась более чем на 30%.** [Подробнее](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)
  - Обновлена **модель PP-Chart2Table** в PP-StructureV3, что еще больше улучшило преобразование графиков в таблицы. На внутренних пользовательских тестах метрика (RMS-F1) **увеличилась на 9,36 процентных пункта (71,24% -> 80,60%).**
  - Запущен новый **конвейер перевода документов PP-DocTranslation на основе PP-StructureV3 и ERNIE 4.5**, поддерживающий перевод документов в формате Markdown, различных PDF-документов со сложной версткой и изображений документов, с сохранением результата в формате Markdown. [Подробнее](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/PP-DocTranslation.html)

- **Новый сервер MCP:** [Details](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/mcp_server.html)
  - **Поддерживает как OCR, так и конвейеры PP-StructureV3.**
  - Поддерживаются три режима работы: локальная библиотека Python, облачный сервис сообщества AIStudio и самостоятельный хостинг.
  - Поддерживается вызов локальных сервисов через stdio и удалённых сервисов через Streamable HTTP.

- **Оптимизация документации:** Улучшены описания в некоторых руководствах пользователя для более комфортного чтения.



<details>
    <summary><strong>История обновлений</strong></summary>

#### **2025.06.26: Релиз PaddleOCR 3.0.3, включает:**

- Исправление ошибки: Исправлена проблема, из-за которой параметр `enable_mkldnn` не действовал, восстановлено поведение использования MKL-DNN для вывода на CPU по умолчанию.

#### 🔥🔥**2025.06.19: Релиз PaddleOCR 3.0.2, включает:**

- **Новые возможности:**
  - Источник загрузки по умолчанию изменен с `BOS` на `HuggingFace`. Пользователи также могут изменить переменную окружения `PADDLE_PDX_MODEL_SOURCE` на `BOS`, чтобы установить источник загрузки моделей обратно на Baidu Object Storage (BOS).
  - Добавлены примеры вызова сервисов для шести языков — C++, Java, Go, C#, Node.js и PHP — для пайплайнов, таких как PP-OCRv5, PP-StructureV3 и PP-ChatOCRv4.
  - Улучшен алгоритм сортировки разделов макета в пайплайне PP-StructureV3, улучшена логика сортировки для сложных вертикальных макетов для достижения лучших результатов.
  - Улучшена логика выбора модели: когда указан язык, но не указана версия модели, система автоматически выберет последнюю версию модели, поддерживающую этот язык.
  - Установлен верхний предел по умолчанию для размера кэша MKL-DNN для предотвращения неограниченного роста, а также предоставлена пользователям возможность настраивать емкость кэша.
  - Обновлены конфигурации по умолчанию для высокопроизводительного инференса для поддержки ускорения Paddle MKL-DNN и оптимизирована логика автоматического выбора конфигурации для более разумного выбора.
  - Скорректирована логика получения устройства по умолчанию для учета фактической поддержки вычислительных устройств установленным фреймворком Paddle, что делает поведение программы более интуитивным.
  - Добавлен пример для Android для PP-OCRv5. [Подробности](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/on_device_deployment.html).

- **Исправления ошибок:**
  - Исправлена проблема с некоторыми параметрами CLI в PP-StructureV3, которые не вступали в силу.
  - Решена проблема, из-за которой `export_paddlex_config_to_yaml` в некоторых случаях работала некорректно.
  - Исправлено несоответствие между фактическим поведением `save_path` и его описанием в документации.
  - Исправлены потенциальные ошибки многопоточности при использовании MKL-DNN в базовом развертывании сервиса.
  - Исправлены ошибки порядка каналов в предварительной обработке изображений для модели Latex-OCR.
  - Исправлены ошибки порядка каналов при сохранении визуализированных изображений в модуле распознавания текста.
  - Решены ошибки порядка каналов в визуализированных результатах таблиц в пайплайне PP-StructureV3.
  - Исправлена проблема переполнения при вычислении `overlap_ratio` в крайне особых обстоятельствах в пайплайне PP-StructureV3.

- **Улучшения документации:**
  - Обновлено описание параметра `enable_mkldnn` в документации, чтобы оно точно отражало фактическое поведение программы.
  - Исправлены ошибки в документации, касающиеся параметров `lang` и `ocr_version`.
  - Добавлены инструкции по экспорту файлов конфигурации производственной линии через CLI.
  - Исправлены отсутствующие столбцы в таблице данных о производительности для PP-OCRv5.
  - Уточнены метрики бенчмарков для PP-StructureV3 для различных конфигураций.

- **Прочее:**
  - Ослаблены ограничения версий для зависимостей, таких как numpy и pandas, восстановлена поддержка Python 3.12.

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
   2. 💻 Нативная поддержка **ERNIE 4.5**, с совместимостью для развертывания больших моделей через PaddleNLP, Ollama, vLLM и другие.
   3. 🤝 Интегрирован [PP-DocBee2](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee2), обеспечивающий извлечение и понимание печатного текста, рукописного текста, печатей, таблиц, диаграмм и других общих элементов в сложных документах.

[История обновлений](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)

</details>

## ⚡ Быстрый старт
### 1. Запустить онлайн-демо
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

### 2. Установка

Установите PaddlePaddle, следуя [Руководству по установке](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html), после чего установите инструментарий PaddleOCR.


```bash
# Если вы хотите использовать только базовую функцию распознавания текста (возвращает координаты и содержание текста, включая серию PP-OCR)
python -m pip install paddleocr
# Если вы хотите использовать все функции, такие как анализ документов, понимание документов, перевод документов, извлечение ключевой информации и т.д.
# python -m pip install "paddleocr[all]"
```

Начиная с версии 3.2.0, помимо вышеупомянутой группы зависимостей `all`, PaddleOCR также поддерживает указание других групп зависимостей для установки некоторых дополнительных функций. Все группы зависимостей, предоставляемые PaddleOCR, перечислены ниже.

| Название группы зависимостей | Соответствующая функция |
| - | - |
| `doc-parser` | Документный парсинг: может использоваться для извлечения элементов макета, таких как таблицы, формулы, штампы, изображения и т.д. из документов; включает такие модели, как PP-StructureV3, PaddleOCR-VL. |
| `ie` | Извлечение информации: может извлекать ключевую информацию, такую как имена, даты, адреса, суммы и т.д. из документов. Включая модели, такие как PP-ChatOCRv4 |
| `trans` | Перевод документов: может переводить документы на другие языки. Включая модели, такие как PP-DocTranslation |
| `all` | Все функции |

### 3. Запуск инференса через CLI
```bash
# Запустить инференс PP-OCRv5
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False  

# Запустить инференс PP-StructureV3
paddleocr pp_structurev3 -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png --use_doc_orientation_classify False --use_doc_unwarping False

# Сначала получите Qianfan API Key, а затем запустите инференс PP-ChatOCRv4
paddleocr pp_chatocrv4_doc -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png -k 驾驶室准乘人数 --qianfan_api_key your_api_key --use_doc_orientation_classify False --use_doc_unwarping False 

# Запустить инференс PaddleOCR-VL
paddleocr doc_parser -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png

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

<details>
   <summary><strong>4.4 Пример для PaddleOCR-VL</strong></summary>

```python
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL()
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png")
for res in output:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")
```

</details>

### 5. Китайские гетерогенные ИИ-ускорители
- [Huawei Ascend](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/other_devices_support/paddlepaddle_install_NPU.html)
- [KUNLUNXIN](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/other_devices_support/paddlepaddle_install_XPU.html)

## 🧩 Дополнительные функции

- Конвертация моделей в формат ONNX: [Получение моделей ONNX](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/obtaining_onnx_models.html).
- Ускорение вывода с использованием движков, таких как OpenVINO, ONNX Runtime, TensorRT, или выполнение вывода с моделями в формате ONNX: [Высокопроизводительный вывод](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/high_performance_inference.html).
- Ускорение вывода с использованием нескольких GPU и многопроцессности: [Параллельный вывод для конвейеров](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/instructions/parallel_inference.html).
- Интегрируйте PaddleOCR в приложения, написанные на C++, C#, Java и других языках: [Сервис](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/serving.html).

## ⛰️ Продвинутые руководства
- [Руководство по PP-OCRv5](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/OCR.html)
- [Руководство по PP-StructureV3](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-StructureV3.html)
- [Руководство по PP-ChatOCRv4](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-ChatOCRv4.html)
- [Руководство по PaddleOCR-VL](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PaddleOCR-VL.html)

## 🔄 Краткий обзор результатов выполнения

<div align="center">
  <p>
     <img width="100%" src="../docs/images/demo.gif" alt="Демо PP-OCRv5">
  </p>
</div>

<div align="center">
  <p>
     <img width="100%" src="../docs/images/blue_v3.gif" alt="Демо PP-StructureV3">
  </p>
</div>

## 🌟 Не пропустите самые свежие новости

⭐ **Поставьте звезду этому репозиторию, чтобы быть в курсе захватывающих обновлений и новых релизов — включая мощные функции OCR и анализа документов!** ⭐

<div align="center">
  <p>
       <img width="1200" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/star_paddleocr.en.gif" alt="Star-Project">
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
| [pathway](https://github.com/pathwaycom/pathway) <a href="https://github.com/pathwaycom/pathway"><img src="https://img.shields.io/github/stars/pathwaycom/pathway"></a>|Python ETL-фреймворк для потоковой обработки, анализа в реальном времени, конвейеров LLM и RAG|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|Инструмент для преобразования документов различных типов в Markdown|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|Бесплатное офлайн-программное обеспечение для пакетного OCR с открытым исходным кодом.|
| [cherry-studio](https://github.com/CherryHQ/cherry-studio) <a href="https://github.com/CherryHQ/cherry-studio"><img src="https://img.shields.io/github/stars/CherryHQ/cherry-studio"></a>|Настольный клиент, поддерживающий нескольких поставщиков LLM.|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |Инструмент парсинга экрана для GUI-агента, основанного исключительно на компьютерном зрении.|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |Система вопросов и ответов на основе любого контента.|
| [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) <a href="https://github.com/opendatalab/PDF-Extract-Kit"><img src="https://img.shields.io/github/stars/opendatalab/PDF-Extract-Kit"></a>|Мощный инструментарий с открытым исходным кодом, предназначенный для эффективного извлечения высококачественного контента из сложных и разнообразных PDF-документов.|
| [Dango-Translator](https://github.com/PantsuDango/Dango-Translator)<a href="https://github.com/PantsuDango/Dango-Translator"><img src="https://img.shields.io/github/stars/PantsuDango/Dango-Translator"></a> |Распознает текст на экране, переводит его и отображает результаты перевода в режиме реального времени.|
| [Узнать больше о проектах](../awesome_projects.md) | [Больше проектов на основе PaddleOCR](../awesome_projects.md)|

## 👩‍👩‍👧‍👦 Контрибьюторы

<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>


## 🌟 Star

[![Star History Chart](https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date)](https://star-history.com/#PaddlePaddle/PaddleOCR&Date)


## 📄 Лицензия
Этот проект выпущен под [лицензией Apache 2.0](LICENSE).

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
```
