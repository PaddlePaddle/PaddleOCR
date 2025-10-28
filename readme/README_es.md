<div align="center">
  <p>
      <img width="100%" src="../docs/images/Banner.png" alt="Banner de PaddleOCR">
  </p>

<!-- language -->
[English](../README.md) | [简体中文](./README_cn.md) | [繁體中文](./README_tcn.md) | [日本語](./README_ja.md) | [한국어](./README_ko.md) | [Français](./README_fr.md) | [Русский](./README_ru.md) | Español | [العربية](./README_ar.md)

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

## 🚀 Introducción
Desde su lanzamiento inicial, PaddleOCR ha sido ampliamente aclamado en las comunidades académica, industrial y de investigación, gracias a sus algoritmos de vanguardia y su rendimiento probado en aplicaciones del mundo real. Ya está impulsando proyectos populares de código abierto como Umi-OCR, OmniParser, MinerU y RAGFlow, convirtiéndose en el conjunto de herramientas de OCR de referencia para desarrolladores de todo el mundo.

El 20 de mayo de 2025, el equipo de PaddlePaddle presentó PaddleOCR 3.0, totalmente compatible con la versión oficial del framework **PaddlePaddle 3.0**. Esta actualización **aumenta aún más la precisión en el reconocimiento de texto**, añade soporte para el **reconocimiento de múltiples tipos de texto** y el **reconocimiento de escritura a mano**, y satisface la creciente demanda de las aplicaciones de grandes modelos para el **análisis (parsing) de alta precisión de documentos complejos**. En combinación con **ERNIE 4.5**, mejora significativamente la precisión en la extracción de información clave. Para la documentación de uso completa, consulte la [Documentación de PaddleOCR 3.0](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html).

Tres nuevas características principales en PaddleOCR 3.0:
- Modelo de Reconocimiento de Texto en Escenarios Universales [PP-OCRv5](../docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.en.md): Un único modelo que maneja cinco tipos de texto diferentes además de escritura a mano compleja. La precisión general de reconocimiento ha aumentado en 13 puntos porcentuales con respecto a la generación anterior. [Demo en línea](https://aistudio.baidu.com/community/app/91660/webUI)

- Solución de Análisis General de Documentos [PP-StructureV3](../docs/version3.x/algorithm/PP-StructureV3/PP-StructureV3.en.md): Ofrece un análisis de alta precisión de PDF con múltiples diseños y escenas, superando a muchas soluciones de código abierto y cerrado en benchmarks públicos. [Demo en línea](https://aistudio.baidu.com/community/app/518494/webUI)

- Solución de Comprensión Inteligente de Documentos [PP-ChatOCRv4](../docs/version3.x/algorithm/PP-ChatOCRv4/PP-ChatOCRv4.en.md): Impulsado nativamente por el gran modelo ERNIE 4.5, logrando una precisión 15 puntos porcentuales mayor que su predecesor. [Demo en línea](https://aistudio.baidu.com/community/app/518493/webUI)

Además de proporcionar una excelente biblioteca de modelos, PaddleOCR 3.0 también ofrece herramientas fáciles de usar que cubren el entrenamiento de modelos, la inferencia y el despliegue de servicios, para que los desarrolladores puedan llevar rápidamente las aplicaciones de IA a producción.
<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Arch.jpg" alt="PaddleOCR Architecture">
  </p>
</div>

**Nota especial**: PaddleOCR 3.x introduce varios cambios significativos en la interfaz. **Es probable que el código antiguo escrito basado en PaddleOCR 2.x no sea compatible con PaddleOCR 3.x**. Asegúrese de que la documentación que está leyendo coincida con la versión de PaddleOCR que está utilizando. [Este documento](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/upgrade_notes.html) explica las razones de la actualización y los principales cambios de PaddleOCR 2.x a 3.x.

## 📣 Últimas actualizaciones

#### **🔥🔥 16.10.2025: PaddleOCR 3.3.0 lanzado**, incluye:

**Lanzamiento de PaddleOCR-VL:**
- **Introducción al modelo:**
    - PaddleOCR-VL es un modelo de última generación (SOTA) y eficiente en recursos, especializado en el análisis de documentos. Su núcleo, PaddleOCR-VL-0.9B, es un modelo compacto y potente de visión-lenguaje (VLM) que integra un codificador visual de resolución dinámica al estilo NaViT y un modelo de lenguaje ERNIE-4.5-0.3B, logrando un reconocimiento preciso de elementos. Este innovador modelo admite 109 idiomas y es excelente en el reconocimiento de elementos complejos (texto, tablas, fórmulas, gráficos, etc.), minimizando el consumo de recursos. Las evaluaciones exhaustivas en benchmarks públicos y privados demuestran que PaddleOCR-VL alcanza el rendimiento SOTA tanto en el análisis de documentos a nivel de página como en el reconocimiento a nivel de elemento, superando ampliamente a las soluciones existentes. También compite con los mejores VLM y ofrece una velocidad de inferencia rápida, lo que lo hace muy adecuado para su aplicación práctica. El modelo está disponible para descarga y uso en [HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL).

- **Características principales:**
    - Arquitectura compacta y potente de visión-lenguaje: ofrece un nuevo modelo eficiente en recursos especialmente diseñado para la inferencia y excelente en el reconocimiento de elementos. Combina un codificador visual de alta resolución dinámico al estilo NaViT y un modelo de lenguaje ERNIE-4.5-0.3B ligero, mejorando significativamente la capacidad de reconocimiento y la eficiencia de decodificación. Esta integración reduce el coste computacional y mantiene una alta precisión, siendo ideal para aplicaciones de procesamiento de documentos eficientes y prácticas.
    - Rendimiento SOTA en el análisis de documentos: PaddleOCR-VL logra un rendimiento de vanguardia tanto en el análisis a nivel de página como en el reconocimiento a nivel de elemento. Supera ampliamente a las soluciones basadas en pipelines tradicionales y compite eficazmente con los principales modelos de visión-lenguaje (VLM). Es excelente en el reconocimiento de elementos complejos como texto, tablas, fórmulas y gráficos, y es capaz de manejar contenidos desafiantes como texto manuscrito o documentos históricos. Esto le otorga una gran versatilidad, adaptándose a múltiples tipos y escenarios de documentos.
    - Soporte multilingüe: PaddleOCR-VL admite 109 idiomas, incluyendo los principales idiomas mundiales (chino, inglés, japonés, latín, coreano, etc.), así como ruso (cirílico), árabe, hindi (devanagari), tailandés y otros sistemas de escritura y estructuras lingüísticas. Este amplio soporte multilingüe es una gran ventaja para el procesamiento de documentos a nivel global y multilingüe.

**Lanzamiento del modelo multilingüe PP-OCRv5:**
- Mejora la precisión y el rango de reconocimiento de caracteres latinos, y añade soporte para cirílico, árabe, devanagari, telugu, tamil, entre otros. En total, cubre el reconocimiento de 109 idiomas; el modelo tiene solo 2 millones de parámetros y, en algunos casos, la precisión mejora más de un 40% respecto a la generación anterior.

#### **🔥🔥2025.08.21: Lanzamiento de PaddleOCR 3.1.1**, incluye:

- **Actualización de los modelos principales:**
    - Se añaden funciones de entrenamiento, inferencia y despliegue para los modelos de reconocimiento PP-OCRv5 en inglés, tailandés y griego. **El modelo en inglés logra una mejora del 11% en precisión en comparación con la versión anterior de PP-OCRv5 en escenarios en inglés; el modelo en tailandés alcanza una precisión del 82,68% y el griego del 89,28%.**

- **Mejoras en las capacidades de despliegue:**
    - **Soporte completo para PaddlePaddle 3.1.0 y 3.1.1.**
    - **Reforma completa de la solución de despliegue local en C++, compatible con Linux y Windows, alcanzando la misma funcionalidad y precisión que la versión en Python.**
    - **Soporte para CUDA 12 para inferencia de alto rendimiento, con opción de usar los backends Paddle Inference u ONNX Runtime.**
    - **Apertura total del código fuente de la solución de despliegue tipo servicio de alta estabilidad, permitiendo a los usuarios personalizar imágenes de Docker o SDK según sus necesidades.**
    - El despliegue tipo servicio de alta estabilidad también soporta llamadas HTTP manuales, lo que permite a los clientes implementar en cualquier lenguaje.

- **Soporte de benchmarks:**
    - **Se proporciona una función detallada de benchmark en toda la cadena de producción, permitiendo medir el tiempo de inferencia de extremo a extremo y los tiempos de ejecución de diferentes capas y módulos, facilitando el análisis de rendimiento.[Aquí](../docs/version3.x/pipeline_usage/instructions/benchmark.en.md) se explica cómo configurar y utilizar la función de prueba de rendimiento (benchmark).**
    - **La documentación incluye valores de referencia (tiempo de inferencia, uso de memoria, etc.) en las principales plataformas de hardware para ayudar a los usuarios a tomar decisiones de despliegue.**

- **Corrección de errores:**
    - Corrección del problema por el cual no se guardaban los registros durante el entrenamiento del modelo.
    - Adaptación de la parte de aumento de datos del modelo matemático a la nueva versión de albumentations, y solución de la advertencia de posible deadlock al utilizar tokenizers en multiproceso.
    - Corrección de las inconsistencias en el comportamiento de banderas como `use_chart_parsing` en el archivo de configuración de PP-StructureV3 respecto a otras producciones.

- **Otras actualizaciones:**
    - **Separación de dependencias obligatorias y opcionales; las funciones básicas de reconocimiento requieren solo las dependencias mínimas, mientras que funciones adicionales como análisis de documentos o extracción de información pueden instalarse según necesidad.**
    - **Soporte para GPU serie 50 de NVIDIA en entorno Windows, consulte la [guía de instalación](../docs/version3.x/installation.en.md) para elegir la versión de Paddle adecuada.**
    - **Los modelos de la serie PP-OCR ahora pueden devolver las coordenadas de cada carácter.**
    - Se añaden fuentes de descarga de modelos como AIStudio y ModelScope, permitiendo su selección.
    - Soporte para la inferencia del módulo de conversión de gráficos a tablas (PP-Chart2Table).
    - Optimización de algunas descripciones en la documentación para mejorar la facilidad de uso.

#### **2025.08.15: Lanzamiento de PaddleOCR 3.1.1**, incluye:

- **Corrección de errores:**
  - Se añadieron los métodos que faltaban `save_vector`, `save_visual_info_list`, `load_vector` y `load_visual_info_list` a la clase `PP-ChatOCRv4`.
  - Se añadieron los parámetros faltantes `glossary` y `llm_request_interval` al método `translate` de la clase `PPDocTranslation`.

- **Optimización de la documentación:**
  - Se añadió una demostración de ejemplo a la documentación de MCP.
  - Se detallaron las versiones del framework PaddlePaddle y de PaddleOCR utilizadas en las pruebas de indicadores de rendimiento.
  - Se corrigieron errores y omisiones en la documentación sobre la línea de producción de traducción de documentos.

- **Otros:**
  - Cambios en las dependencias del servidor MCP: se utilizó la biblioteca pura de Python `puremagic` en lugar de `python-magic` para reducir problemas de instalación.
  - Se volvieron a probar los indicadores de rendimiento de PP-OCRv5 con la versión 3.1.0 de PaddleOCR y se actualizó la documentación.

#### **2025.06.29: Lanzamiento de PaddleOCR 3.1.0**, incluye:

- **Modelos y flujos de trabajo clave:**
  - **Añadido el modelo de reconocimiento de texto multilingüe PP-OCRv5**, que soporta entrenamiento e inferencia para modelos de reconocimiento de texto en 37 idiomas, incluidos francés, español, portugués, ruso, coreano, etc. **Precisión media mejorada en más de un 30%.** [Detalles](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)
  - Actualizado el **modelo PP-Chart2Table** en PP-StructureV3, mejorando aún más la conversión de gráficos a tablas. En conjuntos de evaluación personalizados internos, la métrica (RMS-F1) **aumentó 9,36 puntos porcentuales (71,24% -> 80,60%).**
  - Nuevo **flujo de traducción de documentos, PP-DocTranslation, basado en PP-StructureV3 y ERNIE 4.5**, que soporta la traducción de documentos en formato Markdown, diversos PDF de diseño complejo e imágenes de documentos, guardando los resultados en formato Markdown. [Detalles](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/PP-DocTranslation.html)

- **Nuevo servidor MCP:** [Details](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/mcp_server.html)
  - **Admite tanto OCR como los flujos de trabajo de PP-StructureV3.**
  - Soporta tres modos de trabajo: biblioteca local de Python, servicio en la nube de la comunidad AIStudio y servicio autohospedado.
  - Permite invocar servicios locales a través de stdio y servicios remotos a través de Streamable HTTP.

- **Optimización de la documentación:** Se han mejorado las descripciones en algunas guías de usuario para una experiencia de lectura más fluida.


<details>
    <summary><strong>Historial de actualizaciones</strong></summary>

#### 🔥🔥**2025.06.26: Lanzamiento de PaddleOCR 3.0.3, incluye:**

- Corrección de error: Se resolvió el problema donde el parámetro `enable_mkldnn` no era efectivo, restaurando el comportamiento predeterminado de usar MKL-DNN para la inferencia en CPU.

#### 🔥🔥**2025.06.19: Lanzamiento de PaddleOCR 3.0.2, incluye:**

- **Nuevas características:**

  - La fuente de descarga predeterminada se ha cambiado de `BOS` a `HuggingFace`. Los usuarios también pueden cambiar la variable de entorno `PADDLE_PDX_MODEL_SOURCE` a `BOS` para volver a establecer la fuente de descarga del modelo en Baidu Object Storage (BOS).
  - Se agregaron ejemplos de invocación de servicios para seis idiomas (C++, Java, Go, C#, Node.js y PHP) para pipelines como PP-OCRv5, PP-StructureV3 y PP-ChatOCRv4.
  - Se mejoró el algoritmo de ordenación de particiones de diseño en el pipeline PP-StructureV3, mejorando la lógica de ordenación para diseños verticales complejos para ofrecer mejores resultados.
  - Lógica de selección de modelo mejorada: cuando se especifica un idioma pero no una versión del modelo, el sistema seleccionará automáticamente la última versión del modelo que admita ese idioma.
  - Se estableció un límite superior predeterminado para el tamaño de la caché de MKL-DNN para evitar un crecimiento ilimitado, al tiempo que se permite a los usuarios configurar la capacidad de la caché.
  - Se actualizaron las configuraciones predeterminadas para la inferencia de alto rendimiento para admitir la aceleración de Paddle MKL-DNN y se optimizó la lógica para la selección automática de configuración para elecciones más inteligentes.
  - Se ajustó la lógica para obtener el dispositivo predeterminado para considerar el soporte real de los dispositivos de computación por parte del framework Paddle instalado, lo que hace que el comportamiento del programa sea más intuitivo.
  - Añadido ejemplo de Android para PP-OCRv5. [Detalles](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/on_device_deployment.html).

- **Corrección de errores:**

  - Se solucionó un problema con algunos parámetros de CLI en PP-StructureV3 que no tenían efecto.
  - Se resolvió un problema por el cual `export_paddlex_config_to_yaml` no funcionaba correctamente en ciertos casos.
  - Se corrigió la discrepancia entre el comportamiento real de `save_path` y la descripción de su documentación.
  - Se corrigieron posibles errores de subprocesos múltiples al usar MKL-DNN en la implementación de servicios básicos.
  - Se corrigieron errores en el orden de los canales en el preprocesamiento de imágenes para el modelo Latex-OCR.
  - Se corrigieron errores en el orden de los canales al guardar imágenes visualizadas dentro del módulo de reconocimiento de texto.
  - Se resolvieron errores de orden de canales en los resultados de tablas visualizadas dentro del pipeline de PP-StructureV3.
  - Se solucionó un problema de desbordamiento en el cálculo de `overlap_ratio` en circunstancias extremadamente especiales en el pipeline PP-StructureV3.

- **Mejoras en la documentación:**

  - Se actualizó la descripción del parámetro `enable_mkldnn` en la documentación para reflejar con precisión el comportamiento real del programa.
  - Se corrigieron errores en la documentación con respecto a los parámetros `lang` y `ocr_version`.
  - Se agregaron instrucciones para exportar archivos de configuración de la línea de producción a través de CLI.
  - Se corrigieron las columnas que faltaban en la tabla de datos de rendimiento para PP-OCRv5.
  - Se refinaron las métricas de referencia para PP-StructureV3 en diferentes configuraciones.

- **Otros:**

  - Se flexibilizaron las restricciones de versión en dependencias como numpy y pandas, restaurando el soporte para Python 3.12.

#### **🔥🔥 2025.06.05: Lanzamiento de PaddleOCR 3.0.1, incluye:**

- **Optimización de ciertos modelos y configuraciones de modelos:**
  - Actualizada la configuración de modelo por defecto para PP-OCRv5, cambiando tanto la detección como el reconocimiento de modelos `mobile` a `server`. Para mejorar el rendimiento por defecto en la mayoría de los escenarios, el parámetro `limit_side_len` en la configuración ha sido cambiado de 736 a 64.
  - Añadido un nuevo modelo de clasificación de orientación de línea de texto `PP-LCNet_x1_0_textline_ori` con una precisión del 99.42%. El clasificador de orientación de línea de texto por defecto para los pipelines de OCR, PP-StructureV3 y PP-ChatOCRv4 ha sido actualizado a este modelo.
  - Optimizado el modelo de clasificación de orientación de línea de texto `PP-LCNet_x0_25_textline_ori`, mejorando la precisión en 3.3 puntos porcentuales hasta una precisión actual del 98.85%.

- **Optimizaciones y correcciones de algunos problemas en la versión 3.0.0, [detalles](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)**

🔥🔥2025.05.20: Lanzamiento oficial de **PaddleOCR v3.0**, incluyendo:
- **PP-OCRv5**: Modelo de Reconocimiento de Texto de Alta Precisión para Todos los Escenarios - Texto Instantáneo desde Imágenes/PDFs.
   1. 🌐 Soporte en un único modelo para **cinco** tipos de texto - Procese sin problemas **Chino Simplificado, Chino Tradicional, Pinyin de Chino Simplificado, Inglés** y **Japonés** dentro de un solo modelo.
   2. ✍️ **Reconocimiento de escritura a mano** mejorado: Significativamente mejor en escritura cursiva compleja y caligrafía no estándar.
   3. 🎯 **Ganancia de precisión de 13 puntos** sobre PP-OCRv4, alcanzando un rendimiento de vanguardia (state-of-the-art) en una variedad de escenarios del mundo real.

- **PP-StructureV3**: Solución de Análisis de Documentos de Propósito General – ¡Libere el poder del análisis SOTA de Imágenes/PDFs para escenarios del mundo real!
   1. 🧮 **Análisis de PDF multiescena de alta precisión**, liderando tanto a las soluciones de código abierto como a las de código cerrado en el benchmark OmniDocBench.
   2. 🧠 Capacidades especializadas que incluyen **reconocimiento de sellos**, **conversión de gráficos a tablas**, **reconocimiento de tablas con fórmulas/imágenes anidadas**, **análisis de documentos de texto vertical** y **análisis de estructuras de tablas complejas**.

- **PP-ChatOCRv4**: Solución Inteligente de Comprensión de Documentos – Extraiga Información Clave, no solo texto de Imágenes/PDFs.
   1. 🔥 **Ganancia de precisión de 15 puntos** en la extracción de información clave en archivos PDF/PNG/JPG con respecto a la generación anterior.
   2. 💻 Soporte nativo para **ERNIE 4.5**, con compatibilidad para despliegues de modelos grandes a través de PaddleNLP, Ollama, vLLM y más.
   3. 🤝 Integrado con [PP-DocBee2](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee2), permitiendo la extracción y comprensión de texto impreso, escritura a mano, sellos, tablas, gráficos y otros elementos comunes en documentos complejos.

[Historial de actualizaciones](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)

</details>

## ⚡ Inicio rápido
### 1. Ejecutar demo en línea
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

### 2. Instalación

Instale PaddlePaddle consultando la [Guía de Instalación](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html), y después, instale el toolkit de PaddleOCR.

```bash
# Si solo deseas utilizar la función básica de reconocimiento de texto (devuelve las coordenadas de posición y el contenido del texto), incluyendo la serie PP-OCR
python -m pip install paddleocr
# Si deseas utilizar todas las funciones como análisis de documentos, comprensión de documentos, traducción de documentos, extracción de información clave, etc.
# python -m pip install "paddleocr[all]"
```

A partir de la versión 3.2.0, además del grupo de dependencias `all` mostrado arriba, PaddleOCR también permite instalar algunas funciones opcionales especificando otros grupos de dependencias. Todos los grupos de dependencias que proporciona PaddleOCR son los siguientes:

| Nombre del grupo de dependencias | Funcionalidad correspondiente |
| - | - |
| `doc-parser` | Análisis de documentos: se puede utilizar para extraer elementos de diseño como tablas, fórmulas, sellos, imágenes, etc. de los documentos; incluye modelos como PP-StructureV3 y PaddleOCR-VL. |
| `ie` | Extracción de información: se puede usar para extraer información clave de los documentos, como nombres, fechas, direcciones, montos, etc.; incluye modelos como PP-ChatOCRv4 |
| `trans` | Traducción de documentos: se puede usar para traducir documentos de un idioma a otro; incluye modelos como PP-DocTranslation |
| `all` | Funcionalidad completa |

### 3. Ejecutar inferencia por CLI
```bash
# Ejecutar inferencia de PP-OCRv5
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False  

# Ejecutar inferencia de PP-StructureV3
paddleocr pp_structurev3 -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png --use_doc_orientation_classify False --use_doc_unwarping False

# Obtenga primero la API Key de Qianfan y luego ejecute la inferencia de PP-ChatOCRv4
paddleocr pp_chatocrv4_doc -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png -k 驾驶室准乘人数 --qianfan_api_key your_api_key --use_doc_orientation_classify False --use_doc_unwarping False 

# Ejecutar inferencia de PaddleOCR-VL
paddleocr doc_parser -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png

# Obtener más información sobre "paddleocr ocr"
paddleocr ocr --help
```

### 4. Ejecutar inferencia por API
**4.1 Ejemplo de PP-OCRv5**
```python
from paddleocr import PaddleOCR
# Inicializar la instancia de PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# Ejecutar inferencia de OCR en una imagen de ejemplo
result = ocr.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png")

# Visualizar los resultados y guardar los resultados en JSON
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

<details>
    <summary><strong>4.2 Ejemplo de PP-StructureV3</strong></summary>

```python
from pathlib import Path
from paddleocr import PPStructureV3

pipeline = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
)

# Para Imagen
output = pipeline.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png",
)

# Visualizar los resultados y guardar los resultados en JSON
for res in output:
    res.print() 
    res.save_to_json(save_path="output") 
    res.save_to_markdown(save_path="output")           
```

</details>

<details>
   <summary><strong>4.3 Ejemplo de PP-ChatOCRv4</strong></summary>

```python
from paddleocr import PPChatOCRv4Doc

chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "ernie-3.5-8k",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key",  # su api_key
}

retriever_config = {
    "module_name": "retriever",
    "model_name": "embedding-v1",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "qianfan",
    "api_key": "api_key",  # su api_key
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
# Si se utiliza un modelo grande multimodal, es necesario iniciar el servicio mllm local. Puede consultar la documentación: https://github.com/PaddlePaddle/PaddleX/blob/release/3.0/docs/pipeline_usage/tutorials/vlm_pipelines/doc_understanding.en.md para realizar el despliegue y actualizar la configuración de mllm_chat_bot_config.
if use_mllm:
    mllm_chat_bot_config = {
        "module_name": "chat_bot",
        "model_name": "PP-DocBee",
        "base_url": "http://127.0.0.1:8080/",  # la URL de su servicio mllm local
        "api_type": "openai",
        "api_key": "api_key",  # su api_key
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
   <summary><strong>4.4 Ejemplo de PaddleOCR-VL</strong></summary>

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

## 🧩 Más funciones

- Convertir modelos al formato ONNX: [Obtener modelos ONNX](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/obtaining_onnx_models.html).
- Acelerar la inferencia usando motores como OpenVINO, ONNX Runtime, TensorRT, o realizar inferencia usando modelos en formato ONNX: [Inferencia de alto rendimiento](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/high_performance_inference.html).
- Acelerar la inferencia usando múltiples GPU y múltiples procesos: [Inferencia paralela para pipelines](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/instructions/parallel_inference.html).
- Integra PaddleOCR en aplicaciones escritas en C++, C#, Java, etc.: [Servicio](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/serving.html).

## ⛰️ Tutoriales avanzados
- [Tutorial de PP-OCRv5](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/OCR.html)
- [Tutorial de PP-StructureV3](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-StructureV3.html)
- [Tutorial de PP-ChatOCRv4](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-ChatOCRv4.html)
- [Tutorial de PaddleOCR-VL](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PaddleOCR-VL.html)

## 🔄 Vista rápida de los resultados de ejecución

<div align="center">
  <p>
     <img width="100%" src="../docs/images/demo.gif" alt="Demo de PP-OCRv5">
  </p>
</div>

<div align="center">
  <p>
     <img width="100%" src="../docs/images/blue_v3.gif" alt="Demo de PP-StructureV3">
  </p>
</div>

## 🌟 Mantente Atento

⭐ **Dale una estrella a este repositorio para estar al tanto de emocionantes actualizaciones y nuevos lanzamientos, ¡incluyendo potentes capacidades de OCR y análisis de documentos!** ⭐

<div align="center">
  <p>
       <img width="1200" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/star_paddleocr.en.gif" alt="Star-Project">
  </p>
</div>

## 👩‍👩‍👧‍👦 Comunidad

| Cuenta oficial de PaddlePaddle en WeChat | Únase al grupo de discusión técnica |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |


## 😃 Proyectos increíbles que aprovechan PaddleOCR
¡PaddleOCR no estaría donde está hoy sin su increíble comunidad! 💗 Un enorme agradecimiento a todos nuestros socios de siempre, nuevos colaboradores y a todos los que han volcado su pasión en PaddleOCR, ya sea que los hayamos nombrado o no. ¡Su apoyo alimenta nuestro fuego!

| Nombre del Proyecto | Descripción |
| ------------ | ----------- |
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|Motor de RAG basado en la comprensión profunda de documentos.|
| [pathway](https://github.com/pathwaycom/pathway) <a href="https://github.com/pathwaycom/pathway"><img src="https://img.shields.io/github/stars/pathwaycom/pathway"></a>|Un marco ETL en Python para el procesamiento de flujos, análisis en tiempo real, canalizaciones LLM y RAG.|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|Herramienta de conversión de documentos de múltiples tipos a Markdown.|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|Software de OCR por lotes, sin conexión, gratuito y de código abierto.|
| [cherry-studio](https://github.com/CherryHQ/cherry-studio) <a href="https://github.com/CherryHQ/cherry-studio"><img src="https://img.shields.io/github/stars/CherryHQ/cherry-studio"></a>|Un cliente de escritorio que admite múltiples proveedores de LLM.|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |OmniParser: Herramienta de análisis de pantalla para agentes GUI basados puramente en visión.|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |Preguntas y respuestas basadas en cualquier cosa.|
| [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) <a href="https://github.com/opendatalab/PDF-Extract-Kit"><img src="https://img.shields.io/github/stars/opendatalab/PDF-Extract-Kit"></a>|Un potente toolkit de código abierto diseñado para extraer eficientemente contenido de alta calidad de documentos PDF complejos y diversos.|
| [Dango-Translator](https://github.com/PantsuDango/Dango-Translator)<a href="https://github.com/PantsuDango/Dango-Translator"><img src="https://img.shields.io/github/stars/PantsuDango/Dango-Translator"></a> |Reconoce texto en la pantalla, lo traduce y muestra los resultados de la traducción en tiempo real.|
| [Conozca más proyectos](../awesome_projects.md) | [Más proyectos basados en PaddleOCR](../awesome_projects.md)|

## 👩‍👩‍👧‍👦 Contribuidores

<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>

## 🌟 Star

[![Star History Chart](https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date)](https://star-history.com/#PaddlePaddle/PaddleOCR&Date)


## 📄 Licencia
Este proyecto se publica bajo la [licencia Apache 2.0](LICENSE).

## 🎓 Citación

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
