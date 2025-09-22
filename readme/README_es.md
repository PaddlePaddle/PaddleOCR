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


[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

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
| `doc-parser` | Análisis de documentos: se puede usar para extraer elementos de diseño como tablas, fórmulas, sellos, imágenes, etc. de los documentos; incluye modelos como PP-StructureV3 |
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

## 🧩 Más funciones

- Convertir modelos al formato ONNX: [Obtener modelos ONNX](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/obtaining_onnx_models.html).
- Acelerar la inferencia usando motores como OpenVINO, ONNX Runtime, TensorRT, o realizar inferencia usando modelos en formato ONNX: [Inferencia de alto rendimiento](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/high_performance_inference.html).
- Acelerar la inferencia usando múltiples GPU y múltiples procesos: [Inferencia paralela para pipelines](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/instructions/parallel_inference.html).
- Integra PaddleOCR en aplicaciones escritas en C++, C#, Java, etc.: [Servicio](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/serving.html).

## ⛰️ Tutoriales avanzados
- [Tutorial de PP-OCRv5](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/OCR.html)
- [Tutorial de PP-StructureV3](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-StructureV3.html)
- [Tutorial de PP-ChatOCRv4](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-ChatOCRv4.html)

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
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|Herramienta de conversión de documentos de múltiples tipos a Markdown.|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|Software de OCR por lotes, sin conexión, gratuito y de código abierto.|
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

```
@misc{paddleocr2020,
title={PaddleOCR, Awesome multilingual OCR toolkits based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleOCR}},
year={2020}
}
