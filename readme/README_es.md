<div align="center">
  <p>
      <img width="800" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Banner.png" alt="Star-history">
  </p>



<h3>Kit de OCR líder mundial y motor de IA para documentos</h3>

[English](../README.md) | [简体中文](./README_cn.md) | [繁體中文](./README_tcn.md) | [日本語](./README_ja.md) | [한국어](./README_ko.md) | [Français](./README_fr.md) | [Русский](./README_ru.md) | Español | [العربية](./README_ar.md)

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






**PaddleOCR convierte documentos e imágenes en datos estructurados y listos para LLM (JSON/Markdown) con una precisión líder en la industria. Con más de 70 000 estrellas y la confianza de proyectos de primer nivel como Dify, RAGFlow y Cherry Studio, PaddleOCR es la base para construir aplicaciones inteligentes de RAG y Agentic.**


## 🚀 Características principales

### 📄 Análisis inteligente de documentos (listo para LLM)
> *Transformando contenido visual complejo en datos estructurados para la era de los LLM.*

* **SOTA Document VLM**: Con **PaddleOCR-VL-1.5 (0,9B)**, el modelo de visión y lenguaje ligero líder de la industria para el análisis de documentos. Sobresale en el análisis de documentos complejos en 5 grandes desafíos del "mundo real": **deformación, escaneo, fotografía de pantalla, iluminación y documentos inclinados**, con salidas estructuradas en formatos **Markdown** y **JSON**.
* **Conversión con reconocimiento de estructura**: Impulsado por **PP-StructureV3**, convierte sin problemas PDFs e imágenes complejas en **Markdown** o **JSON**. A diferencia de los modelos de la serie PaddleOCR-VL, proporciona información de coordenadas más detallada, incluyendo coordenadas de celdas de tablas, coordenadas de texto y más.
* **Eficiencia lista para producción**: Logra precisión de nivel comercial con una huella ultrapequeña. Supera a numerosas soluciones de código cerrado en benchmarks públicos, manteniéndose eficiente en recursos para despliegue en el borde o en la nube.

### 🔍 Reconocimiento universal de texto (Scene OCR)
> *El estándar de oro mundial para la detección de texto multilingüe de alta velocidad.*

* **Compatibilidad con más de 100 idiomas**: Reconocimiento nativo de una amplia biblioteca global. Nuestra solución de modelo único **PP-OCRv5** maneja con elegancia documentos mixtos multilingües (chino, inglés, japonés, pinyin, etc.).
* **Dominio de elementos complejos**: Más allá del reconocimiento de texto estándar, admitimos la **detección de texto en escenas naturales** en una amplia gama de entornos, incluyendo documentos de identidad, vistas de calles, libros y componentes industriales.
* **Salto en rendimiento**: PP-OCRv5 ofrece una **mejora de precisión del 13%** respecto a versiones anteriores, manteniendo la "eficiencia extrema" por la que PaddleOCR es famoso.

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Arch.jpg" alt="PaddleOCR Architecture">
  </p>
</div>

### 🛠️ Ecosistema centrado en el desarrollador
* **Integración perfecta**: La opción preferida para el ecosistema de agentes de IA, con integración profunda en **Dify, RAGFlow, Pathway y Cherry Studio**.
* **Motor de datos para LLM**: Un pipeline completo para construir conjuntos de datos de alta calidad, proporcionando un "Motor de Datos" sostenible para el ajuste fino de modelos de lenguaje grandes.
* **Despliegue en un clic**: Compatible con diversos backends de hardware (GPU NVIDIA, CPU Intel, XPU Kunlunxin y diversos aceleradores de IA).


## 📣 Actualizaciones recientes

### 🔥 PaddleOCR v3.5.0: backends de inferencia más flexibles y salida documental más rica
* **Backends de inferencia más flexibles**: cambia sin problemas entre grafo estático de Paddle, grafo dinámico de Paddle y Transformers. PaddleOCR está ahora profundamente integrado con el ecosistema de Hugging Face, y 20 modelos principales admiten Transformers como backend de inferencia.
* **Documentos de Office a Markdown**: convierte formatos de documentos comunes como Word, Excel y PowerPoint a Markdown.
* **Exportación de resultados a DOCX**: las series `PaddleOCR-VL`, `PP-StructureV3` y `PP-DocTranslation` ahora admiten exportar los resultados de análisis a DOCX para verlos y editarlos cómodamente en Microsoft Word.
* **SDK oficial de inferencia en navegador**: se lanzó `PaddleOCR.js`, el SDK oficial de inferencia en navegador, que permite ejecutar `PP-OCRv5` directamente en el navegador.

<details>
<summary><strong>2026.01.29: Lanzamiento de PaddleOCR 3.4.0</strong></summary>
* **PaddleOCR-VL-1.5 (SOTA 0,9B VLM)**: ¡Nuestro último modelo insignia para el análisis de documentos ya está disponible!
    * **94,5% de precisión en OmniDocBench**: Superando a los mejores modelos generales de gran escala y a los analizadores de documentos especializados.
    * **Robustez en el mundo real**: El primero en introducir el algoritmo **PP-DocLayoutV3** para el posicionamiento de formas irregulares, dominando 5 escenarios difíciles: *inclinación, deformación, escaneo, iluminación y fotografía de pantalla*.
    * **Expansión de capacidades**: Ahora admite **reconocimiento de sellos**, **detección de texto** y se expande a **111 idiomas** (incluyendo el tibetano de China y el bengalí).
    * **Dominio de documentos largos**: Admite la fusión automática de tablas entre páginas e identificación jerárquica de encabezados.
    * **Pruébalo ahora**: Disponible en [HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) o en nuestro [Sitio web oficial](https://www.paddleocr.com).

</details>

<details>
<summary><strong>2025.10.16: Lanzamiento de PaddleOCR 3.3.0</strong></summary>

- Lanzamiento de PaddleOCR-VL:
    - **Introducción al modelo**:
        - **PaddleOCR-VL** es un modelo SOTA eficiente en recursos diseñado específicamente para el análisis de documentos. Su componente principal es PaddleOCR-VL-0.9B, un modelo de visión y lenguaje (VLM) compacto pero potente que integra un codificador visual de resolución dinámica al estilo NaViT con el modelo de lenguaje ERNIE-4.5-0.3B para permitir un reconocimiento preciso de elementos. **Este innovador modelo admite eficientemente 109 idiomas y sobresale en el reconocimiento de elementos complejos (p. ej., texto, tablas, fórmulas y gráficos), mientras mantiene un consumo mínimo de recursos**. A través de evaluaciones exhaustivas en benchmarks públicos ampliamente utilizados y benchmarks internos, PaddleOCR-VL logra un rendimiento SOTA tanto en el análisis de documentos a nivel de página como en el reconocimiento a nivel de elemento. Supera significativamente a las soluciones existentes, exhibe una fuerte competitividad frente a los VLM de primer nivel y ofrece velocidades de inferencia rápidas. Estas fortalezas lo hacen altamente adecuado para el despliegue práctico en escenarios del mundo real. El modelo ha sido publicado en [HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL). ¡Todos son bienvenidos a descargarlo y usarlo! Más información de introducción se puede encontrar en [PaddleOCR-VL](https://www.paddleocr.ai/latest/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.html).

    - **Características principales**:
        - **Arquitectura VLM compacta pero potente**: Presentamos un novedoso modelo de visión y lenguaje diseñado específicamente para una inferencia eficiente en recursos, logrando un rendimiento sobresaliente en el reconocimiento de elementos. Al integrar un codificador visual dinámico de alta resolución al estilo NaViT con el modelo de lenguaje ligero ERNIE-4.5-0.3B, mejoramos significativamente las capacidades de reconocimiento y la eficiencia de decodificación del modelo. Esta integración mantiene una alta precisión mientras reduce las demandas computacionales, lo que lo hace adecuado para aplicaciones de procesamiento de documentos eficientes y prácticas.
        - **Rendimiento SOTA en análisis de documentos**: PaddleOCR-VL logra un rendimiento de vanguardia tanto en el análisis de documentos a nivel de página como en el reconocimiento a nivel de elemento. Supera significativamente a las soluciones basadas en pipeline existentes y exhibe una fuerte competitividad frente a los modelos de visión y lenguaje (VLM) líderes en el análisis de documentos. Además, sobresale en el reconocimiento de elementos de documentos complejos, como texto, tablas, fórmulas y gráficos, lo que lo hace adecuado para una amplia gama de tipos de contenido desafiantes, incluyendo texto manuscrito y documentos históricos. Esto lo hace altamente versátil y adecuado para una amplia gama de tipos de documentos y escenarios.
        - **Soporte multilingüe**: PaddleOCR-VL admite 109 idiomas, cubriendo los principales idiomas globales, incluyendo, entre otros, chino, inglés, japonés, latín y coreano, así como idiomas con diferentes escrituras y estructuras, como el ruso (escritura cirílica), el árabe, el hindi (escritura devanagari) y el tailandés. Esta amplia cobertura de idiomas mejora sustancialmente la aplicabilidad de nuestro sistema a escenarios de procesamiento de documentos multilingües y globalizados.

- Lanzamiento del modelo de reconocimiento multilingüe PP-OCRv5:
    - Se mejoró la precisión y cobertura del reconocimiento de escritura latina; se añadió compatibilidad con sistemas de escritura cirílico, árabe, devanagari, telugu, tamil y otros, cubriendo el reconocimiento de 109 idiomas. El modelo tiene solo 2M de parámetros, y la precisión de algunos modelos ha aumentado más de un 40% en comparación con la generación anterior.

</details>


<details>
<summary><strong>2025.08.21: Lanzamiento de PaddleOCR 3.2.0</strong></summary>

- **Adiciones significativas de modelos:**
    - Se introdujeron entrenamiento, inferencia y despliegue para modelos de reconocimiento PP-OCRv5 en inglés, tailandés y griego. **El modelo PP-OCRv5 en inglés ofrece una mejora del 11% en escenarios en inglés en comparación con el modelo principal PP-OCRv5, con los modelos de reconocimiento en tailandés y griego alcanzando precisiones del 82,68% y 89,28%, respectivamente.**

- **Mejoras en las capacidades de despliegue:**
    - **Compatibilidad total con las versiones 3.1.0 y 3.1.1 del framework PaddlePaddle.**
    - **Actualización integral de la solución de despliegue local en C++ de PP-OCRv5, que ahora admite tanto Linux como Windows, con paridad de características y precisión idéntica a la implementación en Python.**
    - **La inferencia de alto rendimiento ahora admite CUDA 12, y la inferencia puede realizarse utilizando los backends Paddle Inference u ONNX Runtime.**
    - **La solución de despliegue orientado a servicios de alta estabilidad ahora es completamente de código abierto, lo que permite a los usuarios personalizar imágenes Docker y SDKs según sea necesario.**
    - La solución de despliegue orientado a servicios de alta estabilidad también admite la invocación mediante solicitudes HTTP construidas manualmente, lo que permite el desarrollo de código del lado del cliente en cualquier lenguaje de programación.

- **Soporte de benchmarks:**
    - **Todas las líneas de producción ahora admiten benchmarking detallado, permitiendo medir el tiempo de inferencia de extremo a extremo, así como datos de latencia por capa y por módulo para ayudar en el análisis de rendimiento. [Aquí](docs/version3.x/pipeline_usage/instructions/benchmark.en.md) se explica cómo configurar y usar la función de benchmark.**
    - **La documentación se ha actualizado para incluir métricas clave para configuraciones de uso común en hardware convencional, como la latencia de inferencia y el uso de memoria, proporcionando referencias de despliegue para los usuarios.**

- **Corrección de errores:**
    - Se resolvió el problema del guardado fallido de registros durante el entrenamiento del modelo.
    - Se actualizó el componente de aumento de datos para modelos de fórmulas para compatibilidad con versiones más recientes de la dependencia albumentations, y se corrigieron advertencias de bloqueo al usar el paquete tokenizers en escenarios multiproceso.
    - Se corrigieron inconsistencias en los comportamientos de los interruptores (p. ej., `use_chart_parsing`) en los archivos de configuración de PP-StructureV3 en comparación con otros pipelines.

- **Otras mejoras:**
    - **Se separaron las dependencias principales y opcionales. Solo se requieren dependencias principales mínimas para el reconocimiento de texto básico; las dependencias adicionales para el análisis de documentos y la extracción de información se pueden instalar según sea necesario.**
    - **Se habilitó la compatibilidad con tarjetas gráficas NVIDIA RTX de la serie 50 en Windows; los usuarios pueden consultar la [guía de instalación](docs/version3.x/installation.en.md) para conocer las versiones correspondientes del framework PaddlePaddle.**
    - **Los modelos de la serie PP-OCR ahora admiten la devolución de coordenadas de caracteres individuales.**
    - Se añadieron fuentes de descarga de modelos de AIStudio, ModelScope y otras, permitiendo a los usuarios especificar la fuente para las descargas de modelos.
    - Se añadió compatibilidad con la conversión de gráficos a tablas a través del módulo PP-Chart2Table.
    - Se optimizaron las descripciones de la documentación para mejorar la usabilidad.
</details>


[Historial de cambios](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)

</details>


## 🚀 Inicio rápido

### Paso 1: Pruébalo en línea
El sitio web oficial de PaddleOCR ofrece un **Centro de experiencia** interactivo y **APIs** — sin necesidad de configuración, solo un clic para experimentar.

👉 [Visitar el sitio web oficial](https://www.paddleocr.com)

### Paso 2: Despliegue local
Para uso local, consulte la siguiente documentación según sus necesidades:

- **Serie PP-OCR**: Consulte la [Documentación de PP-OCR](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html)
- **Serie PaddleOCR-VL**: Consulte la [Documentación de PaddleOCR-VL](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- **PP-StructureV3**: Consulte la [Documentación de PP-StructureV3](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PP-StructureV3.html)
- **Más capacidades**: Consulte la [Documentación de más capacidades](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/pipeline_overview.html)


## 🧩 Más características

- Convertir modelos al formato ONNX: [Obtención de modelos ONNX](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/obtaining_onnx_models.html).
- Acelerar la inferencia usando motores como OpenVINO, ONNX Runtime, TensorRT, o realizar inferencia usando modelos en formato ONNX: [Inferencia de alto rendimiento](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/high_performance_inference.html).
- Acelerar la inferencia usando múltiples GPU y múltiples procesos: [Inferencia paralela para pipelines](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/instructions/parallel_inference.html).
- Integrar PaddleOCR en aplicaciones escritas en C++, C#, Java, etc.: [Serving](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/serving.html).

## 🔄 Resumen rápido de los resultados de ejecución

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


## ✨ Mantente al día

⭐ **¡Dale una estrella a este repositorio para estar al tanto de emocionantes actualizaciones y nuevos lanzamientos, incluyendo potentes capacidades de OCR y análisis de documentos!** ⭐

<div align="center">
  <p>
       <img width="1200" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/star_paddleocr2.en.gif" alt="Star-Project">
  </p>
</div>


## 👩‍👩‍👧‍👦 Comunidad

<div align="center">

| Cuenta oficial de PaddlePaddle en WeChat | Únete al grupo de discusión técnica |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |
</div>


## 😃 Proyectos destacados que utilizan PaddleOCR
¡PaddleOCR no estaría donde está hoy sin su increíble comunidad! 💗 Un enorme agradecimiento a todos nuestros socios de larga data, nuevos colaboradores y a todos los que han volcado su pasión en PaddleOCR — los hayamos mencionado o no. ¡Su apoyo alimenta nuestro fuego!

<div align="center">

| Nombre del proyecto | Descripción |
| ------------ | ----------- |
| [Dify](https://github.com/langgenius/dify) <a href="https://github.com/langgenius/dify"><img src="https://img.shields.io/github/stars/langgenius/dify"></a>|Plataforma lista para producción para el desarrollo de flujos de trabajo agénticos.|
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|Motor RAG basado en la comprensión profunda de documentos.|
| [pathway](https://github.com/pathwaycom/pathway) <a href="https://github.com/pathwaycom/pathway"><img src="https://img.shields.io/github/stars/pathwaycom/pathway"></a>|Framework ETL de Python para procesamiento de flujos, análisis en tiempo real, pipelines de LLM y RAG.|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|Herramienta de conversión de documentos de múltiples tipos a Markdown.|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|Software de OCR offline por lotes, gratuito y de código abierto.|
| [cherry-studio](https://github.com/CherryHQ/cherry-studio) <a href="https://github.com/CherryHQ/cherry-studio"><img src="https://img.shields.io/github/stars/CherryHQ/cherry-studio"></a>|Cliente de escritorio compatible con múltiples proveedores de LLM.|
| [haystack](https://github.com/deepset-ai/haystack)<a href="https://github.com/deepset-ai/haystack"><img src="https://img.shields.io/github/stars/deepset-ai/haystack"></a> |Framework de orquestación de IA para construir aplicaciones LLM personalizables y listas para producción.|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |OmniParser: herramienta de análisis de pantalla para agentes GUI basados únicamente en visión.|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |Preguntas y respuestas basadas en cualquier cosa.|
| [Ver más proyectos](./awesome_projects.md) | [Más proyectos basados en PaddleOCR](./awesome_projects.md)|
</div>

## 👩‍👩‍👧‍👦 Colaboradores

<div align="center">
<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>
</div>

## 🌟 Estrellas

<div align="center">
  <p>
      <img width="800" src="https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date" alt="Star-history">
  </p>
</div>


## 📄 Licencia
Este proyecto se publica bajo la [licencia Apache 2.0](LICENSE).

## 🎓 Cita

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
