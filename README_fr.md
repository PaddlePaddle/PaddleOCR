<div align="center">
  <p>
      <img width="100%" src="./docs/images/Banner.png" alt="Bannière PaddleOCR">
  </p>

<!-- language -->
[English](./README.md) | [简体中文](./README_cn.md) | [繁體中文](./README_tcn.md) | [日本語](./README_ja.md) | [한국어](./README_ko.md) | Français | [Русский](./README_ru.md) | [Español](./README_es.md) | [العربية](./README_ar.md)

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

## 🚀 Introduction
Depuis sa sortie initiale, PaddleOCR a été largement acclamé par les milieux universitaires, industriels et de la recherche, grâce à ses algorithmes de pointe et à ses performances éprouvées dans des applications réelles. Il alimente déjà des projets open-source populaires tels que Umi-OCR, OmniParser, MinerU et RAGFlow, ce qui en fait la boîte à outils OCR de référence pour les développeurs du monde entier.

Le 20 mai 2025, l'équipe de PaddlePaddle a dévoilé PaddleOCR 3.0, entièrement compatible avec la version officielle du framework **PaddlePaddle 3.0**. Cette mise à jour **améliore encore la précision de la reconnaissance de texte**, ajoute la prise en charge de la **reconnaissance de multiples types de texte** et de la **reconnaissance de l'écriture manuscrite**, et répond à la demande croissante des applications de grands modèles pour l'**analyse de haute précision de documents complexes**. Combiné avec **ERNIE 4.5 Turbo**, il améliore considérablement la précision de l'extraction d'informations clés. Pour la documentation d'utilisation complète, veuillez vous référer à la [Documentation de PaddleOCR 3.0](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html).

Trois nouvelles fonctionnalités majeures dans PaddleOCR 3.0 :
- Modèle de reconnaissance de texte pour toutes scènes [PP-OCRv5](./docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.en.md) : Un modèle unique qui gère cinq types de texte différents ainsi que l'écriture manuscrite complexe. La précision globale de la reconnaissance a augmenté de 13 points de pourcentage par rapport à la génération précédente. [Démo en ligne](https://aistudio.baidu.com/community/app/91660/webUI)

- Solution d'analyse de documents générique [PP-StructureV3](./docs/version3.x/algorithm/PP-StructureV3/PP-StructureV3.en.md) : Fournit une analyse de haute précision des PDF multi-mises en page et multi-scènes, surpassant de nombreuses solutions open-source et propriétaires sur les benchmarks publics. [Démo en ligne](https://aistudio.baidu.com/community/app/518494/webUI)

- Solution de compréhension de documents intelligente [PP-ChatOCRv4](./docs/version3.x/algorithm/PP-ChatOCRv4/PP-ChatOCRv4.en.md) : Nativement propulsé par le grand modèle ERNIE 4.5 Turbo, atteignant une précision supérieure de 15 points de pourcentage à celle de son prédécesseur. [Démo en ligne](https://aistudio.baidu.com/community/app/518493/webUI)

En plus de fournir une bibliothèque de modèles exceptionnelle, PaddleOCR 3.0 propose également des outils conviviaux couvrant l'entraînement de modèles, l'inférence et le déploiement de services, afin que les développeurs puissent rapidement mettre en production des applications d'IA.
<div align="center">
  <p>
      <img width="100%" src="./docs/images/Arch.png" alt="Architecture de PaddleOCR">
  </p>
</div>

## 📣 Mises à jour récentes

#### **🔥🔥 05/06/2025 : Publication de PaddleOCR 3.0.1, incluant :**

- **Optimisation de certains modèles et de leurs configurations :**
  - Mise à jour de la configuration par défaut du modèle pour PP-OCRv5, en passant les modèles de détection et de reconnaissance de `mobile` à `server`. Pour améliorer les performances par défaut dans la plupart des scénarios, le paramètre `limit_side_len` dans la configuration a été changé de 736 à 64.
  - Ajout d'un nouveau modèle de classification de l'orientation des lignes de texte `PP-LCNet_x1_0_textline_ori` avec une précision de 99.42%. Le classifieur d'orientation de ligne de texte par défaut pour les pipelines OCR, PP-StructureV3 et PP-ChatOCRv4 a été mis à jour vers ce modèle.
  - Optimisation du modèle de classification de l'orientation des lignes de texte `PP-LCNet_x0_25_textline_ori`, améliorant la précision de 3,3 points de pourcentage pour atteindre une précision actuelle de 98,85%.

- **Optimisations et corrections de certains problèmes de la version 3.0.0, [détails](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)**

🔥🔥20/05/2025 : Lancement officiel de **PaddleOCR v3.0**, incluant :
- **PP-OCRv5** : Modèle de reconnaissance de texte de haute précision pour tous les scénarios - Texte instantané à partir d'images/PDF.
   1. 🌐 Prise en charge par un seul modèle de **cinq** types de texte - Traitez de manière transparente le **chinois simplifié, le chinois traditionnel, le pinyin chinois simplifié, l'anglais** et le **japonais** au sein d'un seul modèle.
   2. ✍️ **Reconnaissance de l'écriture manuscrite** améliorée : Nettement plus performant sur les écritures cursives complexes et non standard.
   3. 🎯 **Gain de précision de 13 points** par rapport à PP-OCRv4, atteignant des performances de pointe dans une variété de scénarios réels.

- **PP-StructureV3** : Analyse de documents à usage général – Libérez une analyse d'images/PDF de pointe pour des scénarios du monde réel !
   1. 🧮 **Analyse de PDF multi-scènes de haute précision**, devançant les solutions open-source et propriétaires sur le benchmark OmniDocBench.
   2. 🧠 Les capacités spécialisées incluent la **reconnaissance de sceaux**, la **conversion de graphiques en tableaux**, la **reconnaissance de tableaux avec formules/images imbriquées**, l'**analyse de documents à texte vertical** et l'**analyse de structures de tableaux complexes**.

- **PP-ChatOCRv4** : Compréhension intelligente de documents – Extrayez des informations clés, pas seulement du texte, à partir d'images/PDF.
   1. 🔥 **Gain de précision de 15 points** dans l'extraction d'informations clés sur les fichiers PDF/PNG/JPG par rapport à la génération précédente.
   2. 💻 Prise en charge native de **ERNIE 4.5 Turbo**, avec une compatibilité pour les déploiements de grands modèles via PaddleNLP, Ollama, vLLM, et plus encore.
   3. 🤝 Intégration de [PP-DocBee2](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee2), permettant l'extraction et la compréhension de texte imprimé, d'écriture manuscrite, de sceaux, de tableaux, de graphiques et d'autres éléments courants dans les documents complexes.

<details>
   <summary><strong>Historique des mises à jour</strong></summary>

- 🔥🔥07/03/2025 : Lancement de **PaddleOCR v2.10**, incluant :

  - **12 nouveaux modèles développés en interne :**
    - **Série [Détection de mise en page](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html)** (3 modèles) : PP-DocLayout-L, M et S -- capables de détecter 23 types de mise en page courants dans divers formats de documents (articles, rapports, examens, livres, magazines, contrats, etc.) en anglais et en chinois. Atteint jusqu'à **90.4% mAP@0.5**, et les fonctionnalités légères peuvent traiter plus de 100 pages par seconde.
    - **Série [Reconnaissance de formules](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/formula_recognition.html)** (2 modèles) : PP-FormulaNet-L et S -- prend en charge la reconnaissance de plus de 50 000 expressions LaTeX, gérant à la fois les formules imprimées et manuscrites. PP-FormulaNet-L offre une **précision supérieure de 6 %** par rapport aux modèles comparables ; PP-FormulaNet-S est 16 fois plus rapide tout en conservant une précision similaire.
    - **Série [Reconnaissance de structure de tableau](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_structure_recognition.html)** (2 modèles) : SLANeXt_wired et SLANeXt_wireless -- modèles nouvellement développés avec une **amélioration de la précision de 6 %** par rapport à SLANet_plus dans la reconnaissance de tableaux complexes.
    - **[Classification de tableau](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_classification.html)** (1 modèle) : PP-LCNet_x1_0_table_cls -- un classifieur ultra-léger pour les tableaux avec et sans fils.

[En savoir plus](https://paddlepaddle.github.io/PaddleOCR/latest/en/update.html)

</details>

## ⚡ Démarrage Rapide
### 1. Lancer la démo en ligne
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

### 2. Installation

Installez PaddlePaddle en vous référant au [Guide d'installation](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html), puis installez la boîte à outils PaddleOCR.

```bash
# Installer paddleocr
pip install paddleocr
```

### 3. Exécuter l'inférence par CLI
```bash
# Exécuter l'inférence PP-OCRv5
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False

# Exécuter l'inférence PP-StructureV3
paddleocr pp_structurev3 -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png --use_doc_orientation_classify False --use_doc_unwarping False

# Obtenez d'abord la clé API Qianfan, puis exécutez l'inférence PP-ChatOCRv4
paddleocr pp_chatocrv4_doc -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png -k 驾驶室准乘人数 --qianfan_api_key your_api_key --use_doc_orientation_classify False --use_doc_unwarping False

# Obtenir plus d'informations sur "paddleocr ocr"
paddleocr ocr --help
```

### 4. Exécuter l'inférence par API
**4.1 Exemple PP-OCRv5**
```python
# Initialiser l'instance de PaddleOCR
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# Exécuter l'inférence OCR sur un exemple d'image
result = ocr.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png")

# Visualiser les résultats et sauvegarder les résultats JSON
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

<details>
    <summary><strong>4.2 Exemple PP-StructureV3</strong></summary>

```python
from pathlib import Path
from paddleocr import PPStructureV3

pipeline = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
)

# Pour une image
output = pipeline.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png",
)

# Visualiser les résultats et sauvegarder les résultats JSON
for res in output:
    res.print() 
    res.save_to_json(save_path="output") 
    res.save_to_markdown(save_path="output")           
```

</details>

<details>
   <summary><strong>4.3 Exemple PP-ChatOCRv4</strong></summary>

```python
from paddleocr import PPChatOCRv4Doc

chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "ernie-3.5-8k",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key",  # votre api_key
}

retriever_config = {
    "module_name": "retriever",
    "model_name": "embedding-v1",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "qianfan",
    "api_key": "api_key",  # votre api_key
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
# Si un grand modèle multimodal est utilisé, le service mllm local doit être démarré. Vous pouvez vous référer à la documentation : https://github.com/PaddlePaddle/PaddleX/blob/release/3.0/docs/pipeline_usage/tutorials/vlm_pipelines/doc_understanding.en.md pour effectuer le déploiement et mettre à jour la configuration mllm_chat_bot_config.
if use_mllm:
    mllm_chat_bot_config = {
        "module_name": "chat_bot",
        "model_name": "PP-DocBee",
        "base_url": "http://127.0.0.1:8080/",  # url de votre service mllm local
        "api_type": "openai",
        "api_key": "api_key",  # votre api_key
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

## ⛰️ Tutoriels avancés
- [Tutoriel PP-OCRv5](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/OCR.html)
- [Tutoriel PP-StructureV3](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-StructureV3.html)
- [Tutoriel PP-ChatOCRv4](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-ChatOCRv4.html)

## 🔄 Aperçu rapide des résultats d'exécution

<div align="center">
  <p>
     <img width="100%" src="./docs/images/demo.gif" alt="Démo PP-OCRv5">
  </p>
</div>

<div align="center">
  <p>
     <img width="100%" src="./docs/images/blue_v3.gif" alt="Démo PP-StructureV3">
  </p>
</div>

## 👩‍👩‍👧‍👦 Communauté

| Compte officiel WeChat de PaddlePaddle | Rejoignez le groupe de discussion technique |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |

## 😃 Projets formidables utilisant PaddleOCR
PaddleOCR ne serait pas là où il est aujourd'hui sans son incroyable communauté ! 💗 Un immense merci à tous nos partenaires de longue date, nos nouveaux collaborateurs, et tous ceux qui ont mis leur passion dans PaddleOCR — que nous vous ayons nommés ou non. Votre soutien nous anime !

| Nom du projet | Description |
| ------------ | ----------- |
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|Moteur RAG basé sur la compréhension profonde des documents.|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|Outil de conversion de documents multi-types en Markdown|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|Logiciel d'OCR hors ligne, gratuit, open-source et par lots.|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a>|Outil d'analyse d'écran pour agent GUI basé sur la vision pure.|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a>|Questions et réponses basées sur n'importe quel contenu.|
| [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) <a href="https://github.com/opendatalab/PDF-Extract-Kit"><img src="https://img.shields.io/github/stars/opendatalab/PDF-Extract-Kit"></a>|Une puissante boîte à outils open-source conçue pour extraire efficacement du contenu de haute qualité à partir de documents PDF complexes et diversifiés.|
| [Dango-Translator](https://github.com/PantsuDango/Dango-Translator)<a href="https://github.com/PantsuDango/Dango-Translator"><img src="https://img.shields.io/github/stars/PantsuDango/Dango-Translator"></a>|Reconnaît le texte à l'écran, le traduit et affiche les résultats de la traduction en temps réel.|
| [En savoir plus](./awesome_projects.md) | [Plus de projets basés sur PaddleOCR](./awesome_projects.md)|

## 👩‍👩‍👧‍👦 Contributeurs

<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>

## 🌟 Star

[![Star History Chart](https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date)](https://star-history.com/#PaddlePaddle/PaddleOCR&Date)

## 📄 Licence
Ce projet est publié sous la [licence Apache 2.0](LICENSE).

## 🎓 Citation

```
@misc{paddleocr2020,
title={PaddleOCR, Awesome multilingual OCR toolkits based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleOCR}},
year={2020}
}
``` 
