<div align="center">
  <p>
      <img width="100%" src="../docs/images/Banner.png" alt="Bannière PaddleOCR">
  </p>

<!-- language -->
[English](../README.md) | [简体中文](./README_cn.md) | [繁體中文](./README_tcn.md) | [日本語](./README_ja.md) | [한국어](./README_ko.md) | Français | [Русский](./README_ru.md) | [Español](./README_es.md) | [العربية](./README_ar.md)

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

Le 20 mai 2025, l'équipe de PaddlePaddle a dévoilé PaddleOCR 3.0, entièrement compatible avec la version officielle du framework **PaddlePaddle 3.0**. Cette mise à jour **améliore encore la précision de la reconnaissance de texte**, ajoute la prise en charge de la **reconnaissance de multiples types de texte** et de la **reconnaissance de l'écriture manuscrite**, et répond à la demande croissante des applications de grands modèles pour l'**analyse de haute précision de documents complexes**. Combiné avec **ERNIE 4.5**, il améliore considérablement la précision de l'extraction d'informations clés. Pour la documentation d'utilisation complète, veuillez vous référer à la [Documentation de PaddleOCR 3.0](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html).

Trois nouvelles fonctionnalités majeures dans PaddleOCR 3.0 :
- Modèle de reconnaissance de texte pour toutes scènes [PP-OCRv5](../docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.en.md) : Un modèle unique qui gère cinq types de texte différents ainsi que l'écriture manuscrite complexe. La précision globale de la reconnaissance a augmenté de 13 points de pourcentage par rapport à la génération précédente. [Démo en ligne](https://aistudio.baidu.com/community/app/91660/webUI)

- Solution d'analyse de documents générique [PP-StructureV3](../docs/version3.x/algorithm/PP-StructureV3/PP-StructureV3.en.md) : Fournit une analyse de haute précision des PDF multi-mises en page et multi-scènes, surpassant de nombreuses solutions open-source et propriétaires sur les benchmarks publics. [Démo en ligne](https://aistudio.baidu.com/community/app/518494/webUI)

- Solution de compréhension de documents intelligente [PP-ChatOCRv4](../docs/version3.x/algorithm/PP-ChatOCRv4/PP-ChatOCRv4.en.md) : Nativement propulsé par le grand modèle ERNIE 4.5, atteignant une précision supérieure de 15 points de pourcentage à celle de son prédécesseur. [Démo en ligne](https://aistudio.baidu.com/community/app/518493/webUI)

En plus de fournir une bibliothèque de modèles exceptionnelle, PaddleOCR 3.0 propose également des outils conviviaux couvrant l'entraînement de modèles, l'inférence et le déploiement de services, afin que les développeurs puissent rapidement mettre en production des applications d'IA.
<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Arch.jpg" alt="PaddleOCR Architecture">
  </p>
</div>

**Remarque spéciale** : PaddleOCR 3.x introduit plusieurs changements importants d’interface. **L'ancien code écrit sur la base de PaddleOCR 2.x est probablement incompatible avec PaddleOCR 3.x**. Veuillez vous assurer que la documentation que vous consultez correspond à la version de PaddleOCR que vous utilisez. [Ce document](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/upgrade_notes.html) explique les raisons de la mise à niveau et les principaux changements entre PaddleOCR 2.x et 3.x.

## 📣 Mises à jour récentes
#### **🔥🔥21/08/2025 : Sortie de PaddleOCR 3.2.0**, comprend :

- **Ajouts majeurs de modèles :**
    - Ajout de l’entraînement, de l’inférence et du déploiement des modèles de reconnaissance PP-OCRv5 en anglais, thaï et grec. **Le modèle anglais PP-OCRv5 offre une amélioration de 11 % dans les scénarios anglophones par rapport au modèle principal PP-OCRv5, tandis que les modèles de reconnaissance thaï et grec atteignent respectivement des précisions de 82,68 % et 89,28 %.**

- **Améliorations des capacités de déploiement :**
    - **Support complet des versions 3.1.0 et 3.1.1 du framework PaddlePaddle.**
    - **Mise à niveau complète de la solution de déploiement local PP-OCRv5 en C++ : compatible Linux et Windows, avec des fonctionnalités et une précision identiques à la version Python.**
    - **Prise en charge des inférences haute performance via CUDA 12, avec possibilité d’utiliser Paddle Inference ou le backend ONNX Runtime.**
    - **La solution de déploiement orientée service, hautement stable, est désormais entièrement open source, permettant aux utilisateurs de personnaliser les images Docker et les SDK selon leurs besoins.**
    - Cette solution prend également en charge l’appel via des requêtes HTTP construites manuellement, permettant le développement du client dans n’importe quel langage de programmation.

- **Support du benchmark :**
    - **Toutes les chaînes de production prennent désormais en charge des benchmarks fins, permettant de mesurer le temps d’inférence de bout en bout ainsi que les temps d’exécution par couche et par module, ce qui facilite l’analyse des performances.[Voici](../docs/version3.x/pipeline_usage/instructions/benchmark.en.md) comment configurer et utiliser la fonctionnalité de benchmark**
    - **La documentation fournit désormais des indicateurs clés (temps d’inférence, occupation mémoire, etc.) sur le matériel courant pour différentes configurations, offrant ainsi des références pour le déploiement.**

- **Corrections de bugs :**
    - Correction d’un problème d’enregistrement des journaux d’entraînement du modèle.
    - Mise à jour de la partie augmentation de données du modèle de formule pour garantir la compatibilité avec les nouvelles versions de la dépendance albumentations, et correction d’un avertissement de blocage lors de l’utilisation du package tokenizers en mode multiprocessus.
    - Correction de l’incohérence du comportement de certains interrupteurs (comme `use_chart_parsing`) dans les fichiers de configuration de PP-StructureV3 par rapport aux autres chaînes de production.

- **Autres améliorations :**
    - **Séparation des dépendances essentielles et optionnelles : seules les dépendances de base sont nécessaires pour la reconnaissance de texte, tandis que les fonctionnalités avancées (analyse documentaire, extraction d’information, etc.) requièrent l’installation de dépendances supplémentaires selon les besoins.**
    - **Prise en charge des cartes graphiques NVIDIA série 50 sous Windows ; les utilisateurs peuvent consulter le [guide d’installation](../docs/version3.x/installation.en.md) pour installer la version appropriée du framework Paddle.**
    - **Les modèles de la série PP-OCR peuvent désormais retourner les coordonnées de chaque caractère individuellement.**
    - Ajout de nouvelles sources de téléchargement des modèles, telles qu’AIStudio et ModelScope, avec la possibilité de spécifier la source désirée.
    - Ajout du support pour la conversion de graphique vers tableau via le module PP-Chart2Table.
    - Optimisation de certaines descriptions de la documentation pour améliorer la facilité d’utilisation.

#### **15/08/2025 : Sortie de PaddleOCR 3.1.1**, comprend :

- **Corrections de bugs :**
  - Ajout des méthodes manquantes `save_vector`, `save_visual_info_list`, `load_vector`, `load_visual_info_list` à la classe `PP-ChatOCRv4`.
  - Ajout des paramètres manquants `glossary` et `llm_request_interval` à la méthode `translate` de la classe `PPDocTranslation`.

- **Optimisation de la documentation :**
  - Ajout d’une démo à la documentation MCP.
  - Ajout des précisions sur les versions du framework PaddlePaddle et de PaddleOCR utilisées pour les tests des indicateurs de performance.
  - Correction des erreurs et oublis dans la documentation de la ligne de production de traduction de documents.

- **Autres :**
  - Modification des dépendances du serveur MCP : utilisation de la bibliothèque pure Python `puremagic` à la place de `python-magic` pour réduire les problèmes d'installation.
  - Retest des indicateurs de performance de PP-OCRv5 avec la version 3.1.0 de PaddleOCR et mise à jour de la documentation.

#### **29/06/2025 : Sortie de PaddleOCR 3.1.0**, comprend :

- **Modèles et pipelines principaux :**
  - **Ajout du modèle de reconnaissance de texte multilingue PP-OCRv5**, prenant en charge l'entraînement et l'inférence pour 37 langues, dont le français, l'espagnol, le portugais, le russe, le coréen, etc. **Précision moyenne améliorée de plus de 30 %.** [Détails](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)
  - Mise à niveau du **modèle PP-Chart2Table** dans PP-StructureV3, améliorant davantage la conversion des graphiques en tableaux. Sur des ensembles d'évaluation internes personnalisés, la métrique (RMS-F1) **a augmenté de 9,36 points de pourcentage (71,24 % -> 80,60 %).**
  - Lancement du **pipeline de traduction de documents, PP-DocTranslation, basé sur PP-StructureV3 et ERNIE 4.5**, prenant en charge la traduction des documents au format Markdown, des PDF à mise en page complexe, et des images de documents, avec sauvegarde des résultats au format Markdown. [Détails](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/PP-DocTranslation.html)

- **Nouveau serveur MCP :** [Details](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/mcp_server.html)
  - **Prend en charge les pipelines OCR et PP-StructureV3.**
  - Prend en charge trois modes de fonctionnement : bibliothèque Python locale, service cloud communautaire AIStudio et service auto-hébergé.
  - Prend en charge l'appel des services locaux via stdio et des services distants via Streamable HTTP.

- **Optimisation de la documentation :** Amélioration des descriptions dans certains guides utilisateurs pour une expérience de lecture plus fluide.


<details>
    <summary><strong>Historique des mises à jour</strong></summary>

#### **26/06/2025 : Publication de PaddleOCR 3.0.3, incluant :**

- Correction de bug : Résolution du problème où le paramètre `enable_mkldnn` ne fonctionnait pas, rétablissant le comportement par défaut d'utilisation de MKL-DNN pour l'inférence CPU.

#### 🔥🔥**19/06/2025 : Publication de PaddleOCR 3.0.2, incluant :**

- **Nouvelles fonctionnalités :**

  - La source de téléchargement par défaut a été changée de `BOS` à `HuggingFace`. Les utilisateurs peuvent également changer la variable d'environnement `PADDLE_PDX_MODEL_SOURCE` en `BOS` pour rétablir la source de téléchargement sur Baidu Object Storage (BOS).
  - Ajout d'exemples d'appel de service pour six langues — C++, Java, Go, C#, Node.js et PHP — pour les pipelines tels que PP-OCRv5, PP-StructureV3 et PP-ChatOCRv4.
  - Amélioration de l'algorithme de tri de partition de mise en page dans le pipeline PP-StructureV3, améliorant la logique de tri pour les mises en page verticales complexes afin de fournir de meilleurs résultats.
  - Logique de sélection de modèle améliorée : lorsqu'une langue est spécifiée mais pas une version de modèle, le système sélectionnera automatiquement la dernière version du modèle prenant en charge cette langue. 
  - Définition d'une limite supérieure par défaut pour la taille du cache MKL-DNN afin d'éviter une croissance illimitée, tout en permettant aux utilisateurs de configurer la capacité du cache.
  - Mise à jour des configurations par défaut pour l'inférence haute performance afin de prendre en charge l'accélération Paddle MKL-DNN et optimisation de la logique de sélection automatique de la configuration pour des choix plus intelligents.
  - Ajustement de la logique d'obtention du périphérique par défaut pour tenir compte du support réel des dispositifs de calcul par le framework Paddle installé, rendant le comportement du programme plus intuitif.
  - Ajout d'un exemple Android pour PP-OCRv5. [Détails](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/on_device_deployment.html).

- **Corrections de bugs :**

  - Correction d'un problème où certains paramètres CLI dans PP-StructureV3 ne prenaient pas effet.
  - Résolution d'un problème où `export_paddlex_config_to_yaml` ne fonctionnait pas correctement dans certains cas.
  - Correction de l'écart entre le comportement réel de `save_path` et sa description dans la documentation.
  - Correction d'erreurs potentielles de multithreading lors de l'utilisation de MKL-DNN dans le déploiement de services de base.
  - Correction des erreurs d'ordre des canaux dans le prétraitement des images pour le modèle Latex-OCR.
  - Correction des erreurs d'ordre des canaux lors de la sauvegarde des images visualisées dans le module de reconnaissance de texte.
  - Résolution des erreurs d'ordre des canaux dans les résultats de tableaux visualisés dans le pipeline PP-StructureV3.
  - Correction d'un problème de débordement dans le calcul de `overlap_ratio` dans des circonstances très spéciales dans le pipeline PP-StructureV3.

- **Améliorations de la documentation :**

  - Mise à jour de la description du paramètre `enable_mkldnn` dans la documentation pour refléter précisément le comportement réel du programme.
  - Correction d'erreurs dans la documentation concernant les paramètres `lang` et `ocr_version`.
  - Ajout d'instructions pour l'exportation des fichiers de configuration de la ligne de production via CLI.
  - Correction des colonnes manquantes dans le tableau de données de performance pour PP-OCRv5.
  - Affinement des métriques de benchmark pour PP-StructureV3 pour différentes configurations.

- **Autres :**

  - Assouplissement des restrictions de version sur les dépendances comme numpy et pandas, restaurant la prise en charge de Python 3.12.

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
   2. 💻 Prise en charge native de **ERNIE 4.5**, avec une compatibilité pour les déploiements de grands modèles via PaddleNLP, Ollama, vLLM, et plus encore.
   3. 🤝 Intégration de [PP-DocBee2](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee2), permettant l'extraction et la compréhension de texte imprimé, d'écriture manuscrite, de sceaux, de tableaux, de graphiques et d'autres éléments courants dans les documents complexes.

[Historique des mises à jour](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)

</details>

## ⚡ Démarrage Rapide
### 1. Lancer la démo en ligne
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

### 2. Installation

Installez PaddlePaddle en vous référant au [Guide d'installation](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html), puis installez la boîte à outils PaddleOCR.

```bash
# Si vous souhaitez uniquement utiliser la fonction de reconnaissance de texte de base (retourne les coordonnées de position et le contenu du texte), y compris la série PP-OCR
python -m pip install paddleocr
# Si vous souhaitez utiliser toutes les fonctionnalités telles que l’analyse de documents, la compréhension de documents, la traduction de documents, l’extraction d’informations clés, etc.
# python -m pip install "paddleocr[all]"
```

À partir de la version 3.2.0, en plus du groupe de dépendances `all` présenté ci-dessus, PaddleOCR prend également en charge l’installation de certaines fonctionnalités optionnelles en spécifiant d’autres groupes de dépendances. Voici tous les groupes de dépendances proposés par PaddleOCR :

| Nom du groupe de dépendances | Fonctionnalité correspondante |
| - | - |
| `doc-parser` | Analyse de documents : permet d’extraire des éléments de mise en page tels que tableaux, formules, tampons, images, etc. à partir des documents ; inclut des modèles comme PP-StructureV3 |
| `ie` | Extraction d’informations : permet d’extraire des informations clés des documents, telles que noms, dates, adresses, montants, etc. ; inclut des modèles comme PP-ChatOCRv4 |
| `trans` | Traduction de documents : permet de traduire des documents d’une langue à une autre ; inclut des modèles comme PP-DocTranslation |
| `all` | Fonctionnalité complète |

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

## 🧩 Fonctionnalités supplémentaires

- Convertir les modèles au format ONNX : [Obtention des modèles ONNX](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/obtaining_onnx_models.html).
- Accélérer l'inférence avec des moteurs comme OpenVINO, ONNX Runtime, TensorRT, ou effectuer l'inférence avec des modèles au format ONNX : [Inférence haute performance](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/high_performance_inference.html).
- Accélérer l'inférence en utilisant plusieurs GPU et plusieurs processus : [Inférence parallèle pour pipelines](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/instructions/parallel_inference.html).
- Intégrez PaddleOCR dans des applications écrites en C++, C#, Java, etc. : [Service](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/serving.html).

## ⛰️ Tutoriels avancés
- [Tutoriel PP-OCRv5](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/OCR.html)
- [Tutoriel PP-StructureV3](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-StructureV3.html)
- [Tutoriel PP-ChatOCRv4](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-ChatOCRv4.html)

## 🔄 Aperçu rapide des résultats d'exécution

<div align="center">
  <p>
     <img width="100%" src="../docs/images/demo.gif" alt="Démo PP-OCRv5">
  </p>
</div>

<div align="center">
  <p>
     <img width="100%" src="../docs/images/blue_v3.gif" alt="Démo PP-StructureV3">
  </p>
</div>

## 🌟 Restez à l'écoute

⭐ **Ajoutez une étoile à ce dépôt pour suivre les mises à jour passionnantes et les nouvelles versions, y compris de puissantes fonctionnalités d'OCR et d'analyse de documents !** ⭐

<div align="center">
  <p>
       <img width="1200" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/star_paddleocr.en.gif" alt="Star-Project">
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
| [En savoir plus](../awesome_projects.md) | [Plus de projets basés sur PaddleOCR](../awesome_projects.md)|

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
