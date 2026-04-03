<div align="center">
  <p>
      <img width="800" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Banner.png" alt="Star-history">
  </p>



<h3>Boîte à outils OCR de pointe mondiale & Moteur d'IA documentaire</h3>

[English](../README.md) | [简体中文](./README_cn.md) | [繁體中文](./README_tcn.md) | [日本語](./README_ja.md) | [한국어](./README_ko.md) | Français | [Русский](./README_ru.md) | [Español](./README_es.md) | [العربية](./README_ar.md)

<!-- icon -->

[![PyPI Downloads](https://static.pepy.tech/badge/paddleocr)](https://pepy.tech/projects/paddleocr)
[![Used by](https://img.shields.io/badge/Used%20by-6k%2B%20repositories-blue)](https://github.com/PaddlePaddle/PaddleOCR/network/dependents)
![python](https://img.shields.io/badge/python-3.8~3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
![hardware](https://img.shields.io/badge/hardware-cpu%2C%20gpu%2C%20xpu%2C%20npu-yellow.svg)

[![AI Studio](https://img.shields.io/badge/PaddleOCR-_Site_Officiel-1927BA?logo=data:image/png;base64,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&labelColor=white)](https://www.paddleocr.com)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PaddlePaddle/PaddleOCR)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](../LICENSE)

</div>






**PaddleOCR convertit des documents et des images en données structurées prêtes pour les LLM (JSON/Markdown) avec une précision de pointe dans l'industrie. Avec plus de 70k étoiles et la confiance de projets de premier plan tels que Dify, RAGFlow et Cherry Studio, PaddleOCR est le socle fondamental pour construire des applications RAG intelligentes et des applications Agentiques.**


## 🚀 Fonctionnalités clés

### 📄 Analyse intelligente de documents (prêt pour les LLM)
> *Transformer des visuels désordonnés en données structurées pour l'ère des LLM.*

* **VLM documentaire de pointe** : Avec **PaddleOCR-VL-1.5 (0,9 milliard de paramètres)**, le modèle vision-langage léger de pointe de l'industrie pour l'analyse de documents. Il excelle dans l'analyse de documents complexes face à 5 grands défis du « monde réel » : **Déformation, Numérisation, Photographie d'écran, Éclairage et Documents inclinés**, avec des sorties structurées aux formats **Markdown** et **JSON**.
* **Conversion avec conscience de la structure** : Propulsé par **PP-StructureV3**, convertissez sans effort des PDF et images complexes en **Markdown** ou **JSON**. Contrairement aux modèles de la série PaddleOCR-VL, il fournit des informations de coordonnées plus fines, incluant les coordonnées des cellules de tableau, les coordonnées du texte, et bien plus encore.
* **Efficacité prête pour la production** : Atteignez une précision de niveau commercial avec une empreinte ultra-réduite. Surpasse de nombreuses solutions propriétaires sur les benchmarks publics tout en restant économe en ressources pour le déploiement en périphérie ou dans le cloud.

### 🔍 Reconnaissance de texte universelle (OCR de scène)
> *L'étalon-or mondial pour la détection de texte multilingue à haute vitesse.*

* **Plus de 100 langues supportées** : Reconnaissance native pour une vaste bibliothèque mondiale. Notre solution **PP-OCRv5** à modèle unique gère élégamment les documents multilingues mixtes (chinois, anglais, japonais, pinyin, etc.).
* **Maîtrise des éléments complexes** : Au-delà de la reconnaissance de texte standard, nous prenons en charge la **détection de texte en scène naturelle** dans une large gamme d'environnements, y compris les pièces d'identité, les vues de rue, les livres et les composants industriels.
* **Bond en performance** : PP-OCRv5 apporte une **amélioration de la précision de 13%** par rapport aux versions précédentes, tout en maintenant l'« Efficacité extrême » pour laquelle PaddleOCR est célèbre.

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Arch.jpg" alt="Architecture PaddleOCR">
  </p>
</div>

### 🛠️ Écosystème centré sur les développeurs
* **Intégration transparente** : Le premier choix pour l'écosystème des agents IA — profondément intégré avec **Dify, RAGFlow, Pathway et Cherry Studio**.
* **Volant de données pour LLM** : Un pipeline complet pour construire des jeux de données de haute qualité, fournissant un « Moteur de données » durable pour l'affinage des grands modèles de langage.
* **Déploiement en un clic** : Prend en charge divers backends matériels (GPU NVIDIA, CPU Intel, XPU Kunlunxin et divers accélérateurs IA).


## 📣 Mises à jour récentes

### 🔥 [2026.01.29] PaddleOCR v3.4.0 publié : L'ère de l'analyse de documents irréguliers
* **PaddleOCR-VL-1.5 (VLM 0,9 milliard de paramètres, état de l'art)** : Notre dernier modèle phare pour l'analyse de documents est désormais disponible !
    * **94,5 % de précision sur OmniDocBench** : Surpasse les grands modèles généralistes de premier rang et les analyseurs de documents spécialisés.
    * **Robustesse dans le monde réel** : Premier à introduire l'algorithme **PP-DocLayoutV3** pour le positionnement de formes irrégulières, maîtrisant 5 scénarios difficiles : *Inclinaison, Déformation, Numérisation, Éclairage et Photographie d'écran*.
    * **Extension des capacités** : Prend désormais en charge la **Reconnaissance de sceaux**, la **Détection de texte**, et s'étend à **111 langues** (incluant le tibétain et le bengali).
    * **Maîtrise des longs documents** : Prend en charge la fusion automatique de tableaux sur plusieurs pages et l'identification hiérarchique des titres.
    * **Essayez-le maintenant** : Disponible sur [HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) ou sur notre [Site officiel](https://www.paddleocr.com).

<details>
<summary><strong>2025.10.16 : Publication de PaddleOCR 3.3.0</strong></summary>

- Publication de PaddleOCR-VL :
    - **Présentation du modèle** :
        - **PaddleOCR-VL** est un modèle de pointe et économe en ressources, spécialement conçu pour l'analyse de documents. Son composant principal est PaddleOCR-VL-0.9B, un modèle vision-langage (VLM) compact mais puissant qui intègre un encodeur visuel à résolution dynamique de style NaViT avec le modèle de langage ERNIE-4.5-0.3B pour permettre une reconnaissance précise des éléments. **Ce modèle innovant prend en charge efficacement 109 langues et excelle dans la reconnaissance d'éléments complexes (par exemple, texte, tableaux, formules et graphiques), tout en maintenant une consommation minimale de ressources**. Grâce à des évaluations complètes sur des benchmarks publics largement utilisés et des benchmarks internes, PaddleOCR-VL atteint des performances de pointe à la fois dans l'analyse de documents au niveau de la page et dans la reconnaissance au niveau des éléments. Il surpasse significativement les solutions existantes, présente une forte compétitivité face aux VLM de premier plan, et offre des vitesses d'inférence rapides. Ces atouts le rendent très adapté au déploiement pratique dans des scénarios du monde réel. Le modèle a été publié sur [HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL). Tout le monde est invité à le télécharger et à l'utiliser ! Plus d'informations d'introduction sont disponibles dans [PaddleOCR-VL](https://www.paddleocr.ai/latest/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.html).

    - **Fonctionnalités principales** :
        - **Architecture VLM compacte mais puissante** : Nous présentons un nouveau modèle vision-langage spécialement conçu pour une inférence économe en ressources, atteignant des performances remarquables dans la reconnaissance d'éléments. En intégrant un encodeur visuel haute résolution dynamique de style NaViT avec le modèle de langage léger ERNIE-4.5-0.3B, nous améliorons considérablement les capacités de reconnaissance et l'efficacité du décodage du modèle. Cette intégration maintient une haute précision tout en réduisant les besoins de calcul, ce qui le rend bien adapté aux applications de traitement de documents efficaces et pratiques.
        - **Performances de pointe en analyse de documents** : PaddleOCR-VL atteint des performances à l'état de l'art à la fois dans l'analyse de documents au niveau de la page et dans la reconnaissance au niveau des éléments. Il surpasse significativement les solutions existantes basées sur des pipelines et présente une forte compétitivité face aux principaux modèles vision-langage (VLM) en analyse de documents. De plus, il excelle dans la reconnaissance d'éléments documentaires complexes, tels que le texte, les tableaux, les formules et les graphiques, ce qui le rend adapté à une large gamme de types de contenu difficiles, y compris le texte manuscrit et les documents historiques. Cela le rend très polyvalent et adapté à une large gamme de types de documents et de scénarios.
        - **Support multilingue** : PaddleOCR-VL prend en charge 109 langues, couvrant les principales langues mondiales, notamment le chinois, l'anglais, le japonais, le latin et le coréen, ainsi que les langues avec des scripts et des structures différents, tels que le russe (script cyrillique), l'arabe, l'hindi (script devanagari) et le thaï. Cette large couverture linguistique améliore considérablement l'applicabilité de notre système aux scénarios de traitement de documents multilingues et mondialisés.

- Publication du modèle de reconnaissance multilingue PP-OCRv5 :
    - Amélioration de la précision et de la couverture de la reconnaissance des scripts latins ; ajout de la prise en charge des systèmes cyrillique, arabe, devanagari, télougou, tamoul et d'autres systèmes linguistiques, couvrant la reconnaissance de 109 langues. Le modèle ne compte que 2 millions de paramètres, et la précision de certains modèles a augmenté de plus de 40 % par rapport à la génération précédente.

</details>


<details>
<summary><strong>2025.08.21 : Publication de PaddleOCR 3.2.0</strong></summary>

- **Ajouts significatifs de modèles :**
    - Introduction de l'entraînement, de l'inférence et du déploiement pour les modèles de reconnaissance PP-OCRv5 en anglais, thaï et grec. **Le modèle PP-OCRv5 anglais apporte une amélioration de 11 % dans les scénarios en anglais par rapport au modèle principal PP-OCRv5, avec les modèles de reconnaissance thaï et grec atteignant des précisions de 82,68 % et 89,28 %, respectivement.**

- **Améliorations des capacités de déploiement :**
    - **Prise en charge complète des versions 3.1.0 et 3.1.1 du framework PaddlePaddle.**
    - **Mise à niveau complète de la solution de déploiement local C++ de PP-OCRv5, prenant désormais en charge Linux et Windows, avec une parité de fonctionnalités et une précision identique à l'implémentation Python.**
    - **L'inférence haute performance prend désormais en charge CUDA 12, et l'inférence peut être effectuée en utilisant le backend Paddle Inference ou ONNX Runtime.**
    - **La solution de déploiement orientée services à haute stabilité est désormais entièrement open-source, permettant aux utilisateurs de personnaliser les images Docker et les SDK selon leurs besoins.**
    - La solution de déploiement orientée services à haute stabilité prend également en charge l'invocation via des requêtes HTTP construites manuellement, permettant le développement de code client dans n'importe quel langage de programmation.

- **Support des benchmarks :**
    - **Toutes les lignes de production prennent désormais en charge des benchmarks granulaires, permettant la mesure du temps d'inférence de bout en bout ainsi que les données de latence par couche et par module pour faciliter l'analyse des performances. [Voici](docs/version3.x/pipeline_usage/instructions/benchmark.en.md) comment configurer et utiliser la fonctionnalité de benchmark.**
    - **La documentation a été mise à jour pour inclure les métriques clés pour les configurations couramment utilisées sur le matériel grand public, telles que la latence d'inférence et l'utilisation de la mémoire, fournissant des références de déploiement pour les utilisateurs.**

- **Corrections de bugs :**
    - Résolution du problème de l'échec de sauvegarde des journaux lors de l'entraînement du modèle.
    - Mise à niveau du composant d'augmentation de données pour les modèles de formules pour la compatibilité avec les nouvelles versions de la dépendance albumentations, et correction des avertissements de blocage lors de l'utilisation du package tokenizers dans des scénarios multi-processus.
    - Correction des incohérences dans les comportements des commutateurs (par exemple, `use_chart_parsing`) dans les fichiers de configuration PP-StructureV3 par rapport aux autres pipelines.

- **Autres améliorations :**
    - **Séparation des dépendances principales et optionnelles. Seules les dépendances principales minimales sont requises pour la reconnaissance de texte de base ; des dépendances supplémentaires pour l'analyse de documents et l'extraction d'informations peuvent être installées selon les besoins.**
    - **Activation de la prise en charge des cartes graphiques NVIDIA RTX série 50 sous Windows ; les utilisateurs peuvent consulter le [guide d'installation](docs/version3.x/installation.en.md) pour les versions correspondantes du framework PaddlePaddle.**
    - **Les modèles de la série PP-OCR prennent désormais en charge le retour des coordonnées de chaque caractère.**
    - Ajout de sources de téléchargement de modèles AIStudio, ModelScope et autres, permettant aux utilisateurs de spécifier la source pour les téléchargements de modèles.
    - Ajout de la prise en charge de la conversion graphique en tableau via le module PP-Chart2Table.
    - Optimisation des descriptions de documentation pour améliorer la facilité d'utilisation.
</details>


[Journal des modifications](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)

</details>


## 🚀 Démarrage rapide

### Étape 1 : Essayer en ligne
Le site officiel de PaddleOCR propose un **Centre d'expérience** interactif et des **API** — aucune configuration requise, un seul clic pour découvrir.

👉 [Visiter le site officiel](https://www.paddleocr.com)

### Étape 2 : Déploiement local
Pour une utilisation locale, veuillez consulter la documentation suivante en fonction de vos besoins :

- **Série PP-OCR** : Voir la [documentation PP-OCR](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html)
- **Série PaddleOCR-VL** : Voir la [documentation PaddleOCR-VL](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- **PP-StructureV3** : Voir la [documentation PP-StructureV3](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PP-StructureV3.html)
- **Autres capacités** : Voir la [documentation sur les autres capacités](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/pipeline_overview.html)


## 🧩 Plus de fonctionnalités

- Convertir des modèles au format ONNX : [Obtenir des modèles ONNX](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/obtaining_onnx_models.html).
- Accélérer l'inférence à l'aide de moteurs tels qu'OpenVINO, ONNX Runtime, TensorRT, ou effectuer une inférence à l'aide de modèles au format ONNX : [Inférence haute performance](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/high_performance_inference.html).
- Accélérer l'inférence à l'aide de plusieurs GPU et plusieurs processus : [Inférence parallèle pour les pipelines](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/instructions/parallel_inference.html).
- Intégrer PaddleOCR dans des applications écrites en C++, C#, Java, etc. : [Services](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/serving.html).

## 🔄 Aperçu rapide des résultats d'exécution

### PP-OCRv5

<div align="center">
  <p>
       <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PP-OCRv5_demo.gif" alt="Démo PP-OCRv5">
  </p>
</div>



### PP-StructureV3

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PP-StructureV3_demo.gif" alt="Démo PP-StructureV3">
  </p>
</div>

### PaddleOCR-VL

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PaddleOCR-VL_demo.gif" alt="Démo PP-StructureV3">
  </p>
</div>


## ✨ Restez informé

⭐ **Mettez une étoile à ce dépôt pour suivre les mises à jour passionnantes et les nouvelles versions, y compris les puissantes capacités d'OCR et d'analyse de documents !** ⭐

<div align="center">
  <p>
       <img width="1200" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/star_paddleocr2.en.gif" alt="Star-Project">
  </p>
</div>


## 👩‍👩‍👧‍👦 Communauté

<div align="center">

| Compte officiel WeChat de PaddlePaddle | Rejoindre le groupe de discussion technique |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |
</div>


## 😃 Projets remarquables utilisant PaddleOCR
PaddleOCR n'en serait pas là aujourd'hui sans son incroyable communauté ! 💗 Un immense merci à tous nos partenaires de longue date, aux nouveaux collaborateurs et à tous ceux qui ont mis leur passion dans PaddleOCR — que nous vous ayons cités ou non. Votre soutien alimente notre feu !

<div align="center">

| Nom du projet | Description |
| ------------ | ----------- |
| [Dify](https://github.com/langgenius/dify) <a href="https://github.com/langgenius/dify"><img src="https://img.shields.io/github/stars/langgenius/dify"></a>|Plateforme prête pour la production pour le développement de flux de travail agentiques.|
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|Moteur RAG basé sur la compréhension approfondie des documents.|
| [pathway](https://github.com/pathwaycom/pathway) <a href="https://github.com/pathwaycom/pathway"><img src="https://img.shields.io/github/stars/pathwaycom/pathway"></a>|Framework Python ETL pour le traitement de flux, l'analytique en temps réel, les pipelines LLM et le RAG.|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|Outil de conversion de documents multi-types en Markdown.|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|Logiciel OCR hors ligne par lots, gratuit et open-source.|
| [cherry-studio](https://github.com/CherryHQ/cherry-studio) <a href="https://github.com/CherryHQ/cherry-studio"><img src="https://img.shields.io/github/stars/CherryHQ/cherry-studio"></a>|Un client de bureau prenant en charge plusieurs fournisseurs de LLM.|
| [haystack](https://github.com/deepset-ai/haystack)<a href="https://github.com/deepset-ai/haystack"><img src="https://img.shields.io/github/stars/deepset-ai/haystack"></a> |Framework d'orchestration IA pour construire des applications LLM personnalisables et prêtes pour la production.|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |OmniParser : Outil d'analyse d'écran pour agent GUI basé sur la vision pure.|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |Questions et réponses basées sur n'importe quoi.|
| [En savoir plus sur les projets](./awesome_projects.md) | [Plus de projets basés sur PaddleOCR](./awesome_projects.md)|
</div>

## 👩‍👩‍👧‍👦 Contributeurs

<div align="center">
<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>
</div>

## 🌟 Étoiles

<div align="center">
  <p>
      <img width="800" src="https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date" alt="Star-history">
  </p>
</div>


## 📄 Licence
Ce projet est publié sous la [licence Apache 2.0](LICENSE).

## 🎓 Citation

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
