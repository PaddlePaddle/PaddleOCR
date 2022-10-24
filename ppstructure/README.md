English | [简体中文](README_ch.md)

# PP-Structure

- [1. Introduction](#1-introduction)
- [2. Features](#2-features)
- [3. Results](#3-results)
  - [3.1 Layout analysis and table recognition](#31-layout-analysis-and-table-recognition)
  - [3.2 Layout Recovery](#32-layout-recovery)
  - [3.3 KIE](#33-kie)
- [4. Quick start](#4-quick-start)
- [5. Model List](#5-model-list)

## 1. Introduction

PP-Structure is an intelligent document analysis system developed by the PaddleOCR team, which aims to help developers better complete tasks related to document understanding such as layout analysis and table recognition.

The pipeline of PP-StructureV2 system is shown below. The document image first passes through the image direction correction module to identify the direction of the entire image and complete the direction correction. Then, two tasks of layout information analysis and key information extraction can be completed.

- In the layout analysis task, the image first goes through the layout analysis model to divide the image into different areas such as text, table, and figure, and then analyze these areas separately. For example, the table area is sent to the form recognition module for structured recognition, and the text area is sent to the OCR engine for text recognition. Finally, the layout recovery module restores it to a word or pdf file with the same layout as the original image;
- In the key information extraction task, the OCR engine is first used to extract the text content, and then the SER(semantic entity recognition) module obtains the semantic entities in the image, and finally the RE(relationship extraction) module obtains the correspondence between the semantic entities, thereby extracting the required key information.
<img src="https://user-images.githubusercontent.com/14270174/195265734-6f4b5a7f-59b1-4fcc-af6d-89afc9bd51e1.jpg" width="100%"/>

More technical details: 👉 [PP-StructureV2 Technical Report](https://arxiv.org/abs/2210.05391)

PP-StructureV2 supports independent use or flexible collocation of each module. For example, you can use layout analysis alone or table recognition alone. Click the corresponding link below to get the tutorial for each independent module:

- [Layout Analysis](layout/README.md)
- [Table Recognition](table/README.md)
- [Key Information Extraction](kie/README.md)
- [Layout Recovery](recovery/README.md)

## 2. Features

The main features of PP-StructureV2 are as follows:
- Support layout analysis of documents in the form of images/pdfs, which can be divided into areas such as **text, titles, tables, figures, formulas, etc.**;
- Support common Chinese and English **table detection** tasks;
- Support structured table recognition, and output the final result to **Excel file**;
- Support multimodal-based Key Information Extraction (KIE) tasks - **Semantic Entity Recognition** (SER) and **Relation Extraction (RE);
- Support **layout recovery**, that is, restore the document in word or pdf format with the same layout as the original image;
- Support customized training and multiple inference deployment methods such as python whl package quick start;
- Connect with the semi-automatic data labeling tool PPOCRLabel, which supports the labeling of layout analysis, table recognition, and SER.

## 3. Results

PP-StructureV2 supports the independent use or flexible collocation of each module. For example, layout analysis can be used alone, or table recognition can be used alone. Only the visualization effects of several representative usage methods are shown here.

### 3.1 Layout analysis and table recognition

The figure shows the pipeline of layout analysis + table recognition. The image is first divided into four areas of image, text, title and table by layout analysis, and then OCR detection and recognition is performed on the three areas of image, text and title, and the table is performed table recognition, where the image will also be stored for use.
<img src="docs/table/ppstructure.GIF" width="100%"/>

### 3.2 Layout recovery

The following figure shows the effect of layout recovery based on the results of layout analysis and table recognition in the previous section.
<img src="./docs/recovery/recovery.jpg" width="100%"/>

### 3.3 KIE

* SER

Different colored boxes in the figure represent different categories.

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/197464552-69de557f-edff-4c7f-acbf-069df1ba097f.png" width="600">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/25809855/186095702-9acef674-12af-4d09-97fc-abf4ab32600e.png" width="600">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185539141-68e71c75-5cf7-4529-b2ca-219d29fa5f68.jpg" width="600">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185310636-6ce02f7c-790d-479f-b163-ea97a5a04808.jpg" width="600">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185539517-ccf2372a-f026-4a7c-ad28-c741c770f60a.png" width="600">
</div>

* RE

In the figure, the red box represents `Question`, the blue box represents `Answer`, and `Question` and `Answer` are connected by green lines.

<div align="center">
    <img src="https://user-images.githubusercontent.com/25809855/186094813-3a8e16cc-42e5-4982-b9f4-0134dfb5688d.png" width="600">
</div>  

<div align="center">
    <img src="https://user-images.githubusercontent.com/25809855/186095641-5843b4da-34d7-4c1c-943a-b1036a859fe3.png" width="600">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185393805-c67ff571-cf7e-4217-a4b0-8b396c4f22bb.jpg" width="600">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185540080-0431e006-9235-4b6d-b63d-0b3c6e1de48f.jpg" width="600">
</div>

## 4. Quick start

Start from [Quick Start](./docs/quickstart_en.md).

## 5. Model List

Some tasks need to use both the structured analysis models and the OCR models. For example, the table recognition task needs to use the table recognition model for structured analysis, and the OCR model to recognize the text in the table. Please select the appropriate models according to your specific needs.

For structural analysis related model downloads, please refer to:
- [PP-Structure Model Zoo](./docs/models_list_en.md)

For OCR related model downloads, please refer to:
- [PP-OCR Model Zoo](../doc/doc_en/models_list_en.md)
