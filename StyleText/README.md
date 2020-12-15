English | [简体中文](README_ch.md)

### Contents
- [1. Introduction](#Introduction)
- [2. Preparation](#Preparation)
- [3. Demo](#Demo)
- [4. Advanced Usage](#Advanced_Usage)
- [5. Code Structure](#Code_structure)


<a name="Introduction"></a>
### Introduction

<div align="center">
    <img src="doc/images/3.png" width="800">
</div>

<div align="center">
    <img src="doc/images/1.png" width="600">
</div>


The Style-Text data synthesis tool is a tool based on Baidu's self-developed text editing algorithm "Editing Text in the Wild" [https://arxiv.org/abs/1908.03047](https://arxiv.org/abs/1908.03047).

Different from the commonly used GAN-based data synthesis tools, the main framework of Style-Text includes:
* (1) Text foreground style transfer module.
* (2) Background extraction module.
* (3) Fusion module.

After these three steps, you can quickly realize the image text style transfer. The following figure is some results of the data synthesis tool.

<div align="center">
    <img src="doc/images/2.png" width="1000">
</div>


<a name="Preparation"></a>
#### Preparation

1. Please refer the [QUICK INSTALLATION](../doc/doc_en/installation_en.md) to install PaddlePaddle. Python3 environment is strongly recommended.
2. Download the pretrained models and unzip:

```bash
cd StyleText
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/style_text/style_text_models.zip
unzip style_text_models.zip
```

If you save the model in another location, please modify the address of the model file in `configs/config.yml`, and you need to modify these three configurations at the same time:

```
bg_generator:
  pretrain: style_text_rec/bg_generator
...
text_generator:
  pretrain: style_text_models/text_generator
...
fusion_generator:
  pretrain: style_text_models/fusion_generator
```

<a name="Demo"></a>
### Demo

#### Synthesis single image

1. You can run `tools/synth_image` and generate the demo image.

```python
python3 -m tools.synth_image -c configs/config.yml
```

2. The results are `fake_bg.jpg`, `fake_text.jpg` and `fake_fusion.jpg` as shown in the figure above. Above them:
   * `fake_text.jpg` is the generated image with the same font style as `Style Input`;
   * `fake_bg.jpg` is the generated image of `Style Input` after removing foreground.
   * `fake_fusion.jpg` is the final result, that is synthesised by `fake_text.jpg` and `fake_bg.jpg`.  

3. If want to generate image by other `Style Input` or `Text Input`, you can modify the `tools/synth_image.py`:
   * `img = cv2.imread("examples/style_images/1.jpg")`: the path of `Style Input`;
   * `corpus = "PaddleOCR"`: the `Text Input`;
   * Notice：modify the language option(`language = "en"`) to adapt `Text Input`, that support `en`, `ch`, `ko`.

4. We also provide `batch_synth_images` mothod, that can combine corpus and pictures in pairs to generate a batch of data.


#### Batch synthesis

Before the start, you need to prepare some data as material.
First, you should have the style reference data for synthesis tasks, which are generally used as datasets for OCR recognition tasks.

1. The referenced dataset can be specifed in `configs/dataset_config.yml`:
   * `StyleSampler`:
     * `method`: The method of `StyleSampler`.
     * `image_home`: The directory of pictures.
     * `label_file`: The list of pictures path if `with_label` is `false`, otherwise, the label file path.
     * `with_label`: The `label_file` is label file or not.

   * `CorpusGenerator`:
     * `method`: The mothod of `CorpusGenerator`. If `FileCorpus` used, you need modify `corpus_file` and `language` accordingly, if `EnNumCorpus`, other configurations is not needed.
     * `language`: The language of the corpus. Needed if method is not `EnNumCorpus`.
     * `corpus_file`: The corpus file path. Needed if method is not `EnNumCorpus`.

We provide a general dataset containing Chinese, English and Korean (50,000 images in all) for your trial ([download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/style_text/chkoen_5w.tar)), some examples are given below :

<div align="center">
     <img src="doc/images/5.png" width="800">
</div>

2. You can run the following command to start synthesis task:

   ``` bash
   python -m tools.synth_dataset.py -c configs/dataset_config.yml
   ```

3. You can using the following command to start multiple synthesis tasks in a multi-threaded manner, which needed to specifying tags by `-t`:

   ```bash
   python -m tools.synth_dataset.py -t 0 -c configs/dataset_config.yml
   python -m tools.synth_dataset.py -t 1 -c configs/dataset_config.yml
   ```




<a name="Advanced_Usage"></a>
### Advanced Usage
We take two scenes as examples, which are metal surface English number recognition and general Korean recognition, to illustrate practical cases of using StyleText to synthesize data to improve text recognition. The following figure shows some examples of real scene images and composite images:

<div align="center">
    <img src="doc/images/6.png" width="800">
</div>


After adding the above synthetic data for training, the accuracy of the recognition model is improved, which is shown in the following table:

| Scenario | Characters | Raw Data | Test Data | Only Use Raw Data</br>Recognition Accuracy | New Synthetic Data | Simultaneous Use of Synthetic Data</br>Recognition Accuracy | Index Improvement |
| -------- | ---------- | -------- | -------- | ----------- --------------- | ------------ | --------------------- -| -------- |
| Metal surface | English and numbers | 2203 | 650 | 0.5938 | 20000 | 0.7546 | 16% |
| Random background | Korean | 5631 | 1230 | 0.3012 | 100000 | 0.5057 | 20% |


<a name="Code_structure"></a>
### Code Structure
```
style_text_rec
|-- arch
|   |-- base_module.py
|   |-- decoder.py
|   |-- encoder.py
|   |-- spectral_norm.py
|   `-- style_text_rec.py
|-- configs
|   |-- config.yml
|   `-- dataset_config.yml
|-- engine
|   |-- corpus_generators.py
|   |-- predictors.py
|   |-- style_samplers.py
|   |-- synthesisers.py
|   |-- text_drawers.py
|   `-- writers.py
|-- examples
|   |-- corpus
|   |   `-- example.txt
|   |-- image_list.txt
|   `-- style_images
|       |-- 1.jpg
|       `-- 2.jpg
|-- fonts
|   |-- ch_standard.ttf
|   |-- en_standard.ttf
|   `-- ko_standard.ttf
|-- tools
|   |-- __init__.py
|   |-- synth_dataset.py
|   `-- synth_image.py
`-- utils
    |-- config.py
    |-- load_params.py
    |-- logging.py
    |-- math_functions.py
    `-- sys_funcs.py
```
