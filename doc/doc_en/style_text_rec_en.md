### Quick Start

`Style-Text` is an improvement of the SRNet network proposed in Baidu's self-developed text editing algorithm "Editing Text in the Wild". It is different from the commonly used GAN methods. This tool decomposes the text synthesis task into three sub-modules to improve the effect of synthetic data: text style transfer module, background extraction module and fusion module. 

The following figure shows some example results. In addition, the actual `nameplate text recognition` scene and `the Korean text recognition` scene verify the effectiveness of the synthesis tool, as follows.


#### Preparation

1. Please refer the [QUICK INSTALLATION](./installation_en.md) to install PaddlePaddle. Python3 environment is strongly recommended.
2. Download the pretrained models and unzip:

```bash
cd tools/style_text_rec
wget /path/to/style_text_models.zip
unzip style_text_models.zip
```

You can dowload models [here](https://paddleocr.bj.bcebos.com/dygraph_v2.0/style_text/style_text_models.zip). If you save the model files in other folders, please edit the three model paths in `configs/config.yml`:

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



#### Demo

1. You can use the following commands to run a demo：

```bash
python -m tools.synth_image -c configs/config.yml
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

### Advanced Usage

#### Components

`Style Text Rec` mainly contains the following components：

* `style_samplers`: It can sample `Style Input` from a dataset. Now, We only provide `DatasetSampler`.

* `corpus_generators`: It can generate corpus. Now, wo only provide two `corpus_generators`:
  * `EnNumCorpus`: It can generate a random string according to a given length,  including uppercase and lowercase English letters, numbers and spaces.
  * `FileCorpus`: It can read a text file and randomly return the words in it.

* `text_drawers`: It can generate `Text Input`(text picture in standard font according to the input corpus). Note that when using, you have to modify the language information according to the corpus.

* `predictors`: It can call the deep learning model to generate new data based on the `Style Input` and `Text Input`.

* `writers`: It can write the generated pictures(`fake_bg.jpg`, `fake_text.jpg` and `fake_fusion.jpg`) and label information to the disk.

* `synthesisers`: It can call the all modules to complete the work.

### Generate Dataset

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

2. You can run the following command to start synthesis task:

   ``` bash
   python -m tools.synth_dataset.py -c configs/dataset_config.yml
   ```

3. You can using the following command to start multiple synthesis tasks in a multi-threaded manner, which needed to specifying tags by `-t`:
   
   ```bash
   python -m tools.synth_dataset.py -t 0 -c configs/dataset_config.yml
   python -m tools.synth_dataset.py -t 1 -c configs/dataset_config.yml
   ```

### OCR Recognition Training

After completing the above operations, you can get the synthetic data set for OCR recognition. Next, please complete the training by refering to [OCR Recognition Document](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/recognition. md#%E5%90%AF%E5%8A%A8%E8%AE%AD%E7%BB%83).