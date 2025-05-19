---
typora-copy-images-to: images
comments: true
---

# 场景文本识别算法-NRTR

## 1. 算法简介

论文信息：
> [NRTR: A No-Recurrence Sequence-to-Sequence Model For Scene Text Recognition](https://arxiv.org/abs/1806.00926)
> Fenfen Sheng and Zhineng Chen and Bo Xu
> ICDAR, 2019

`NRTR`使用MJSynth和SynthText两个文字识别数据集训练，在IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估，算法复现效果如下：

|模型|骨干网络|配置文件|Acc|下载链接|
| --- | --- | --- | --- | --- |
|NRTR|MTB|[rec_mtb_nrtr.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_mtb_nrtr.yml)|84.21%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mtb_nrtr_train.tar)|

## 2. 环境配置

请先参考[《运行环境准备》](../../ppocr/environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](../../ppocr/blog/clone.md)克隆项目代码。

## 3. 模型训练、评估、预测

### 3.1 模型训练

请参考[文本识别训练教程](../../ppocr/model_train/recognition.md)。PaddleOCR对代码进行了模块化，训练`NRTR`识别模型时需要**更换配置文件**为`NRTR`的[配置文件](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_mtb_nrtr.yml)。

#### 启动训练

具体地，在完成数据准备后，便可以启动训练，训练命令如下：

```bash linenums="1"
#单卡训练（训练周期长，不建议）
python3 tools/train.py -c configs/rec/rec_mtb_nrtr.yml

# 多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_mtb_nrtr.yml
```

### 3.2 评估

可下载已训练完成的模型文件，使用如下命令进行评估：

```bash linenums="1"
# 注意将pretrained_model的路径设置为本地路径。
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_mtb_nrtr.yml -o Global.pretrained_model=./rec_mtb_nrtr_train/best_accuracy
```

### 3.3 预测

使用如下命令进行单张图片预测：

```bash linenums="1"
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/infer_rec.py -c configs/rec/rec_mtb_nrtr.yml -o Global.infer_img='./doc/imgs_words_en/word_10.png' Global.pretrained_model=./rec_mtb_nrtr_train/best_accuracy
# 预测文件夹下所有图像时，可修改infer_img为文件夹，如 Global.infer_img='./doc/imgs_words_en/'。
```

## 4. 推理部署

### 4.1 Python推理

首先将训练得到best模型，转换成inference model。这里以训练完成的模型为例（[模型下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mtb_nrtr_train.tar) )，可以使用如下命令进行转换：

```bash linenums="1"
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/export_model.py -c configs/rec/rec_mtb_nrtr.yml -o Global.pretrained_model=./rec_mtb_nrtr_train/best_accuracy Global.save_inference_dir=./inference/rec_mtb_nrtr/
```

**注意：**

- 如果您是在自己的数据集上训练的模型，并且调整了字典文件，请注意修改配置文件中的`character_dict_path`是否是所需要的字典文件。
- 如果您修改了训练时的输入大小，请修改`tools/export_model.py`文件中的对应NRTR的`infer_shape`。

转换成功后，在目录下有三个文件：

```text linenums="1"
/inference/rec_mtb_nrtr/
    ├── inference.pdiparams         # 识别inference模型的参数文件
    ├── inference.pdiparams.info    # 识别inference模型的参数信息，可忽略
    └── inference.pdmodel           # 识别inference模型的program文件
```

执行如下命令进行模型推理：

```bash linenums="1"
python3 tools/infer/predict_rec.py --image_dir='./doc/imgs_words_en/word_10.png' --rec_model_dir='./inference/rec_mtb_nrtr/' --rec_algorithm='NRTR' --rec_image_shape='1,32,100' --rec_char_dict_path='./ppocr/utils/EN_symbol_dict.txt'
# 预测文件夹下所有图像时，可修改image_dir为文件夹，如 --image_dir='./doc/imgs_words_en/'。
```

![img](./images/word_10.png)

执行命令后，上面图像的预测结果（识别的文本和得分）会打印到屏幕上，示例如下：

```bash linenums="1"
Predicts of ./doc/imgs_words_en/word_10.png:('pain', 0.9465042352676392)
```

**注意**：

- 训练上述模型采用的图像分辨率是[1，32，100]，需要通过参数`rec_image_shape`设置为您训练时的识别图像形状。
- 在推理时需要设置参数`rec_char_dict_path`指定字典，如果您修改了字典，请修改该参数为您的字典文件。
- 如果您修改了预处理方法，需修改`tools/infer/predict_rec.py`中NRTR的预处理为您的预处理方法。

### 4.2 C++推理部署

由于C++预处理后处理还未支持NRTR，所以暂未支持

### 4.3 Serving服务化部署

暂不支持

### 4.4 更多推理部署

暂不支持

## 5. FAQ

1. `NRTR`论文中使用Beam搜索进行解码字符，但是速度较慢，这里默认未使用Beam搜索，以贪婪搜索进行解码字符。

## 6. 发行公告

1. release/2.6更新NRTR代码结构，新版NRTR可加载旧版(release/2.5及之前)模型参数，使用下面示例代码将旧版模型参数转换为新版模型参数：

    <details>
    <summary>详情</summary>

    ```python linenums="1"
    params = paddle.load('path/' + '.pdparams') # 旧版本参数
    state_dict = model.state_dict() # 新版模型参数
    new_state_dict = {}

    for k1, v1 in state_dict.items():

        k = k1
        if 'encoder' in k and 'self_attn' in k and 'qkv' in k and 'weight' in k:

            k_para = k[:13] + 'layers.' + k[13:]
            q = params[k_para.replace('qkv', 'conv1')].transpose((1, 0, 2, 3))
            k = params[k_para.replace('qkv', 'conv2')].transpose((1, 0, 2, 3))
            v = params[k_para.replace('qkv', 'conv3')].transpose((1, 0, 2, 3))

            new_state_dict[k1] = np.concatenate([q[:, :, 0, 0], k[:, :, 0, 0], v[:, :, 0, 0]], -1)

        elif 'encoder' in k and 'self_attn' in k and 'qkv' in k and 'bias' in k:

            k_para = k[:13] + 'layers.' + k[13:]
            q = params[k_para.replace('qkv', 'conv1')]
            k = params[k_para.replace('qkv', 'conv2')]
            v = params[k_para.replace('qkv', 'conv3')]

            new_state_dict[k1] = np.concatenate([q, k, v], -1)

        elif 'encoder' in k and 'self_attn' in k and 'out_proj' in k:

            k_para = k[:13] + 'layers.' + k[13:]
            new_state_dict[k1] = params[k_para]

        elif 'encoder' in k and 'norm3' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            new_state_dict[k1] = params[k_para.replace('norm3', 'norm2')]

        elif 'encoder' in k and 'norm1' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            new_state_dict[k1] = params[k_para]


        elif 'decoder' in k and 'self_attn' in k and 'qkv' in k and 'weight' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            q = params[k_para.replace('qkv', 'conv1')].transpose((1, 0, 2, 3))
            k = params[k_para.replace('qkv', 'conv2')].transpose((1, 0, 2, 3))
            v = params[k_para.replace('qkv', 'conv3')].transpose((1, 0, 2, 3))
            new_state_dict[k1] = np.concatenate([q[:, :, 0, 0], k[:, :, 0, 0], v[:, :, 0, 0]], -1)

        elif 'decoder' in k and 'self_attn' in k and 'qkv' in k and 'bias' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            q = params[k_para.replace('qkv', 'conv1')]
            k = params[k_para.replace('qkv', 'conv2')]
            v = params[k_para.replace('qkv', 'conv3')]
            new_state_dict[k1] = np.concatenate([q, k, v], -1)

        elif 'decoder' in k and 'self_attn' in k and 'out_proj' in k:

            k_para = k[:13] + 'layers.' + k[13:]
            new_state_dict[k1] = params[k_para]

        elif 'decoder' in k and 'cross_attn' in k and 'q' in k and 'weight' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            k_para = k_para.replace('cross_attn', 'multihead_attn')
            q = params[k_para.replace('q', 'conv1')].transpose((1, 0, 2, 3))
            new_state_dict[k1] = q[:, :, 0, 0]

        elif 'decoder' in k and 'cross_attn' in k and 'q' in k and 'bias' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            k_para = k_para.replace('cross_attn', 'multihead_attn')
            q = params[k_para.replace('q', 'conv1')]
            new_state_dict[k1] = q

        elif 'decoder' in k and 'cross_attn' in k and 'kv' in k and 'weight' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            k_para = k_para.replace('cross_attn', 'multihead_attn')
            k = params[k_para.replace('kv', 'conv2')].transpose((1, 0, 2, 3))
            v = params[k_para.replace('kv', 'conv3')].transpose((1, 0, 2, 3))
            new_state_dict[k1] = np.concatenate([k[:, :, 0, 0], v[:, :, 0, 0]], -1)

        elif 'decoder' in k and 'cross_attn' in k and 'kv' in k and 'bias' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            k_para = k_para.replace('cross_attn', 'multihead_attn')
            k = params[k_para.replace('kv', 'conv2')]
            v = params[k_para.replace('kv', 'conv3')]
            new_state_dict[k1] = np.concatenate([k, v], -1)

        elif 'decoder' in k and 'cross_attn' in k and 'out_proj' in k:

            k_para = k[:13] + 'layers.' + k[13:]
            k_para = k_para.replace('cross_attn', 'multihead_attn')
            new_state_dict[k1] = params[k_para]
        elif 'decoder' in k and 'norm' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            new_state_dict[k1] = params[k_para]
        elif 'mlp' in k and 'weight' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            k_para = k_para.replace('fc', 'conv')
            k_para = k_para.replace('mlp.', '')
            w = params[k_para].transpose((1, 0, 2, 3))
            new_state_dict[k1] = w[:, :, 0, 0]
        elif 'mlp' in k and 'bias' in k:
            k_para = k[:13] + 'layers.' + k[13:]
            k_para = k_para.replace('fc', 'conv')
            k_para = k_para.replace('mlp.', '')
            w = params[k_para]
            new_state_dict[k1] = w

        else:
            new_state_dict[k1] = params[k1]

        if list(new_state_dict[k1].shape) != list(v1.shape):
            print(k1)


    for k, v1 in state_dict.items():
        if k not in new_state_dict.keys():
            print(1, k)
        elif list(new_state_dict[k].shape) != list(v1.shape):
            print(2, k)



    model.set_state_dict(new_state_dict)
    paddle.save(model.state_dict(), 'nrtrnew_from_old_params.pdparams')

    ```

    </details>

2. 新版相比与旧版，代码结构简洁，推理速度有所提高。

## 引用

```bibtex
@article{Sheng2019NRTR,
  title     = {NRTR: A No-Recurrence Sequence-to-Sequence Model For Scene Text Recognition},
  author    = {Fenfen Sheng and Zhineng Chen and Bo Xu},
  booktitle = {ICDAR},
  year      = {2019},
  url       = {http://arxiv.org/abs/1806.00926},
  pages     = {781-786}
}
```
