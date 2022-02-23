# 文本生成高性能加速

## 子目录结构

以下是 FasterTransformer 的高性能加速自定义 op 相关文件的目录结构。

```text
.
├── faster_transformer/       # 基于自定义 op FasterTransformer 子路径
  ├── sample/                 # 基于 FasterTransformer 使用样例
  ├── src/                    # 自定义 OP C++ CUDA 代码
  └── transformer/            # Python API 封装脚本
└── patches                   # 自定义 op 第三方库自定义补丁代码
```

## 使用环境说明

* 本项目依赖于 PaddlePaddle 2.1.0 及以上版本或适当的 develop 版本
* CMake >= 3.10
* CUDA 10.1 或 10.2（需要 PaddlePaddle 框架一致）
* gcc 版本需要与编译 PaddlePaddle 版本一致，比如使用 gcc8.2
* 推荐使用 Python3
* [FasterTransformer](https://github.com/NVIDIA/FasterTransformer/tree/v3.1#setup) 使用必要的环境
* 环境依赖
  - attrdict
  - pyyaml
  ```shell
  pip install attrdict pyyaml
  ```

## 快速开始

我们实现了基于 FasterTransformer 的自定义 op 的接入，用于加速文本生成模型在 GPU 上的预测性能。接下来，我们将分别介绍基于 Python 动态图和预测库使用 FasterTransformer 自定义 op 的方式，包括 op 的编译与使用。

## Python 动态图使用自定义 op

### JIT 自动编译

目前当基于动态图使用 FasterTransformer 预测加速自定义 op 时，PaddleNLP 提供了 Just In Time 的自动编译，在一些 API 上，用户无需关注编译流程，可以直接执行对应的 API，程序会自动编译需要的第三方库。

以 Transformer 为例，可以直接调用 `TransformerGenerator()` 这个 API，程序会自动编译。

目前支持 JIT 的预测加速 API 有：
* `FasterTransformer()/TransformerGenerator()`: 支持 Transformer 模型的预测加速功能。使用示例可以参考 [Transformer 预测加速使用示例-sample](./faster_transformer/sample/decoding_sample.py)，[Transformer 预测加速使用示例-机器翻译](../../examples/machine_translation/transformer/faster_transformer/)。
* `FasterGPT()`: 支持 GPT 模型的预测加速功能。使用示例可以参考 [GPT 预测加速使用示例](../../examples/language_model/gpt/faster_gpt/)。
* `FasterUnifiedTransformer()`: 支持 UnifiedTransformer 模型的预测加速功能。使用示例可以参考 [UnifiedTransformer 预测加速使用示例](../../examples/dialogue/unified_transformer/)。
* `FasterUNIMOText()`: 支持 UNIMOText 模型预测加速功能。使用示例可以参考 [UNIMOText 预测加速使用示例](../../examples/text_generation/unimo-text/faster_unimo/)。
* `FasterBART()`: 支持 BART 模型预测加速功能。使用示例可以参考 [BART 预测加速使用示例-sample](./faster_transformer/sample/bart_decoding_sample.py)，[BART 预测加速使用示例-文本摘要](../../examples/text_summarization/bart)。

具体使用方法可以参考 API 文档或是使用示例。

### 编译自定义OP

除了自动编译外，如果需要自行编译，我们已经提供对应的 CMakeLists.txt，可以参考使用如下的方式完成编译。

#### PaddleNLP 准备

首先，如果需要从源码自行编译，可以直接使用 Python 的 package 下的 paddlenlp，或是可从 github 克隆一个 PaddleNLP，并重新编译:

以下以从 github 上 clone 一个新版 PaddleNLP 为例:

``` sh
git clone https://github.com/PaddlePaddle/PaddleNLP.git
```

其次，配置环境变量，让我们可以使用当前 clone 的 paddlenlp，并进入到自定义 OP 的路径，准备后续的编译操作：

``` sh
export PYTHONPATH=$PWD/PaddleNLP/:$PYTHONPATH
cd PaddleNLP/paddlenlp/ops/
```

#### 编译

编译之前，请确保安装的 PaddlePaddle 的版本高于 2.1.0 或是基于最新的 develop 分支的代码编译，并且正常可用。

编译自定义 OP 可以参照一下步骤：

``` sh
mkdir build
cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release -DPY_CMD=python3.x
make -j
cd ../
```

可以使用的编译选项包括：
* `-DPY_CMD`: 指定当前装有 PaddlePaddle 版本的 python 环境，比如 `-DPY_CMD=python3.7`。若未指定 `-DPY_CMD` 将会默认使用系统命令 `python` 对应的 Python。
* `-DSM`: 是指的所用 GPU 的 compute capability，建议不使用该选项设置，未设置时将自动检测。如要设置，需根据 [compute capability](https://developer.nvidia.com/zh-cn/cuda-gpus#compute) 进行设置，如 V100 时设置 `-DSM=70` 或 T4 时设置 `-DSM=75`。
* `-DWITH_GPT`: 是否编译带有 GPT 相关的 lib。若使用 GPT-2 高性能推理，需要加上 `-DWITH_GPT=ON`。默认为 OFF。
* `-DWITH_UNIFIED`: 是否编译带有 Unified Transformer 或是 UNIMOText 相关的 lib。若使用，需要加上 `-DWITH_UNIFIED=ON`。默认为 ON。
* `-DWITH_BART`: 是否编译带有 BART 支持的相关 lib。若使用，需要加上 `-DWITH_BART=ON`。默认为 ON。
* `-DWITH_DECODER`: 是否编译带有 decoder 优化的 lib。默认为 ON。

最终，编译会在 `./build/lib/` 路径下，产出 `libdecoding_op.so`，即需要的 FasterTransformer decoding 执行的库。

### 使用 Transformer decoding 高性能推理

编写 python 脚本的时候，调用 `FasterTransformer` API 即可实现 Transformer 模型的高性能预测。

若当前环境下没有需要的自定义 op 的动态库，将会使用 JIT 自动编译需要的动态库。如果需要自行编译自定义 op 所需的动态库，可以参考前文所述完成编译。编译好后，使用 `FasterTransformer(decoding_lib="/path/to/lib", ...)` 可以完成导入。

举例如下：

``` python
from paddlenlp.ops import FasterTransformer

transformer = FasterTransformer(
    src_vocab_size=args.src_vocab_size,
    trg_vocab_size=args.trg_vocab_size,
    max_length=args.max_length + 1,
    n_layer=args.n_layer,
    n_head=args.n_head,
    d_model=args.d_model,
    d_inner_hid=args.d_inner_hid,
    dropout=args.dropout,
    weight_sharing=args.weight_sharing,
    bos_id=args.bos_idx,
    eos_id=args.eos_idx,
    decoding_strategy=args.decoding_strategy,
    beam_size=args.beam_size,
    topk=args.topk,
    topp=args.topp,
    max_out_len=args.max_out_len,
    decoding_lib=args.decoding_lib,
    use_fp16_decoding=args.use_fp16_decoding)
```

更详细的例子可以参考 `./faster_transformer/sample/decoding_sample.py` 以及 `./sample/encoder_decoding_sample.py`，我们提供了更详细用例。

#### Transformer decoding 示例代码

使用 PaddlePaddle 仅执行 decoding 测试（float32）：

``` sh
export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memory_to_use=0.1
# 执行 decoding_gemm 目的是基于当前环境、配置，提前确定一个性能最佳的矩阵乘算法，不是必要的步骤
./build/third-party/build/fastertransformer/bin/decoding_gemm 32 4 8 64 30000 32 512 0
python ./faster_transformer/sample/decoding_sample.py --config ./faster_transformer/sample/config/decoding.sample.yaml --decoding_lib ./build/lib/libdecoding_op.so
```

使用 PaddlePaddle 仅执行 decoding 测试（float16）：
执行 float16 的 decoding，需要在执行的时候，加上 `--use_fp16_decoding` 选项。

``` sh
export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memory_to_use=0.1
# 执行 decoding_gemm 目的是基于当前环境、配置，提前确定一个性能最佳的矩阵乘算法，不是必要的步骤
./build/third-party/build/fastertransformer/bin/decoding_gemm 32 4 8 64 30000 32 512 1
python ./faster_transformer/sample/decoding_sample.py --config ./faster_transformer/sample/config/decoding.sample.yaml --decoding_lib ./build/lib/libdecoding_op.so --use_fp16_decoding
```

其中，`decoding_gemm` 不同参数的意义可以参考 [FasterTransformer 文档](https://github.com/NVIDIA/FasterTransformer/tree/v3.1#execute-the-decoderdecoding-demos)。这里提前执行 `decoding_gemm`，可以在当前路径下生成一个 config 文件，里面会包含针对当前 decoding 部分提供的配置下，性能最佳的矩阵乘的算法，并在执行的时候读入这个数据。

### 使用 GPT-2 decoding 高性能推理

与 `FasterTransformer` 类似，可以通过一下方式调用 GPT-2 相关优化：

``` python
from paddlenlp.ops import FasterGPT
from paddlenlp.transformers import GPTModel, GPTForPretraining

MODEL_CLASSES = {
    "gpt2-medium-en": (GPTForPretraining, GPTTokenizer),
}

model_class, tokenizer_class = MODEL_CLASSES[args.model_name]
tokenizer = tokenizer_class.from_pretrained(args.model_name)
model = model_class.from_pretrained(args.model_name)

# Define model
gpt = FasterGPT(
    model=model,
    decoding_lib=args.decoding_lib,
    use_fp16_decoding=args.use_fp16_decoding)
```

目前，GPT-2 的高性能预测接口 `FasterGPT()` 要求 batch 内输入的样本的长度都是相同的。并且，仅支持 topk-sampling 和 topp-sampling，不支持 beam-search。

若当前环境下没有需要的自定义 op 的动态库，将会使用 JIT 自动编译需要的动态库。如果需要自行编译自定义 op 所需的动态库，可以参考前文所述完成编译。编译好后，使用 `FasterGPT(decoding_lib="/path/to/lib", ...)` 可以完成导入。

更详细的例子可以参考 `./faster_transformer/sample/gpt_sample.py`，我们提供了更详细用例。

#### GPT-2 decoding 示例代码

使用 PaddlePaddle 仅执行 decoding 测试（float32）：

``` sh
export CUDA_VISIBLE_DEVICES=0
python ./faster_transformer/sample/gpt_sample.py --model_name_or_path gpt2-medium-en --batch_size 1 --topk 4 --topp 0.0 --max_length 32 --start_token "<|endoftext|>" --end_token "<|endoftext|>" --temperature 1.0
```

其中，各个选项的意义如下：
* `--model_name_or_path`: 预训练模型的名称或是路径。
* `--decoding_lib`: 指向 `libdecoding_op.so` 的路径。需要包含 `libdecoding_op.so`。若不指定或是不存在则将自动进行 jit 编译产出该 lib。
* `--batch_size`: 一个 batch 内，样本数目的大小。
* `--topk`: 执行 topk-sampling 的时候的 `k` 的大小，默认是 4。
* `--topp`: 执行 topp-sampling 的时候的阈值的大小，默认是 0.0 表示不执行 topp-sampling。
* `--max_length`: 最长的生成长度。
* `--start_token`: 字符串，表示任意生成的时候的开始 token。
* `--end_token`: 字符串，生成的结束 token。
* `--temperature`: temperature 的设定。
* `--use_fp16_decoding`: 是否使用 fp16 进行推理。

若当前环境下没有需要的自定义 op 的动态库，将会使用 JIT 自动编译需要的动态库。如果需要自行编译自定义 op 所需的动态库，可以参考前文。编译好后，可以在执行 `gpt_sample.py` 时使用 `--decoding_lib ../../../../paddlenlp/ops/build/lib/libdecoding_op.so` 可以完成导入。

## C++ 预测库使用自定义 op

### 编译自定义OP

在 C++ 预测库使用自定义 OP 需要将实现的 C++、CUDA 代码**以及 C++ 预测的 demo**编译成一个可执行文件。因预测库支持方式与 Python 不同，这个过程将不会产生自定义 op 的动态库，将直接得到可执行文件。我们已经提供对应的 CMakeLists.txt ，可以参考使用如下的方式完成编译。并获取执行 demo。

#### PaddleNLP 准备

首先，因为需要基于当前环境重新编译，当前的 paddlenlp 的 python 包里面并不包含 FasterTransformer 相关 lib，需要从源码自行编译，可以直接使用 Python 的 package 下的 paddlenlp，或是可从 github 克隆一个 PaddleNLP，并重新编译:

以下以从 github 上 clone 一个新版 PaddleNLP 为例:

``` sh
git clone https://github.com/PaddlePaddle/PaddleNLP.git
```

其次，让我们可以使用当前 clone 的 paddlenlp，并进入到自定义 OP 的路径，准备后续的编译操作：

``` sh
cd PaddleNLP/paddlenlp/ops/
```

#### 编译

编译之前，请确保安装的 PaddlePaddle 的版本高于 2.1.0 或是基于最新的 develop 分支的代码编译，并且正常可用。

编译自定义 OP 可以参照一下步骤：

``` sh
mkdir build
cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release -DPADDLE_LIB=/path/to/paddle_inference_lib/ -DDEMO=./demo/transformer_e2e.cc -DON_INFER=ON -DWITH_MKL=ON
make -j
cd ../
```

可以使用的编译选项包括：
* `-DPADDLE_LIB`: 需要指明使用的 PaddlePaddle 预测库的路径 `/path/to/paddle_inference_install_dir/`，需要使用的 PaddlePaddle 的 lib 可以选择自行编译或者直接从官网下载 [paddle_inference_linux_lib](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#linux)。需要注意的是，在该路径下，预测库的组织结构满足：
  ```text
  .
  ├── CMakeCache.txt
  ├── paddle/
    ├── include/
    └── lib/
  ├── third_party/
    ├── cudaerror/
    ├── install/
    └── threadpool/
  └── version.txt
  ```
* `-DDEMO`: 说明预测库使用 demo 的位置。比如指定 -DDEMO=./demo/transformer_e2e.cc 或是 -DDEMO=./demo/gpt.cc。最好使用绝对路径，若使用相对路径，需要是相对于 `PaddleNLP/paddlenlp/ops/faster_transformer/src/` 的相对路径。
* `-DSM`: 是指的所用 GPU 的 compute capability，建议不使用该选项设置，未设置时将自动检测。如要设置，需根据 [compute capability](https://developer.nvidia.com/zh-cn/cuda-gpus#compute) 进行设置，如 V100 时设置 `-DSM=70` 或 T4 时设置 `-DSM=75`。
* `-DWITH_GPT`: 是否编译带有 GPT 相关的 lib。若使用 GPT-2 高性能推理，需要加上 `-DWITH_GPT=ON`。默认为 OFF。
* `-DWITH_UNIFIED`: 是否编译带有 Unified Transformer 或是 UNIMOText 相关的 lib。若使用，需要加上 `-DWITH_UNIFIED=ON`。默认为 ON。
* `-DWITH_BART`: 是否编译带有 BART 支持的相关 lib。若使用，需要加上 `-DWITH_BART=ON`。默认为 ON。
* `-DWITH_DECODER`: 是否编译带有 decoder 优化的 lib。默认为 ON。
* `-DWITH_MKL`: 若当前是使用的 mkl 的 Paddle lib，那么需要打开 MKL 以引入 MKL 相关的依赖。
* `-DON_INFER`: 是否编译 paddle inference 预测库。
* **当使用预测库的自定义 op 的时候，请务必开启 `-DON_INFER=ON` 选项，否则，不会得到预测库的可执行文件。**

#### 执行 Transformer decoding on PaddlePaddle

编译完成后，在 `build/bin/` 路径下将会看到 `transformer_e2e` 的一个可执行文件。通过设置对应的设置参数完成执行的过程。

``` sh
cd bin/
./transformer_e2e -batch_size <batch_size> -gpu_id <gpu_id> -model_dir <model_directory> -vocab_file <dict_file> -data_file <input_data>
```

举例说明：

``` sh
cd bin/
# 执行 decoding_gemm 目的是基于当前环境、配置，提前确定一个性能最佳的矩阵乘算法，不是必要的步骤
../third-party/build/fastertransformer/bin/decoding_gemm 8 5 8 64 38512 256 512 0
./transformer_e2e -batch_size 8 -gpu_id 0 -model_dir ./infer_model/ -vocab_file DATA_HOME/WMT14ende/WMT14.en-de/wmt14_ende_data_bpe/vocab_all.bpe.33708 -data_file DATA_HOME/WMT14ende/WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.en
```

其中：
* `decoding_gemm` 不同参数的意义可以参考 [FasterTransformer 文档](https://github.com/NVIDIA/FasterTransformer/tree/v3.1#execute-the-decoderdecoding-demos)。这里提前执行 `decoding_gemm`，可以在当前路径下生成一个 config 文件，里面会包含针对当前 decoding 部分提供的配置下，性能最佳的矩阵乘的算法，并在执行的时候读入这个数据。
* `DATA_HOME` 则是 `paddlenlp.utils.env.DATA_HOME` 返回的路径。

预测所需要的模型文件，可以通过 `PaddleNLP/examples/machine_translation/transformer/faster_transformer/README.md` 文档中所记述的方式导出。

#### 执行 GPT decoding on PaddlePaddle

如果需要使用 Paddle Inference 预测库针对 GPT 进行预测，首先，需要导出预测模型，可以通过 `./faster_transformer/sample/gpt_export_model_sample.py` 脚本获取预测库用模型，执行方式如下所示：

``` sh
python ./faster_transformer/sample/gpt_export_model_sample.py --model_name_or_path gpt2-medium-en  --topk 4 --topp 0.0 --max_out_len 32 --start_token "<|endoftext|>" --end_token "<|endoftext|>" --temperature 1.0 --inference_model_dir ./infer_model/
```

各个选项的意义与上文的 `gpt_sample.py` 的选项相同。额外新增一个 `--inference_model_dir` 选项用于指定保存的模型文件、词表等文件。

若当前环境下没有需要的自定义 op 的动态库，将会使用 JIT 自动编译需要的动态库。如果需要自行编译自定义 op 所需的动态库，可以参考前文。编译好后，可以在执行 `gpt_export_model_sample.py` 时使用 `--decoding_lib ../../../../paddlenlp/ops/build/lib/libdecoding_op.so` 可以完成导入。

注意：如果是自行编译的话，这里的 `libdecoding_op.so` 的动态库是参照前文中 **`Python 动态图使用自定义 op`** 编译出来的 lib，与 **`C++ 预测库使用自定义 op`** 编译产出不同。因此，在使用预测库前，还需要额外导出模型：
  * 一次用于获取 Python 动态图下的 lib，用到 Python 端进行模型导出。
  * 一次获取编译的基于预测库的可执行文件

若是使用的模型是 gpt2-medium-en，保存之后，`./infer_model/` 目录下组织的结构如下：

``` text
.
├── gpt.pdiparams       # 保存的参数文件
├── gpt.pdiparams.info  # 保存的一些变量描述信息，预测不会用到
├── gpt.pdmodel         # 保存的模型文件
├── merges.txt          # bpe
└── vocab.txt           # 词表
```

同理，完成编译后，可以在 `build/bin/` 路径下将会看到 `gpt` 的一个可执行文件。通过设置对应的设置参数完成执行的过程。

``` sh
cd bin/
./gpt -batch_size 1 -gpu_id 0 -model_dir path/to/model -vocab_file path/to/vocab -start_token "<|endoftext|>" -end_token "<|endoftext|>"
```
