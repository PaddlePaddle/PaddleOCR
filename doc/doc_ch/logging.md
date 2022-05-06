## 记录实验输出指标和模型

PaddleOCR 支持两种实验输出指标记录工具，直接集成于训练API。它们分别是：[VisualDL](https://readthedocs.org/projects/visualdl/) and [Weights & Biases](https://docs.wandb.ai/)。

### VisualDL

VisualDL 是基于PaddlePaddle的可视化分析工具。使用此工具，所有训练输出指标都可记录并展示于VisualDL面板之中。在配置文件的`Global`部分添加如下行，即可启用该功能：

```
Global:
    use_visualdl: True
```

在终端运行如下命令，即可进行数据可视化：

```shell
visualdl --logdir <save_model_dir>
```

在浏览器打开`http://localhost:8040`，即可查看。

### Weights & Biases

W&B是机器学习运维工具，可以用于实验追踪，记录运行中的超参数和输出指标(metric)，然后对数据、模型以及结果进行可视化和比较，并快速与同事分享你的发现。PaddleOCR也直接集成了此工具，只需安装`wandb` sdk，并登陆wandb账号，即可启用该工具。

```shell
pip install wandb
wandb login
```

如果还没有wandb账号，可以点击[此处](https://wandb.ai/site)免费注册。

在配置文件的`Global`部分添加如下行，即可使用W&B进行数据、模型的追踪与可视化：

```
Global:
    use_wandb: True
```

若要添加更多相关参数（具体参数说明参见[配置文件内容与生成](./config.md)）到 `WandbLogger` 中，可以在配置文件中新增`wandb`部分，如下所示：

```
wandb:
    project: my_project
    entity: my_team
```

配置文件中的这些参数如，项目名称`project`、实体名称`entity`（默认为登录用户）、存储元数据的目录`save_dir`（默认为 `./wandb`）等用于实例化 `WandbLogger` 对象。 在训练过程中，调用 `log_metrics` 函数仅从`rank=0`进程中分别记录训练和评估步骤中的训练和评估指标。

在每个模型保存步骤中，WandbLogger 使用`log_model`函数记录模型以及相关元数据和标签，显示模型保存的epoch、模型是否最佳等。

上面提到的所有日志记录都集成到 `program.train` 函数中，并将生成如下类似的面板：

![W&B Dashboard](../imgs_en/wandb_metrics.png)

![W&B Models](../imgs_en/wandb_models.png)

对于记录图像、音频、视频或任何其他形式的数据的更高级用法，可以使用 `WandbLogger().run.log`。 [此处](https://docs.wandb.ai/examples) 提供了有关如何记录不同类型数据的更多示例。

数据展示面板的链接会在每个训练任务的开始和结束时打印到终端控制台，也可以通过浏览器登录 W&B 帐户来访问它。

### 使用多个日志记录器

只需将上述两个标志都设置为 True，VisualDL 和 W&B 也可以同时使用。
