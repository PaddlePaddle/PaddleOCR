# PaddleOCR 3.x 升级说明

## 1. PaddleOCR 为什么要从 2.x 升级到 3.x？

自 2021 年 2 月发布 2.0 版本以来，PaddleOCR 社区已走过四年多的快速发展期，GitHub star 数量、社区用户和贡献者、issue 与 PR 数量等均有指数级的增长。在多语种识别、版面分析等新需求的推动下，PaddleOCR 在 2.x 系列中不断增添功能，但最初以轻量化为核心的架构已难以应对功能繁荣带来的复杂性与维护成本。

随着代码中模块分支与“桥接”层频繁增加，重复实现、接口不统一的问题愈发突出，测试难度也不断增加，开发效率严重受限；而旧版依赖与最新 PaddlePaddle 更新的不兼容，限制了对飞桨新特性的使用，进一步拖慢了训练与推理速度。这种状况下，继续在现有架构基础上打补丁，只会带来更多技术债与系统脆弱性。

另一方面，基于 Transformer 的视觉语言大模型正在为文档理解、图文摘要、智能校对等高级应用场景注入新动能。社区迫切期待，这类模型能够突破传统 OCR 识别的局限，直接发挥其更强的上下文理解与推理能力。同时，传统的 OCR 小模型亦可与大模型协同工作，既能满足大模型在文档解析等方面的输入需求，又能通过大小模型协同，实现优势互补，进一步提升系统整体性能。

此外，飞桨框架于 2025 年 4 月发布的 3.0 正式版，在训推一体化、国产硬件适配等方面实现了颠覆性升级，这也对 PaddleOCR 在训练和推理层面提出了新的改造需求。

综合以上背景，我们决定对 PaddleOCR 进行一次重大、非兼容性升级——从 2.x 跳至 3.x。新版本将在架构层面实现模块化、插件化设计，在尽可能不改变用户使用习惯的同时，结合大模型，提供更加丰富的功能，并充分利用飞桨 3.0 的新特性，既清理冗余、降低维护成本，又为性能与功能扩展提供更坚实的基础。

## 2. PaddleOCR 2.x 到 3.x 主要升级内容

本次升级内容主要可分为三个部分：

1. **新增多条模型产线**：推出 PP-OCRv5、PP-StructureV3、PP-ChatOCR v4 等多条模型产线，并补充覆盖多种方向的基础模型，重点增强了多文字类型识别、手写体识别等能力，满足大模型应用对复杂文档高精度解析的旺盛需求。用户可直接开箱使用，提升开发效率。
2. **重构部署能力，统一推理接口**：PaddleOCR 3.x 融合了飞桨 [PaddleX](../version3.x/paddleocr_and_paddlex.md) 工具的底层能力，全面升级推理、部署模块，修正 2.x 版本中的设计错误，统一并优化了 Python API 和命令行接口（CLI）。部署能力现覆盖高性能推理、服务化部署及端侧部署三大场景。
3. **适配飞桨 3.0，优化训练流程**：新版本已兼容飞桨 3.0 的 CINN 编译器等最新特性，并对模型命名体系进行了更新，采用更规范、统一的命名规则，为后续迭代与维护奠定基础。

对于 PaddleOCR 2.x 中的部分历史遗留功能，PaddleOCR 3.x 目前仍提供了一定程度的兼容支持。详情请参阅 [历史遗留功能](../version2.x/legacy/index.md)。

## 3. 将 PaddleOCR 2.x 的推理代码移到 PaddleOCR 3.x

对于 OCR 任务，PaddleOCR 3.x 仍然支持与 PaddleOCR 2.x 类似的用法。以 Python API 为例，以下是 PaddleOCR 2.x 的常见使用方式：

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="en")
result = ocr.ocr("img.png")
for res in result:
    for line in res:
        print(line)
        
# 可视化
from PIL import Image
from paddleocr import draw_ocr
result = result[0]
image = Image.open(img_path).convert("RGB")
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path="simfang.ttf")
im_show = Image.fromarray(im_show)
im_show.save("result.jpg")
```

在 PaddleOCR 3.x 中，以上流程得到了进一步简化，示例如下：

```python
from paddleocr import PaddleOCR

# 基础的初始化参数保持一致
ocr = PaddleOCR(lang="en")
result = ocr.ocr("img.png")
# 也可以使用新的统接口
# result = ocr.predict("img.png")
for res in result:
    # 可直接调用方法打印识别结果，无需嵌套循环
    res.print()

# 可视化及结果保存更为简洁
res.save_to_img("result")
```

需要特别指出的是，PaddleOCR 2.x 提供的 `PPStructure` 在 PaddleOCR 3.x 中已被移除。建议使用功能更丰富、解析效果更好的 `PPStructureV3` 替代，并参考相关文档了解新接口的用法。

此外，在 PaddleOCR 2.x 中，可以通过在构造 `PaddleOCR` 对象时传入 `show_log` 参数来控制日志输出。然而，这种设计存在局限：由于所有 `PaddleOCR` 实例共享一个日志器，当一个实例设置了日志行为后，其它实例也会受到影响，这显然不符合预期。为了解决这一问题，PaddleOCR 3.x 引入了全新的日志系统。详细内容请参阅 [日志](../version3.x/logging.md)。

## 4. PaddleOCR 3.0 已知问题

PaddleOCR 3.x 仍在持续迭代与优化中，目前已知存在以下尚待完善之处：

1. 对 C++ 本地部署的支持尚不完整。
2. 暂未提供性能与 PaddleOCR 2.x 中 PaddleServing 部署方案对齐的高性能服务化部署方案。
3. 端侧部署目前仅支持部分重点模型，其余模型尚未开放支持。

如果你在使用过程中遇到问题，欢迎随时在 issue 区提交反馈。我们也诚挚邀请更多社区用户参与到 PaddleOCR 的建设中来，感谢大家一直以来的关注与支持！
