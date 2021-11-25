# 社区贡献说明

感谢大家长久以来对PaddleOCR的支持和关注，与广大开发者共同构建一个专业、和谐、相互帮助的开源社区是PaddleOCR的目标。本文档展示了已有的社区贡献、对于各类贡献说明、新的机会与流程，希望贡献流程更加高效、路径更加清晰。

PaddleOCR希望可以通过AI的力量助力任何一位有梦想的开发者实现自己的想法，享受创造价值带来的愉悦。

<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR" />
</a>

> 上图为PaddleOCR目前的Contributor，定期更新

## 1. 社区贡献

### 1.1 为PaddleOCR新增功能

- 非常感谢 [authorfu](https://github.com/authorfu) 贡献Android([#340](https://github.com/PaddlePaddle/PaddleOCR/pull/340))和[xiadeye](https://github.com/xiadeye) 贡献IOS的demo代码([#325](https://github.com/PaddlePaddle/PaddleOCR/pull/325))
- 非常感谢 [tangmq](https://gitee.com/tangmq) 给PaddleOCR增加Docker化部署服务，支持快速发布可调用的Restful API服务([#507](https://github.com/PaddlePaddle/PaddleOCR/pull/507))。
- 非常感谢 [lijinhan](https://github.com/lijinhan) 给PaddleOCR增加java SpringBoot 调用OCR Hubserving接口完成对OCR服务化部署的使用([#1027](https://github.com/PaddlePaddle/PaddleOCR/pull/1027))。
- 非常感谢 [Evezerest](https://github.com/Evezerest)， [ninetailskim](https://github.com/ninetailskim)， [edencfc](https://github.com/edencfc)， [BeyondYourself](https://github.com/BeyondYourself)， [1084667371](https://github.com/1084667371) 贡献了[PPOCRLabel](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/PPOCRLabel/README_ch.md) 的完整代码。

### 1.2 基于PaddleOCR的社区贡献

- 【最新】完整的C#版本标注工具 [FastOCRLabel](https://gitee.com/BaoJianQiang/FastOCRLabel) (@ [包建强](https://gitee.com/BaoJianQiang) )
- 通用型桌面级即时翻译工具 [DangoOCR离线版](https://github.com/PantsuDango/DangoOCR) (@ [PantsuDango](https://github.com/PantsuDango))
- 获取OCR识别结果的key-value [paddleOCRCorrectOutputs](https://github.com/yuranusduke/paddleOCRCorrectOutputs) (@ [yuranusduke](https://github.com/yuranusduke))
- 截屏转文字工具  [scr2txt](https://github.com/lstwzd/scr2txt) (@ [lstwzd](https://github.com/lstwzd))
- 身份证复印件识别 [id_card_ocr](https://github.com/baseli/id_card_ocr)(@ [baseli](https://github.com/baseli))
- 能看懂表格图片的数据助手：[Paddle_Table_Image_Reader](https://github.com/thunder95/Paddle_Table_Image_Reader) (@ [thunder95][https://github.com/thunder95])
- 英文视频自动生成字幕 [AI Studio项目](https://aistudio.baidu.com/aistudio/projectdetail/1054614?channelType=0&channel=0)( @ [叶月水狐](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/322052))

### 1.3 代码与文档优化


- 非常感谢 [zhangxin](https://github.com/ZhangXinNan)([Blog](https://blog.csdn.net/sdlypyzq)) 贡献新的可视化方式、添加.gitgnore、处理手动设置PYTHONPATH环境变量的问题([#210](https://github.com/PaddlePaddle/PaddleOCR/pull/210))。
- 非常感谢 [lyl120117](https://github.com/lyl120117) 贡献打印网络结构的代码([#304](https://github.com/PaddlePaddle/PaddleOCR/pull/304))。
- 非常感谢 [BeyondYourself](https://github.com/BeyondYourself) 给PaddleOCR提了很多非常棒的建议，并简化了PaddleOCR的部分代码风格([so many commits)](https://github.com/PaddlePaddle/PaddleOCR/commits?author=BeyondYourself)。

- 非常感谢 [Khanh Tran](https://github.com/xxxpsyduck) 和 [Karl Horky](https://github.com/karlhorky) 贡献修改英文文档。

### 1.4 多语言语料

- 非常感谢 [xiangyubo](https://github.com/xiangyubo) 贡献手写中文OCR数据集([#321](https://github.com/PaddlePaddle/PaddleOCR/pull/321))。
- 非常感谢 [Mejans](https://github.com/Mejans) 给PaddleOCR增加新语言奥克西坦语Occitan的字典和语料([#954](https://github.com/PaddlePaddle/PaddleOCR/pull/954))。

## 2. 贡献说明
### 2.1 新增功能类

PaddleOCR非常欢迎社区贡献以PaddleOCR为核心的各种服务、部署实例与软件应用，经过认证的社区贡献会被添加在上述社区贡献表中，为广大开发者增加曝光，也是PaddleOCR的荣耀，其中：

- 项目形式：官方社区认证的项目代码应有良好的规范和结构，同时，还应配备一个详细的README.md，说明项目的使用方法。通过在requirements.txt文件中增加一行 `paddleocr` 可以自动收录到PaddleOCR的usedby中。

- 合入方式：如果是对PaddleOCR现有工具的更新升级，则会合入主repo。如果为PaddleOCR拓展了新功能，请先与官方人员联系，确认项目是否合入主repo，*即使新功能未合入主repo，我们同样也会以社区贡献的方式为您的个人项目增加曝光。*


### 2.2 代码优化

如果您在使用PaddleOCR时遇到了代码bug、功能不符合预期等问题，可以为PaddleOCR贡献您的修改，其中：

- Python代码规范可参考[附录1：Python代码规范](./code_and_doc.md/#附录1)。

-  提交代码前请再三确认不会引入新的bug，并在PR中描述优化点。如果该PR解决了某个issue，请在PR中连接到该issue。所有的PR都应该遵守附录3中的[3.2.10 提交代码的一些约定。](./code_and_doc.md/#提交代码的一些约定)

- 请在提交之前参考下方的[附录3：Pull Request说明](./code_and_doc.md/#附录3)。如果您对git的提交流程不熟悉，同样可以参考附录3的3.2节。

**最后请在PR的题目中加上标签`【third-party】` , 在说明中@Evezerest，拥有此标签的PR将会被高优处理**。

### 2.3 文档优化

如果您在使用PaddleOCR时遇到了文档表述不清楚、描述缺失、链接失效等问题，可以为PaddleOCR贡献您的修改。文档书写规范请参考[附录2：文档规范](./code_and_doc.md/#附录2)。**最后请在PR的题目中加上标签`【third-party】` , 在说明中@Evezerest，拥有此标签的PR将会被高优处理。**

## 3. 更多贡献机会

我们非常鼓励开发者使用PaddleOCR实现自己的想法，同时我们也列出一些经过分析后认为有价值的拓展方向，供大家参考

- 功能类：IOS端侧demo、前后处理工具、针对各种垂类场景的检测识别模型（如手写体、公式）。
- 文档类：PaddleOCR在各种垂类行业的应用案例（可在公众号中推广）。

## 4. 联系我们

PaddleOCR非常欢迎广大开发者在有意向贡献前与我们联系，这样可以大大降低PR过程中的沟通成本。同时，如果您觉得某些想法个人难以实现，我们也可以通过SIG的形式定向为项目招募志同道合的开发者一起共建。通过SIG渠道贡献的项目将会获得深层次的研发支持与运营资源。

我们推荐的贡献流程是：

- 通过在github issue的题目中增加  `【third-party】` 标记，说明遇到的问题（以及解决的思路）或想拓展的功能，等待值班人员回复。例如 `【third-party】为PaddleOCR贡献IOS示例`
- 与我们沟通确认技术方案或bug、优化点准确无误后进行功能新增或相应的修改，代码与文档遵循相关规范。
- PR链接到上述issue，等待review。

## 5. 致谢与后续

  - 合入代码之后，首页README末尾新增感谢贡献，默认链接为github名字及主页，如果有需要更换主页，也可以联系我们。
  - 新增重要功能类，会在用户群广而告之，享受开源社区荣誉时刻。
  - **如果您有基于PaddleOCR的贡献，但未出现在上述列表中，请按照 `4. 联系我们` 的步骤与我们联系。**
