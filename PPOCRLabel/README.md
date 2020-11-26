# PPOCRLabel

PPOCRLabel是一款适用于OCR领域的半自动化图形标注工具，使用python3和pyqt5编写，支持矩形框标注和四点标注模式，导出格式可直接用于PPOCR检测和识别模型的训练。

<img src="./data/gif/steps.gif" width="100%"/>

## 安装

### 1. 安装PaddleOCR
参考[PaddleOCR安装文档](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/installation.md)准备好PaddleOCR

### 2. 安装PPOCRLabel
#### Windows + Anaconda

下载安装[Anaconda](https://www.anaconda.com/download/#download) (Python 3+)

```
conda install pyqt=5
cd ./PPOCRLabel # 将目录切换到PPOCRLabel文件夹下
pyrcc5 -o libs/resources.py resources.qrc
python PPOCRLabel.py
```

#### Ubuntu Linux

```
sudo apt-get install pyqt5-dev-tools
sudo apt-get install trash-cli
cd ./PPOCRLabel # 将目录切换到PPOCRLabel文件夹下
sudo pip3 install -r requirements/requirements-linux-python3.txt
make qt5py3
python3 PPOCRLabel.py
```

#### macOS
```
pip3 install pyqt5
pip3 uninstall opencv-python # 由于mac版本的opencv与pyqt有冲突，需先手动卸载opencv
pip3 install opencv-contrib-python-headless # 安装headless版本的open-cv
cd ./PPOCRLabel # 将目录切换到PPOCRLabel文件夹下
make qt5py3
python3 PPOCRLabel.py
```

## 使用

### 操作步骤

1. 安装与运行：使用上述命令安装与运行程序。
2. 打开文件夹：在菜单栏点击 “文件” - "打开目录" 选择待标记图片的文件夹<sup>[1]</sup>.
3. 自动标注：点击 ”自动标注“，使用PPOCR超轻量模型对图片文件名前图片状态<sup>[2]</sup>为 “X” 的图片进行自动标注。
4. 手动标注：点击 “矩形标注”（推荐直接在英文模式下点击键盘中的 “W”)，用户可对当前图片中模型未检出的部分进行手动绘制标记框。点击键盘P，则使用四点标注模式（或点击“编辑” - “四点标注”），用户依次点击4个点后，双击左键表示标注完成。
5. 标记框绘制完成后，用户点击 “确认”，检测框会先被预分配一个 “待识别” 标签。
6. 重新识别：将图片中的所有检测画绘制/调整完成后，点击 “重新识别”，PPOCR模型会对当前图片中的**所有检测框**重新识别<sup>[3]</sup>。
7. 内容更改：双击识别结果，对不准确的识别结果进行手动更改。
8. 保存：点击 “保存”，图片状态切换为 “√”，跳转至下一张。
9. 删除：点击 “删除图像”，图片将会被删除至回收站。
10. 标注结果：关闭应用程序或切换文件路径后，手动保存过的标签将会被存放在所打开图片文件夹下的*Label.txt*中。在菜单栏点击 “PaddleOCR” - "保存识别结果"后，会将此类图片的识别训练数据保存在*crop_img*文件夹下，识别标签保存在*rec_gt.txt*中<sup>[4]</sup>。

### 注意

[1] PPOCRLabel以文件夹为基本标记单位，打开待标记的图片文件夹后，不会在窗口栏中显示图片，而是在点击 "选择文件夹" 之后直接将文件夹下的图片导入到程序中。

[2] 图片状态表示本张图片用户是否手动保存过，未手动保存过即为 “X”，手动保存过为 “√”。点击 “自动标注”按钮后，PPOCRLabel不会对状态为 “√” 的图片重新标注。

[3] 点击“重新识别”后，模型会对图片中的识别结果进行覆盖。因此如果在此之前手动更改过识别结果，有可能在重新识别后产生变动。

[4] PPOCRLabel产生的文件包括一下几种，请勿手动更改其中内容，否则会引起程序出现异常。

|    文件名     |                             说明                             |
| :-----------: | :----------------------------------------------------------: |
|   Label.txt   | 检测标签，可直接用于PPOCR检测模型训练。用户每保存10张检测结果后，程序会进行自动写入。当用户关闭应用程序或切换文件路径后同样会进行写入。 |
| fileState.txt | 图片状态标记文件，保存当前文件夹下已经被用户手动确认过的图片名称。 |
|  Cache.cach   |              缓存文件，保存模型自动识别的结果。              |
|  rec_gt.txt   | 识别标签。可直接用于PPOCR识别模型训练。需用户手动点击菜单栏“PaddleOCR” - "保存识别结果"后产生。 |
|   crop_img    |   识别数据。按照检测框切割后的图片。与rec_gt.txt同时产生。   |

### 参考资料

1.[Tzutalin. LabelImg. Git code (2015)](https://github.com/tzutalin/labelImg)
