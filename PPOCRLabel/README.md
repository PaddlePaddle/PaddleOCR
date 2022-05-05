English | [简体中文](README_ch.md)

# PPOCRLabel

PPOCRLabel is a semi-automatic graphic annotation tool suitable for OCR field, with built-in PPOCR model to automatically detect and re-recognize data. It is written in python3 and pyqt5, supporting rectangular box annotation and four-point annotation modes. Annotations can be directly used for the training of PPOCR detection and recognition models.

<img src="./data/gif/steps_en.gif" width="100%"/>

### Recent Update

- 2022.02：（by [PeterH0323](https://github.com/peterh0323) ）
  - Add KIE Mode by using `--kie`, for [detection + identification + keyword extraction] labeling.
- 2022.01：（by [PeterH0323](https://github.com/peterh0323) ）
  - Improve user experience: prompt for the number of files and labels, optimize interaction, and fix bugs such as only use CPU when inference
  - New functions: Support using `C` or `X` to rotate box.
- 2021.11.17:
  - Support install and start PPOCRLabel through the whl package (by [d2623587501](https://github.com/d2623587501))
  - Dataset segmentation: Divide the annotation file into training, verification and testing parts (refer to section 3.5 below, by [MrCuiHao](https://github.com/MrCuiHao))
- 2021.8.11:
  - New functions: Open the dataset folder, image rotation (Note: Please delete the label box before rotating the image) (by [Wei-JL](https://github.com/Wei-JL))
  - Added shortcut key description (Help-Shortcut Key), repaired the direction shortcut key movement function under batch processing (by [d2623587501](https://github.com/d2623587501))
- 2021.2.5: New batch processing and undo functions (by [Evezerest](https://github.com/Evezerest)):
  - **Batch processing function**: Press and hold the Ctrl key to select the box, you can move, copy, and delete in batches.
  - **Undo function**: In the process of drawing a four-point label box or after editing the box, press Ctrl+Z to undo the previous operation.
  - Fix image rotation and size problems, optimize the process of editing the mark frame (by [ninetailskim](https://github.com/ninetailskim)、 [edencfc](https://github.com/edencfc)).
- 2021.1.11: Optimize the labeling experience (by [edencfc](https://github.com/edencfc)),
  - Users can choose whether to pop up the label input dialog after drawing the detection box in "View - Pop-up Label Input Dialog".
  - The recognition result scrolls synchronously when users click related detection box.
  - Click to modify the recognition result.(If you can't change the result, please switch to the system default input method, or switch back to the original input method again)
- 2020.12.18: Support re-recognition of a single label box (by [ninetailskim](https://github.com/ninetailskim) ), perfect shortcut keys.



## 1. Installation and Run

### 1.1 Install PaddlePaddle

```bash
pip3 install --upgrade pip

# If you have cuda9 or cuda10 installed on your machine, please run the following command to install
python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple

# If you only have cpu on your machine, please run the following command to install
python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

For more software version requirements, please refer to the instructions in [Installation Document](https://www.paddlepaddle.org.cn/install/quick) for operation.

### 1.2 Install and Run PPOCRLabel

PPOCRLabel can be started in two ways: whl package and Python script. The whl package form is more convenient to start, and the python script to start is convenient for secondary development.

#### Windows

```bash
pip install PPOCRLabel  # install

# Select label mode and run 
PPOCRLabel  # [Normal mode] for [detection + recognition] labeling
PPOCRLabel --kie True # [KIE mode] for [detection + recognition + keyword extraction] labeling
```

> If you getting this error `OSError: [WinError 126] The specified module could not be found` when you install shapely on windows. Please try to download Shapely whl file using http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely.
>
> Reference: [Solve shapely installation on windows](https://stackoverflow.com/questions/44398265/install-shapely-oserror-winerror-126-the-specified-module-could-not-be-found)
>

#### Ubuntu Linux

```bash
pip3 install PPOCRLabel
pip3 install trash-cli

# Select label mode and run 
PPOCRLabel  # [Normal mode] for [detection + recognition] labeling
PPOCRLabel --kie True # [KIE mode] for [detection + recognition + keyword extraction] labeling
```

#### MacOS
```bash
pip3 install PPOCRLabel
pip3 install opencv-contrib-python-headless==4.2.0.32

# Select label mode and run 
PPOCRLabel  # [Normal mode] for [detection + recognition] labeling
PPOCRLabel --kie True # [KIE mode] for [detection + recognition + keyword extraction] labeling
```

#### 1.2.2 Run PPOCRLabel by Python Script
If you modify the PPOCRLabel file (for example, specifying a new built-in model), it will be more convenient to see the results by running the Python script. If you still want to start with the whl package, you need to uninstall the whl package in the current environment and then recompile it according to the next section.

```bash
cd ./PPOCRLabel  # Switch to the PPOCRLabel directory

# Select label mode and run 
python PPOCRLabel.py  # [Normal mode] for [detection + recognition] labeling
python PPOCRLabel.py --kie True # [KIE mode] for [detection + recognition + keyword extraction] labeling
```

#### 1.2.3 Build and Install the Whl Package Locally
Compile and install a new whl package, where 1.0.2 is the version number, you can specify the new version in 'setup.py'.
```bash
cd PaddleOCR/PPOCRLabel
python3 setup.py bdist_wheel
pip3 install dist/PPOCRLabel-1.0.2-py2.py3-none-any.whl
```


## 2. Usage

### 2.1 Steps

1. Build and launch using the instructions above.

2. Click 'Open Dir' in Menu/File to select the folder of the picture.<sup>[1]</sup>

3. Click 'Auto recognition', use PPOCR model to automatically annotate images which marked with 'X' <sup>[2]</sup>before the file name.

4. Create Box:

   4.1 Click 'Create RectBox' or press 'W' in English keyboard mode to draw a new rectangle detection box. Click and release left mouse to select a region to annotate the text area.

   4.2 Press 'Q' to enter four-point labeling mode which enables you to create any four-point shape by clicking four points with the left mouse button in succession and DOUBLE CLICK the left mouse as the signal of labeling completion.

5. After the marking frame is drawn, the user clicks "OK", and the detection frame will be pre-assigned a "TEMPORARY" label.

6. Click 're-Recognition', model will rewrite ALL recognition results in ALL detection box<sup>[3]</sup>.

7. Single click the result in 'recognition result' list to manually change inaccurate recognition results.

8. **Click "Check", the image status will switch to "√",then the program automatically jump to the next.**

9. Click "Delete Image" and the image will be deleted to the recycle bin.

10. Labeling result: the user can export the label result manually through the menu "File - Export Label", while the program will also export automatically if "File - Auto export Label Mode" is selected. The manually checked label will be stored in *Label.txt* under the opened picture folder. Click "File"-"Export Recognition Results" in the menu bar, the recognition training data of such pictures will be saved in the *crop_img* folder, and the recognition label will be saved in *rec_gt.txt*<sup>[4]</sup>.

### 2.2 Note

[1] PPOCRLabel uses the opened folder as the project. After opening the image folder, the picture will not be displayed in the dialog. Instead, the pictures under the folder will be directly imported into the program after clicking "Open Dir".

[2] The image status indicates whether the user has saved the image manually. If it has not been saved manually it is "X", otherwise it is "√", PPOCRLabel will not relabel pictures with a status of "√".

[3] After clicking "Re-recognize", the model will overwrite ALL recognition results in the picture. Therefore, if the recognition result has been manually changed before, it may change after re-recognition.

[4] The files produced by PPOCRLabel can be found under the opened picture folder including the following, please do not manually change the contents, otherwise it will cause the program to be abnormal.

|   File name   |                         Description                          |
| :-----------: | :----------------------------------------------------------: |
|   Label.txt   | The detection label file can be directly used for PPOCR detection model training. After the user saves 5 label results, the file will be automatically exported. It will also be written when the user closes the application or changes the file folder. |
| fileState.txt | The picture status file save the image in the current folder that has been manually confirmed by the user. |
|  Cache.cach   |    Cache files to save the results of model recognition.     |
|  rec_gt.txt   | The recognition label file, which can be directly used for PPOCR identification model training, is generated after the user clicks on the menu bar "File"-"Export recognition result". |
|   crop_img    | The recognition data, generated at the same time with *rec_gt.txt* |



## 3. Explanation

### 3.1 Shortcut keys

| Shortcut keys            | Description                                      |
|--------------------------|--------------------------------------------------|
| Ctrl + Shift + R         | Re-recognize all the labels of the current image |
| W                        | Create a rect box                                |
| Q                        | Create a four-points box                         |
| X                        | Rotate the box anti-clockwise                    |
| C                        | Rotate the box clockwise                         |
| Ctrl + E                 | Edit label of the selected box                   |
| Ctrl + X                 | Change key class of the box when enable `--kie`  |
| Ctrl + R                 | Re-recognize the selected box                    |
| Ctrl + C                 | Copy and paste the selected box                  |
| Ctrl + Left Mouse Button | Multi select the label box                       |
| Backspace                 | Delete the selected box                          |
| Ctrl + V                 | Check image                                      |
| Ctrl + Shift + d         | Delete image                                     |
| D                        | Next image                                       |
| A                        | Previous image                                   |
| Ctrl++                   | Zoom in                                          |
| Ctrl--                   | Zoom out                                         |
| ↑→↓←                     | Move selected box                                |

### 3.2 Built-in Model

- Default model: PPOCRLabel uses the Chinese and English ultra-lightweight OCR model in PaddleOCR by default, supports Chinese, English and number recognition, and multiple language detection.

- Model language switching: Changing the built-in model language is supportable by clicking "PaddleOCR"-"Choose OCR Model" in the menu bar. Currently supported languages​include French, German, Korean, and Japanese.
  For specific model download links, please refer to [PaddleOCR Model List](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_en/models_list_en.md#multilingual-recognition-modelupdating)

- **Custom Model**: If users want to replace the built-in model with their own inference model, they can follow the [Custom Model Code Usage](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_en/whl_en.md#31-use-by-code) by modifying PPOCRLabel.py for [Instantiation of PaddleOCR class](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/PPOCRLabel/PPOCRLabel.py#L86) :

  add parameter `det_model_dir`  in `self.ocr = PaddleOCR(use_pdserving=False, use_angle_cls=True, det=True, cls=True, use_gpu=gpu, lang=lang) `

### 3.3 Export Label Result

PPOCRLabel supports three ways to export Label.txt

- Automatically export: After selecting "File - Auto Export Label Mode", the program will automatically write the annotations into Label.txt every time the user confirms an image. If this option is not turned on, it will be automatically exported after detecting that the user has manually checked 5 images.

  > The automatically export mode is turned off by default

- Manual export: Click "File-Export Marking Results" to manually export the label.

- Close application export


### 3.4 Export Partial Recognition Results

For some data that are difficult to recognize, the recognition results will not be exported by **unchecking** the corresponding tags in the recognition results checkbox. The unchecked recognition result is saved as `True` in the `difficult` variable in the label file `label.txt`.

> *Note: The status of the checkboxes in the recognition results still needs to be saved manually by clicking Save Button.*

### 3.5 Dataset division

- Enter the following command in the terminal to execute the dataset division script:

    ```
  cd ./PPOCRLabel # Change the directory to the PPOCRLabel folder
  python gen_ocr_train_val_test.py --trainValTestRatio 6:2:2 --datasetRootPath ../train_data 
  ```

  Parameter Description:

  - `trainValTestRatio` is the division ratio of the number of images in the training set, validation set, and test set, set according to your actual situation, the default is `6:2:2`

  - `datasetRootPath` is the storage path of the complete dataset labeled by PPOCRLabel. The default path is `PaddleOCR/train_data` .
  ```
  |-train_data
    |-crop_img
      |- word_001_crop_0.png
      |- word_002_crop_0.jpg
      |- word_003_crop_0.jpg
      | ...
    | Label.txt
    | rec_gt.txt
    |- word_001.png
    |- word_002.jpg
    |- word_003.jpg
    | ...
  ```
  
### 3.6 Error message

- If paddleocr is installed with whl, it has a higher priority than calling PaddleOCR class with paddleocr.py, which may cause an exception if whl package is not updated.

- For Linux users, if you get an error starting with **objc[XXXXX]** when opening the software, it proves that your opencv version is too high. It is recommended to install version 4.2:

    ```
    pip install opencv-python==4.2.0.32
    ```
- If you get an error starting with **Missing string id **,you need to recompile resources:
    ```
    pyrcc5 -o libs/resources.py resources.qrc
    ```
- If you get an error ``` module 'cv2' has no attribute 'INTER_NEAREST'```, you need to delete all opencv related packages first, and then reinstall the 4.2.0.32  version of headless opencv
    ```
    pip install opencv-contrib-python-headless==4.2.0.32
    ```



### 4. Related

1.[Tzutalin. LabelImg. Git code (2015)](https://github.com/tzutalin/labelImg)
