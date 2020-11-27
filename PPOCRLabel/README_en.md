# PPOCRLabel

PPOCRLabel is a semi-automatic graphic annotation tool suitable for OCR field. It is written in python3 and pyqt5. Support rectangular frame labeling and four-point labeling mode. Annotations can be directly used for the training of PPOCR detection and recognition models.

<img src="./data/gif/steps.gif" width="100%"/>

## Installation

### 1. Install PaddleOCR

Refer to [PaddleOCR installation document](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/installation.md) to prepare PaddleOCR

### 2. Install PPOCRLabel

#### Windows + Anaconda

Download and install [Anaconda](https://www.anaconda.com/download/#download) (Python 3+)

```
conda install pyqt=5
cd ./PPOCRLabel # Change the directory to the PPOCRLabel folder
pyrcc5 -o libs/resources.py resources.qrc
python PPOCRLabel.py
```

#### Ubuntu Linux

```
sudo apt-get install pyqt5-dev-tools
sudo apt-get install trash-cli
cd ./PPOCRLabel # Change the directory to the PPOCRLabel folder
sudo pip3 install -r requirements/requirements-linux-python3.txt
make qt5py3
python3 PPOCRLabel.py
```

#### macOS
```
pip3 install pyqt5
pip3 uninstall opencv-python # Uninstall opencv manually as it conflicts with pyqt
pip3 install opencv-contrib-python-headless # Install the headless version of opencv
cd ./PPOCRLabel # Change the directory to the PPOCRLabel folder
make qt5py3
python3 PPOCRLabel.py
```

## Usage

### Steps

1. Build and launch using the instructions above.

2. Click 'Open Dir' in Menu/File to select the folder of the picture.<sup>[1]</sup>

3. Click 'Auto recognition', use PPOCR model to automatically annotate images which marked with 'X' <sup>[2]</sup>before the file name.

4. Create Box:

   4.1 Click 'Create RectBox' or press 'W' in English keyboard mode to draw a new rectangle detection box. Click and release left mouse to select a region to annotate the text area.

   4.2 Press 'P' to enter four-point labeling mode which enables you to create any four-point shape by clicking four points with the left mouse button in succession and DOUBLE CLICK the left mouse as the signal of labeling completion.

5. After the marking frame is drawn, the user clicks "OK", and the detection frame will be pre-assigned a "TEMPORARY" label.

6. Click 're-Recognition', model will rewrite ALL recognition results in ALL detection box<sup>[3]</sup>.

7. Double click the result in 'recognition result' list to manually change inaccurate recognition results.

8. Click "Save", the image status will switch to "√",then the program automatically jump to the next.

9. Click "Delete Image" and the image will be deleted to the recycle bin.

10. Labeling result: After closing the application or switching the file path, the manually saved label will be stored in *Label.txt* under the opened picture folder.
    Click "PaddleOCR"-"Save Recognition Results" in the menu bar, the recognition training data of such pictures will be saved in the *crop_img* folder, and the recognition label will be saved in *rec_gt.txt*<sup>[4]</sup>.

### Note

[1] PPOCRLabel uses the opened folder as the project. After opening the image folder, the picture will not be displayed in the dialog. Instead, the pictures under the folder will be directly imported into the program after clicking "Open Dir".

[2] The image status indicates whether the user has saved the image manually. If it has not been saved manually it is "X", otherwise it is "√", PPOCRLabel will not relabel pictures with a status of "√".

[3] After clicking "Re-recognize", the model will overwrite ALL recognition results in the picture.
Therefore, if the recognition result has been manually changed before, it may change after re-recognition.

[4] The files produced by PPOCRLabel include the following, please do not manually change the contents, otherwise it will cause the program to be abnormal.

|   File name   |                         Description                          |
| :-----------: | :----------------------------------------------------------: |
|   Label.txt   | The detection label file can be directly used for PPOCR detection model training. After the user saves 10 label results, the file will be automatically saved. It will also be written when the user closes the application or changes the file folder. |
| fileState.txt | The picture status file save the image in the current folder that has been manually confirmed by the user. |
|  Cache.cach   |    Cache files to save the results of model recognition.     |
|  rec_gt.txt   | The recognition label file, which can be directly used for PPOCR identification model training, is generated after the user clicks on the menu bar "PaddleOCR"-"Save recognition result". |
|   crop_img    | The recognition data, generated at the same time with *rec_gt.txt* |

## Related

1.[Tzutalin. LabelImg. Git code (2015)](https://github.com/tzutalin/labelImg)
