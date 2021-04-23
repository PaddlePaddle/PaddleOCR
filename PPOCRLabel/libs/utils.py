# Copyright (c) <2015-Present> Tzutalin
# Copyright (C) 2013  MIT, Computer Science and Artificial Intelligence Laboratory. Bryan Russell, Antonio Torralba,
# William T. Freeman. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
# SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from math import sqrt
from libs.ustr import ustr
import hashlib
import re
import sys
import cv2
import numpy as np

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *


def newIcon(icon, iconSize=None):
    if iconSize is not None:
        return QIcon(QIcon(':/' + icon).pixmap(iconSize,iconSize))
    else:
        return QIcon(':/' + icon)


def newButton(text, icon=None, slot=None):
    b = QPushButton(text)
    if icon is not None:
        b.setIcon(newIcon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b


def newAction(parent, text, slot=None, shortcut=None, icon=None,
              tip=None, checkable=False, enabled=True, iconSize=None):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QAction(text, parent)
    if icon is not None:
        if iconSize is not None:
            a.setIcon(newIcon(icon, iconSize))
        else:
            a.setIcon(newIcon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    return a


def addActions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)


def labelValidator():
    return QRegExpValidator(QRegExp(r'^[^ \t].+'), None)


class struct(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def fmtShortcut(text):
    mod, key = text.split('+', 1)
    return '<b>%s</b>+<b>%s</b>' % (mod, key)


def generateColorByText(text):
    s = ustr(text)
    hashCode = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
    r = int((hashCode / 255) % 255)
    g = int((hashCode / 65025)  % 255)
    b = int((hashCode / 16581375)  % 255)
    return QColor(r, g, b, 100)

def have_qstring():
    '''p3/qt5 get rid of QString wrapper as py3 has native unicode str type'''
    return not (sys.version_info.major >= 3 or QT_VERSION_STR.startswith('5.'))

def util_qt_strlistclass():
    return QStringList if have_qstring() else list

def natural_sort(list, key=lambda s:s):
    """
    Sort the list into natural alphanumeric order.
    """
    def get_alphanum_key_func(key):
        convert = lambda text: int(text) if text.isdigit() else text
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]
    sort_key = get_alphanum_key_func(key)
    list.sort(key=sort_key)


def get_rotate_crop_image(img, points):
    # Use Green's theory to judge clockwise or counterclockwise
    # author: biyanhua
    d = 0.0
    for index in range(-1, 3):
        d += -0.5 * (points[index + 1][1] + points[index][1]) * (
                    points[index + 1][0] - points[index][0])
    if d < 0: # counterclockwise
        tmp = np.array(points)
        points[1], points[3] = tmp[3], tmp[1]

    try:
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
    except Exception as e:
        print(e)

def stepsInfo(lang='en'):
    if lang == 'ch':
        msg = "1. 安装与运行：使用上述命令安装与运行程序。\n" \
              "2. 打开文件夹：在菜单栏点击 “文件” - 打开目录 选择待标记图片的文件夹.\n"\
              "3. 自动标注：点击 ”自动标注“，使用PPOCR超轻量模型对图片文件名前图片状态为 “X” 的图片进行自动标注。\n" \
              "4. 手动标注：点击 “矩形标注”（推荐直接在英文模式下点击键盘中的 “W”)，用户可对当前图片中模型未检出的部分进行手动" \
              "绘制标记框。点击键盘P，则使用四点标注模式（或点击“编辑” - “四点标注”），用户依次点击4个点后，双击左键表示标注完成。\n" \
              "5. 标记框绘制完成后，用户点击 “确认”，检测框会先被预分配一个 “待识别” 标签。\n" \
              "6. 重新识别：将图片中的所有检测画绘制/调整完成后，点击 “重新识别”，PPOCR模型会对当前图片中的**所有检测框**重新识别。\n" \
              "7. 内容更改：双击识别结果，对不准确的识别结果进行手动更改。\n" \
              "8. 保存：点击 “保存”，图片状态切换为 “√”，跳转至下一张。\n" \
              "9. 删除：点击 “删除图像”，图片将会被删除至回收站。\n" \
              "10. 标注结果：关闭应用程序或切换文件路径后，手动保存过的标签将会被存放在所打开图片文件夹下的" \
              "*Label.txt*中。在菜单栏点击 “PaddleOCR” - 保存识别结果后，会将此类图片的识别训练数据保存在*crop_img*文件夹下，" \
              "识别标签保存在*rec_gt.txt*中。\n"
    else:
        msg = "1. Build and launch using the instructions above.\n" \
              "2. Click 'Open Dir' in Menu/File to select the folder of the picture.\n"\
              "3. Click 'Auto recognition', use PPOCR model to automatically annotate images which marked with 'X' before the file name."\
              "4. Create Box:\n"\
                   "4.1 Click 'Create RectBox' or press 'W' in English keyboard mode to draw a new rectangle detection box. Click and release left mouse to select a region to annotate the text area.\n"\
                   "4.2 Press 'P' to enter four-point labeling mode which enables you to create any four-point shape by clicking four points with the left mouse button in succession and DOUBLE CLICK the left mouse as the signal of labeling completion.\n"\
              "5. After the marking frame is drawn, the user clicks 'OK', and the detection frame will be pre-assigned a TEMPORARY label.\n"\
              "6. Click re-Recognition, model will rewrite ALL recognition results in ALL detection box.\n"\
              "7. Double click the result in 'recognition result' list to manually change inaccurate recognition results.\n"\
              "8. Click 'Save', the image status will switch to '√',then the program automatically jump to the next.\n"\
              "9. Click 'Delete Image' and the image will be deleted to the recycle bin.\n"\
              "10. Labeling result: After closing the application or switching the file path, the manually saved label will be stored in *Label.txt* under the opened picture folder.\n"\
              "    Click PaddleOCR-Save Recognition Results in the menu bar, the recognition training data of such pictures will be saved in the *crop_img* folder, and the recognition label will be saved in *rec_gt.txt*.\n"
    return msg