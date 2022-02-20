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

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# pyrcc5 -o libs/resources.py resources.qrc
import argparse
import ast
import codecs
import json
import os.path
import platform
import subprocess
import sys
from functools import partial

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    print("Please install pyqt5...")

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../PaddleOCR')))
sys.path.append("..")

from paddleocr import PaddleOCR
from libs.constants import *
from libs.utils import *
from libs.settings import Settings
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR, DEFAULT_LOCK_COLOR
from libs.stringBundle import StringBundle
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.autoDialog import AutoDialog
from libs.labelDialog import LabelDialog
from libs.colorDialog import ColorDialog
from libs.ustr import ustr
from libs.hashableQListWidgetItem import HashableQListWidgetItem
from libs.editinlist import EditInList

__appname__ = 'PPOCRLabel'


class MainWindow(QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self,
                 lang="ch",
                 gpu=False,
                 kei_mode=False,
                 default_filename=None,
                 default_predefined_class_file=None,
                 default_save_dir=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)
        self.setWindowState(Qt.WindowMaximized)  # set window max
        self.activateWindow()  # PPOCRLabel goes to the front when activate

        # Load setting in the main thread
        self.settings = Settings()
        self.settings.load()
        settings = self.settings
        self.lang = lang
        self.kie_mode = kei_mode
        # Load string bundle for i18n
        if lang not in ['ch', 'en']:
            lang = 'en'
        self.stringBundle = StringBundle.getBundle(localeStr='zh-CN' if lang == 'ch' else 'en')  # 'en'
        getStr = lambda strId: self.stringBundle.getString(strId)

        self.defaultSaveDir = default_save_dir
        self.ocr = PaddleOCR(use_pdserving=False,
                             use_angle_cls=True,
                             det=True,
                             cls=True,
                             use_gpu=gpu,
                             lang=lang,
                             show_log=False)

        if os.path.exists('./data/paddle.png'):
            result = self.ocr.ocr('./data/paddle.png', cls=True, det=True)

        # For loading all image under a directory
        self.mImgList = []
        self.mImgList5 = []
        self.dirname = None
        self.labelHist = []
        self.lastOpenDir = None
        self.result_dic = []
        self.result_dic_locked = []
        self.changeFileFolder = False
        self.haveAutoReced = False
        self.labelFile = None
        self.currIndex = 0

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False
        self._beginner = True
        self.screencastViewer = self.getAvailableScreencastViewer()
        self.screencast = "https://github.com/PaddlePaddle/PaddleOCR"

        # Load predefined classes to the list
        self.loadPredefinedClasses(default_predefined_class_file)

        # Main widgets and related state.
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)
        self.autoDialog = AutoDialog(parent=self)

        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.itemsToShapesbox = {}
        self.shapesToItemsbox = {}
        self.prevLabelText = getStr('tempLabel')
        self.noLabelText = getStr('nullLabel')
        self.model = 'paddle'
        self.PPreader = None
        self.autoSaveNum = 5

        #  ================== File List  ==================

        filelistLayout = QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)

        self.fileListWidget = QListWidget()
        self.fileListWidget.itemClicked.connect(self.fileitemDoubleClicked)
        self.fileListWidget.setIconSize(QSize(25, 25))
        filelistLayout.addWidget(self.fileListWidget)

        self.AutoRecognition = QToolButton()
        self.AutoRecognition.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.AutoRecognition.setIcon(newIcon('Auto'))
        autoRecLayout = QHBoxLayout()
        autoRecLayout.setContentsMargins(0, 0, 0, 0)
        autoRecLayout.addWidget(self.AutoRecognition)
        autoRecContainer = QWidget()
        autoRecContainer.setLayout(autoRecLayout)
        filelistLayout.addWidget(autoRecContainer)

        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.fileListName = getStr('fileList')
        self.fileDock = QDockWidget(self.fileListName, self)
        self.fileDock.setObjectName(getStr('files'))
        self.fileDock.setWidget(fileListContainer)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.fileDock)

        #  ================== Key List  ==================
        if self.kie_mode:
            self.keyList = QListWidget()

            # self.keyList.itemActivated.connect(self.boxSelectionChanged)
            self.keyList.itemSelectionChanged.connect(self.keyListSelectionChanged)
            self.keyList.itemDoubleClicked.connect(self.editBox)
            # Connect to itemChanged to detect checkbox changes.
            self.keyList.itemChanged.connect(self.keyListItemChanged)
            self.keyListDockName = getStr('keyListTitle')
            self.keyListDock = QDockWidget(self.keyListDockName, self)
            self.keyListDock.setWidget(self.keyList)
            self.keyListDock.setFeatures(QDockWidget.NoDockWidgetFeatures)
            filelistLayout.addWidget(self.keyListDock)

        #  ================== Right Area  ==================
        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)

        # Buttons
        self.editButton = QToolButton()
        self.reRecogButton = QToolButton()
        self.reRecogButton.setIcon(newIcon('reRec', 30))
        self.reRecogButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        self.newButton = QToolButton()
        self.newButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.SaveButton = QToolButton()
        self.SaveButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.DelButton = QToolButton()
        self.DelButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        leftTopToolBox = QHBoxLayout()
        leftTopToolBox.addWidget(self.newButton)
        leftTopToolBox.addWidget(self.reRecogButton)
        leftTopToolBoxContainer = QWidget()
        leftTopToolBoxContainer.setLayout(leftTopToolBox)
        listLayout.addWidget(leftTopToolBoxContainer)

        #  ================== Label List  ==================
        # Create and add a widget for showing current label items
        self.labelList = EditInList()
        labelListContainer = QWidget()
        labelListContainer.setLayout(listLayout)
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.clicked.connect(self.labelList.item_clicked)

        # Connect to itemChanged to detect checkbox changes.
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelListDockName = getStr('recognitionResult')
        self.labelListDock = QDockWidget(self.labelListDockName, self)
        self.labelListDock.setWidget(self.labelList)
        self.labelListDock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        listLayout.addWidget(self.labelListDock)

        #  ================== Detection Box  ==================
        self.BoxList = QListWidget()

        # self.BoxList.itemActivated.connect(self.boxSelectionChanged)
        self.BoxList.itemSelectionChanged.connect(self.boxSelectionChanged)
        self.BoxList.itemDoubleClicked.connect(self.editBox)
        # Connect to itemChanged to detect checkbox changes.
        self.BoxList.itemChanged.connect(self.boxItemChanged)
        self.BoxListDockName = getStr('detectionBoxposition')
        self.BoxListDock = QDockWidget(self.BoxListDockName, self)
        self.BoxListDock.setWidget(self.BoxList)
        self.BoxListDock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        listLayout.addWidget(self.BoxListDock)

        #  ================== Lower Right Area  ==================
        leftbtmtoolbox = QHBoxLayout()
        leftbtmtoolbox.addWidget(self.SaveButton)
        leftbtmtoolbox.addWidget(self.DelButton)
        leftbtmtoolboxcontainer = QWidget()
        leftbtmtoolboxcontainer.setLayout(leftbtmtoolbox)
        listLayout.addWidget(leftbtmtoolboxcontainer)

        self.dock = QDockWidget(getStr('boxLabelText'), self)
        self.dock.setObjectName(getStr('labels'))
        self.dock.setWidget(labelListContainer)

        #  ================== Zoom Bar  ==================
        self.imageSlider = QSlider(Qt.Horizontal)
        self.imageSlider.valueChanged.connect(self.CanvasSizeChange)
        self.imageSlider.setMinimum(-9)
        self.imageSlider.setMaximum(510)
        self.imageSlider.setSingleStep(1)
        self.imageSlider.setTickPosition(QSlider.TicksBelow)
        self.imageSlider.setTickInterval(1)

        op = QGraphicsOpacityEffect()
        op.setOpacity(0.2)
        self.imageSlider.setGraphicsEffect(op)

        self.imageSlider.setStyleSheet("background-color:transparent")
        self.imageSliderDock = QDockWidget(getStr('ImageResize'), self)
        self.imageSliderDock.setObjectName(getStr('IR'))
        self.imageSliderDock.setWidget(self.imageSlider)
        self.imageSliderDock.setFeatures(QDockWidget.DockWidgetFloatable)
        self.imageSliderDock.setAttribute(Qt.WA_TranslucentBackground)
        self.addDockWidget(Qt.RightDockWidgetArea, self.imageSliderDock)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)
        self.zoomWidgetValue = self.zoomWidget.value()

        self.msgBox = QMessageBox()

        #  ================== Thumbnail ==================
        hlayout = QHBoxLayout()
        m = (0, 0, 0, 0)
        hlayout.setSpacing(0)
        hlayout.setContentsMargins(*m)
        self.preButton = QToolButton()
        self.preButton.setIcon(newIcon("prev", 40))
        self.preButton.setIconSize(QSize(40, 100))
        self.preButton.clicked.connect(self.openPrevImg)
        self.preButton.setStyleSheet('border: none;')
        self.preButton.setShortcut('a')
        self.iconlist = QListWidget()
        self.iconlist.setViewMode(QListView.IconMode)
        self.iconlist.setFlow(QListView.TopToBottom)
        self.iconlist.setSpacing(10)
        self.iconlist.setIconSize(QSize(50, 50))
        self.iconlist.setMovement(QListView.Static)
        self.iconlist.setResizeMode(QListView.Adjust)
        self.iconlist.itemClicked.connect(self.iconitemDoubleClicked)
        self.iconlist.setStyleSheet("QListWidget{ background-color:transparent; border: none;}")
        self.iconlist.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nextButton = QToolButton()
        self.nextButton.setIcon(newIcon("next", 40))
        self.nextButton.setIconSize(QSize(40, 100))
        self.nextButton.setStyleSheet('border: none;')
        self.nextButton.clicked.connect(self.openNextImg)
        self.nextButton.setShortcut('d')

        hlayout.addWidget(self.preButton)
        hlayout.addWidget(self.iconlist)
        hlayout.addWidget(self.nextButton)

        iconListContainer = QWidget()
        iconListContainer.setLayout(hlayout)
        iconListContainer.setFixedHeight(100)

        #  ================== Canvas ==================
        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.setDrawingShapeToSquare(settings.get(SETTING_DRAW_SQUARE, False))

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(partial(self.newShape, False))
        self.canvas.shapeMoved.connect(self.updateBoxlist)  # self.setDirty
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        centerLayout = QVBoxLayout()
        centerLayout.setContentsMargins(0, 0, 0, 0)
        centerLayout.addWidget(scroll)
        centerLayout.addWidget(iconListContainer, 0, Qt.AlignCenter)
        centerContainer = QWidget()
        centerContainer.setLayout(centerLayout)

        self.setCentralWidget(centerContainer)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)

        self.dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable)
        self.fileDock.setFeatures(QDockWidget.NoDockWidgetFeatures)

        #  ================== Actions ==================
        action = partial(newAction, self)
        quit = action(getStr('quit'), self.close,
                      'Ctrl+Q', 'quit', getStr('quitApp'))

        opendir = action(getStr('openDir'), self.openDirDialog,
                         'Ctrl+u', 'open', getStr('openDir'))

        open_dataset_dir = action(getStr('openDatasetDir'), self.openDatasetDirDialog,
                                  'Ctrl+p', 'open', getStr('openDatasetDir'), enabled=False)

        save = action(getStr('save'), self.saveFile,
                      'Ctrl+V', 'verify', getStr('saveDetail'), enabled=False)

        alcm = action(getStr('choosemodel'), self.autolcm,
                      'Ctrl+M', 'next', getStr('tipchoosemodel'))

        deleteImg = action(getStr('deleteImg'), self.deleteImg, 'Ctrl+Shift+D', 'close', getStr('deleteImgDetail'),
                           enabled=True)

        resetAll = action(getStr('resetAll'), self.resetAll, None, 'resetall', getStr('resetAllDetail'))

        color1 = action(getStr('boxLineColor'), self.chooseColor1,
                        'Ctrl+L', 'color_line', getStr('boxLineColorDetail'))

        createMode = action(getStr('crtBox'), self.setCreateMode,
                            'w', 'new', getStr('crtBoxDetail'), enabled=False)
        editMode = action('&Edit\nRectBox', self.setEditMode,
                          'Ctrl+J', 'edit', u'Move and edit Boxs', enabled=False)

        create = action(getStr('crtBox'), self.createShape,
                        'w', 'objects', getStr('crtBoxDetail'), enabled=False)

        delete = action(getStr('delBox'), self.deleteSelectedShape,
                        'Alt+X', 'delete', getStr('delBoxDetail'), enabled=False)

        copy = action(getStr('dupBox'), self.copySelectedShape,
                      'Ctrl+C', 'copy', getStr('dupBoxDetail'),
                      enabled=False)

        hideAll = action(getStr('hideBox'), partial(self.togglePolygons, False),
                         'Ctrl+H', 'hide', getStr('hideAllBoxDetail'),
                         enabled=False)
        showAll = action(getStr('showBox'), partial(self.togglePolygons, True),
                         'Ctrl+A', 'hide', getStr('showAllBoxDetail'),
                         enabled=False)

        help = action(getStr('tutorial'), self.showTutorialDialog, None, 'help', getStr('tutorialDetail'))
        showInfo = action(getStr('info'), self.showInfoDialog, None, 'help', getStr('info'))
        showSteps = action(getStr('steps'), self.showStepsDialog, None, 'help', getStr('steps'))
        showKeys = action(getStr('keys'), self.showKeysDialog, None, 'help', getStr('keys'))

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action(getStr('zoomin'), partial(self.addZoom, 10),
                        'Ctrl++', 'zoom-in', getStr('zoominDetail'), enabled=False)
        zoomOut = action(getStr('zoomout'), partial(self.addZoom, -10),
                         'Ctrl+-', 'zoom-out', getStr('zoomoutDetail'), enabled=False)
        zoomOrg = action(getStr('originalsize'), partial(self.setZoom, 100),
                         'Ctrl+=', 'zoom', getStr('originalsizeDetail'), enabled=False)
        fitWindow = action(getStr('fitWin'), self.setFitWindow,
                           'Ctrl+F', 'fit-window', getStr('fitWinDetail'),
                           checkable=True, enabled=False)
        fitWidth = action(getStr('fitWidth'), self.setFitWidth,
                          'Ctrl+Shift+F', 'fit-width', getStr('fitWidthDetail'),
                          checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut,
                       zoomOrg, fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(getStr('editLabel'), self.editLabel,
                      'Ctrl+E', 'edit', getStr('editLabelDetail'),
                      enabled=False)

        #  ================== New Actions ==================
        AutoRec = action(getStr('autoRecognition'), self.autoRecognition,
                         '', 'Auto', getStr('autoRecognition'), enabled=False)

        reRec = action(getStr('reRecognition'), self.reRecognition,
                       'Ctrl+Shift+R', 'reRec', getStr('reRecognition'), enabled=False)

        singleRere = action(getStr('singleRe'), self.singleRerecognition,
                            'Ctrl+R', 'reRec', getStr('singleRe'), enabled=False)

        createpoly = action(getStr('creatPolygon'), self.createPolygon,
                            'q', 'new', getStr('creatPolygon'), enabled=True)

        saveRec = action(getStr('saveRec'), self.saveRecResult,
                         '', 'save', getStr('saveRec'), enabled=False)

        saveLabel = action(getStr('saveLabel'), self.saveLabelFile,  #
                           'Ctrl+S', 'save', getStr('saveLabel'), enabled=False)

        undoLastPoint = action(getStr("undoLastPoint"), self.canvas.undoLastPoint,
                               'Ctrl+Z', "undo", getStr("undoLastPoint"), enabled=False)

        rotateLeft = action(getStr("rotateLeft"), partial(self.rotateImgAction, 1),
                            'Ctrl+Alt+L', "rotateLeft", getStr("rotateLeft"), enabled=False)

        rotateRight = action(getStr("rotateRight"), partial(self.rotateImgAction, -1),
                             'Ctrl+Alt+R', "rotateRight", getStr("rotateRight"), enabled=False)

        undo = action(getStr("undo"), self.undoShapeEdit,
                      'Ctrl+Z', "undo", getStr("undo"), enabled=False)

        lock = action(getStr("lockBox"), self.lockSelectedShape,
                      None, "lock", getStr("lockBoxDetail"),
                      enabled=False)

        self.editButton.setDefaultAction(edit)
        self.newButton.setDefaultAction(create)
        self.DelButton.setDefaultAction(deleteImg)
        self.SaveButton.setDefaultAction(save)
        self.AutoRecognition.setDefaultAction(AutoRec)
        self.reRecogButton.setDefaultAction(reRec)
        # self.preButton.setDefaultAction(openPrevImg)
        # self.nextButton.setDefaultAction(openNextImg)

        #  ================== Zoom layout ==================
        zoomLayout = QHBoxLayout()
        zoomLayout.addStretch()
        self.zoominButton = QToolButton()
        self.zoominButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.zoominButton.setDefaultAction(zoomIn)
        self.zoomoutButton = QToolButton()
        self.zoomoutButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.zoomoutButton.setDefaultAction(zoomOut)
        self.zoomorgButton = QToolButton()
        self.zoomorgButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.zoomorgButton.setDefaultAction(zoomOrg)
        zoomLayout.addWidget(self.zoominButton)
        zoomLayout.addWidget(self.zoomorgButton)
        zoomLayout.addWidget(self.zoomoutButton)

        zoomContainer = QWidget()
        zoomContainer.setLayout(zoomLayout)
        zoomContainer.setGeometry(0, 0, 30, 150)

        shapeLineColor = action(getStr('shapeLineColor'), self.chshapeLineColor,
                                icon='color_line', tip=getStr('shapeLineColorDetail'),
                                enabled=False)
        shapeFillColor = action(getStr('shapeFillColor'), self.chshapeFillColor,
                                icon='color', tip=getStr('shapeFillColorDetail'),
                                enabled=False)

        # Label list context menu.
        labelMenu = QMenu()
        addActions(labelMenu, (edit, delete))

        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu)

        # Draw squares/rectangles
        self.drawSquaresOption = QAction(getStr('drawSquares'), self)
        self.drawSquaresOption.setCheckable(True)
        self.drawSquaresOption.setChecked(settings.get(SETTING_DRAW_SQUARE, False))
        self.drawSquaresOption.triggered.connect(self.toogleDrawSquare)

        # Store actions for further handling.
        self.actions = struct(save=save, resetAll=resetAll, deleteImg=deleteImg,
                              lineColor=color1, create=create, delete=delete, edit=edit, copy=copy,
                              saveRec=saveRec, singleRere=singleRere, AutoRec=AutoRec, reRec=reRec,
                              createMode=createMode, editMode=editMode,
                              shapeLineColor=shapeLineColor, shapeFillColor=shapeFillColor,
                              zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
                              fitWindow=fitWindow, fitWidth=fitWidth,
                              zoomActions=zoomActions, saveLabel=saveLabel,
                              undo=undo, undoLastPoint=undoLastPoint, open_dataset_dir=open_dataset_dir,
                              rotateLeft=rotateLeft, rotateRight=rotateRight, lock=lock,
                              fileMenuActions=(opendir, open_dataset_dir, saveLabel, resetAll, quit),
                              beginner=(), advanced=(),
                              editMenu=(createpoly, edit, copy, delete, singleRere, None, undo, undoLastPoint,
                                        None, rotateLeft, rotateRight, None, color1, self.drawSquaresOption, lock),
                              beginnerContext=(create, edit, copy, delete, singleRere, rotateLeft, rotateRight, lock),
                              advancedContext=(createMode, editMode, edit, copy,
                                               delete, shapeLineColor, shapeFillColor),
                              onLoadActive=(create, createMode, editMode),
                              onShapesPresent=(hideAll, showAll))

        # menus
        self.menus = struct(
            file=self.menu('&' + getStr('mfile')),
            edit=self.menu('&' + getStr('medit')),
            view=self.menu('&' + getStr('mview')),
            autolabel=self.menu('&PaddleOCR'),
            help=self.menu('&' + getStr('mhelp')),
            recentFiles=QMenu('Open &Recent'),
            labelList=labelMenu)

        self.lastLabel = None
        # Add option to enable/disable labels being displayed at the top of bounding boxes
        self.displayLabelOption = QAction(getStr('displayLabel'), self)
        self.displayLabelOption.setShortcut("Ctrl+Shift+P")
        self.displayLabelOption.setCheckable(True)
        self.displayLabelOption.setChecked(settings.get(SETTING_PAINT_LABEL, False))
        self.displayLabelOption.triggered.connect(self.togglePaintLabelsOption)

        self.labelDialogOption = QAction(getStr('labelDialogOption'), self)
        self.labelDialogOption.setShortcut("Ctrl+Shift+L")
        self.labelDialogOption.setCheckable(True)
        self.labelDialogOption.setChecked(settings.get(SETTING_PAINT_LABEL, False))
        self.labelDialogOption.triggered.connect(self.speedChoose)

        self.autoSaveOption = QAction(getStr('autoSaveMode'), self)
        self.autoSaveOption.setCheckable(True)
        self.autoSaveOption.setChecked(settings.get(SETTING_PAINT_LABEL, False))
        self.autoSaveOption.triggered.connect(self.autoSaveFunc)

        addActions(self.menus.file,
                   (opendir, open_dataset_dir, None, saveLabel, saveRec, self.autoSaveOption, None, resetAll, deleteImg,
                    quit))

        addActions(self.menus.help, (showKeys, showSteps, showInfo))
        addActions(self.menus.view, (
            self.displayLabelOption, self.labelDialogOption,
            None,
            hideAll, showAll, None,
            zoomIn, zoomOut, zoomOrg, None,
            fitWindow, fitWidth))

        addActions(self.menus.autolabel, (AutoRec, reRec, alcm, None, help))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        addActions(self.canvas.menus[0], self.actions.beginnerContext)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QImage()
        self.filePath = ustr(default_filename)
        self.lastOpenDir = None
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False
        # Add Chris
        self.difficult = False

        # Fix the compatible issue for qt4 and qt5. Convert the QStringList to python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recentFileQStringList = settings.get(SETTING_RECENT_FILES)
                self.recentFiles = [ustr(i) for i in recentFileQStringList]
            else:
                self.recentFiles = recentFileQStringList = settings.get(SETTING_RECENT_FILES)

        size = settings.get(SETTING_WIN_SIZE, QSize(1200, 800))

        position = QPoint(0, 0)
        saved_position = settings.get(SETTING_WIN_POSE, position)
        # Fix the multiple monitors issue
        for i in range(QApplication.desktop().screenCount()):
            if QApplication.desktop().availableGeometry(i).contains(saved_position):
                position = saved_position
                break
        self.resize(size)
        self.move(position)
        saveDir = ustr(settings.get(SETTING_SAVE_DIR, None))
        self.lastOpenDir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))

        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))
        Shape.line_color = self.lineColor = QColor(settings.get(SETTING_LINE_COLOR, DEFAULT_LINE_COLOR))
        Shape.fill_color = self.fillColor = QColor(settings.get(SETTING_FILL_COLOR, DEFAULT_FILL_COLOR))
        self.canvas.setDrawingColor(self.lineColor)
        # Add chris
        Shape.difficult = self.difficult

        # ADD:
        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Since loading the file may take some time, make sure it runs in the background.
        if self.filePath and os.path.isdir(self.filePath):
            self.queueEvent(partial(self.importDirImages, self.filePath or ""))
        elif self.filePath:
            self.queueEvent(partial(self.loadFile, self.filePath or ""))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # Display cursor coordinates at the right of status bar
        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)

        # Open Dir if deafult file
        if self.filePath and os.path.isdir(self.filePath):
            self.openDirDialog(dirpath=self.filePath, silent=True)

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.canvas.setDrawingShapeToSquare(False)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            # Draw rectangle if Ctrl is pressed
            self.canvas.setDrawingShapeToSquare(True)

    def noShapes(self):
        return not self.itemsToShapes

    def populateModeActions(self):
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        self.menus.edit.clear()
        actions = (self.actions.create,)  # if self.beginner() else (self.actions.createMode, self.actions.editMode)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.itemsToShapesbox.clear()  # ADD
        self.shapesToItemsbox.clear()
        self.labelList.clear()
        self.BoxList.clear()
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()
        self.labelCoordinates.clear()
        # self.comboBox.cb.clear()
        self.result_dic = []

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def currentBox(self):
        items = self.BoxList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    def getAvailableScreencastViewer(self):
        osName = platform.system()

        if osName == 'Windows':
            return ['C:\\Program Files\\Internet Explorer\\iexplore.exe']
        elif osName == 'Linux':
            return ['xdg-open']
        elif osName == 'Darwin':
            return ['open']

    ## Callbacks ##
    def showTutorialDialog(self):
        subprocess.Popen(self.screencastViewer + [self.screencast])

    def showInfoDialog(self):
        from libs.__init__ import __version__
        msg = u'Name:{0} \nApp Version:{1} \n{2} '.format(__appname__, __version__, sys.version_info)
        QMessageBox.information(self, u'Information', msg)

    def showStepsDialog(self):
        msg = stepsInfo(self.lang)
        QMessageBox.information(self, u'Information', msg)

    def showKeysDialog(self):
        msg = keysInfo(self.lang)
        QMessageBox.information(self, u'Information', msg)

    def createShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.actions.create.setEnabled(False)
        self.canvas.fourpoint = False

    def createPolygon(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.canvas.fourpoint = True
        self.actions.create.setEnabled(False)
        self.actions.undoLastPoint.setEnabled(True)

    def rotateImg(self, filename, k, _value):

        self.actions.rotateRight.setEnabled(_value)
        pix = cv2.imread(filename)
        pix = np.rot90(pix, k)
        cv2.imwrite(filename, pix)
        self.canvas.update()
        self.loadFile(filename)

    def rotateImgWarn(self):
        if self.lang == 'ch':
            self.msgBox.warning(self, "提示", "\n 该图片已经有标注框,旋转操作会打乱标注,建议清除标注框后旋转。")
        else:
            self.msgBox.warning(self, "Warn", "\n The picture already has a label box, "
                                              "and rotation will disrupt the label. "
                                              "It is recommended to clear the label box and rotate it.")

    def rotateImgAction(self, k=1, _value=False):

        filename = self.mImgList[self.currIndex]

        if os.path.exists(filename):
            if self.itemsToShapesbox:
                self.rotateImgWarn()
            else:
                self.saveFile()
                self.dirty = False
                self.rotateImg(filename=filename, k=k, _value=True)
        else:
            self.rotateImgWarn()
            self.actions.rotateRight.setEnabled(False)
            self.actions.rotateLeft.setEnabled(False)

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    def setCreateMode(self):
        assert self.advanced()
        self.toggleDrawMode(False)

    def setEditMode(self):
        assert self.advanced()
        self.toggleDrawMode(True)
        self.labelSelectionChanged()

    def updateFileMenu(self):
        currFilePath = self.filePath

        def exists(filename):
            return os.path.exists(filename)

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def editLabel(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        text = self.labelDialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            # item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    # =================== detection box related functions ===================
    def boxItemChanged(self, item):
        shape = self.itemsToShapesbox[item]

        box = ast.literal_eval(item.text())
        # print('shape in labelItemChanged is',shape.points)
        if box != [(int(p.x()), int(p.y())) for p in shape.points]:
            # shape.points = box
            shape.points = [QPointF(p[0], p[1]) for p in box]

            # QPointF(x,y)
            # shape.line_color = generateColorByText(shape.label)
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, True)  # item.checkState() == Qt.Checked

    def editBox(self):  # ADD
        if not self.canvas.editing():
            return
        item = self.currentBox()
        if not item:
            return
        text = self.labelDialog.popUp(item.text())

        imageSize = str(self.image.size())
        width, height = self.image.width(), self.image.height()
        if text:
            try:
                text_list = eval(text)
            except:
                msg_box = QMessageBox(QMessageBox.Warning, 'Warning', 'Please enter the correct format')
                msg_box.exec_()
                return
            if len(text_list) < 4:
                msg_box = QMessageBox(QMessageBox.Warning, 'Warning', 'Please enter the coordinates of 4 points')
                msg_box.exec_()
                return
            for box in text_list:
                if box[0] > width or box[0] < 0 or box[1] > height or box[1] < 0:
                    msg_box = QMessageBox(QMessageBox.Warning, 'Warning', 'Out of picture size')
                    msg_box.exec_()
                    return

            item.setText(text)
            # item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    def updateBoxlist(self):
        self.canvas.selectedShapes_hShape = []
        if self.canvas.hShape != None:
            self.canvas.selectedShapes_hShape = self.canvas.selectedShapes + [self.canvas.hShape]
        else:
            self.canvas.selectedShapes_hShape = self.canvas.selectedShapes
        for shape in self.canvas.selectedShapes_hShape:
            item = self.shapesToItemsbox[shape]  # listitem
            text = [(int(p.x()), int(p.y())) for p in shape.points]
            item.setText(str(text))
        self.actions.undo.setEnabled(True)
        self.setDirty()

    def indexTo5Files(self, currIndex):
        if currIndex < 2:
            return self.mImgList[:5]
        elif currIndex > len(self.mImgList) - 3:
            return self.mImgList[-5:]
        else:
            return self.mImgList[currIndex - 2: currIndex + 3]

    # Tzutalin 20160906 : Add file list and dock to move faster
    def fileitemDoubleClicked(self, item=None):
        self.currIndex = self.mImgList.index(ustr(os.path.join(os.path.abspath(self.dirname), item.text())))
        filename = self.mImgList[self.currIndex]
        if filename:
            self.mImgList5 = self.indexTo5Files(self.currIndex)
            # self.additems5(None)
            self.loadFile(filename)

    def iconitemDoubleClicked(self, item=None):
        self.currIndex = self.mImgList.index(ustr(os.path.join(item.toolTip())))
        filename = self.mImgList[self.currIndex]
        if filename:
            self.mImgList5 = self.indexTo5Files(self.currIndex)
            # self.additems5(None)
            self.loadFile(filename)

    def CanvasSizeChange(self):
        if len(self.mImgList) > 0 and self.imageSlider.hasFocus():
            self.zoomWidget.setValue(self.imageSlider.value())

    def shapeSelectionChanged(self, selected_shapes):
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            self.shapesToItems[shape].setSelected(True)
            self.shapesToItemsbox[shape].setSelected(True)

        self.labelList.scrollToItem(self.currentItem())  # QAbstractItemView.EnsureVisible
        self.BoxList.scrollToItem(self.currentBox())

        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.singleRere.setEnabled(n_selected)
        self.actions.delete.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected == 1)
        self.actions.lock.setEnabled(n_selected)

    def addLabel(self, shape):
        shape.paintLabel = self.displayLabelOption.isChecked()
        item = HashableQListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked) if shape.difficult else item.setCheckState(Qt.Checked)
        # Checked means difficult is False
        # item.setBackground(generateColorByText(shape.label))
        self.itemsToShapes[item] = shape
        self.shapesToItems[shape] = item
        self.labelList.addItem(item)
        # print('item in add label is ',[(p.x(), p.y()) for p in shape.points], shape.label)

        # ADD for box
        item = HashableQListWidgetItem(str([(int(p.x()), int(p.y())) for p in shape.points]))
        self.itemsToShapesbox[item] = shape
        self.shapesToItemsbox[shape] = item
        self.BoxList.addItem(item)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)
        self.updateComboBox()

        # update show counting
        self.BoxListDock.setWindowTitle(self.BoxListDockName + f" ({self.BoxList.count()})")
        self.labelListDock.setWindowTitle(self.labelListDockName + f" ({self.labelList.count()})")

    def remLabels(self, shapes):
        if shapes is None:
            # print('rm empty label')
            return
        for shape in shapes:
            item = self.shapesToItems[shape]
            self.labelList.takeItem(self.labelList.row(item))
            del self.shapesToItems[shape]
            del self.itemsToShapes[item]
            self.updateComboBox()

            # ADD:
            item = self.shapesToItemsbox[shape]
            self.BoxList.takeItem(self.BoxList.row(item))
            del self.shapesToItemsbox[shape]
            del self.itemsToShapesbox[item]
            self.updateComboBox()

    def loadLabels(self, shapes):
        s = []
        for label, points, line_color, fill_color, difficult in shapes:
            shape = Shape(label=label, line_color=line_color)
            for x, y in points:

                # Ensure the labels are within the bounds of the image. If not, fix them.
                x, y, snapped = self.canvas.snapPointToCanvas(x, y)
                if snapped:
                    self.setDirty()

                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            # shape.locked = False
            shape.close()
            s.append(shape)

            # if line_color:
            #     shape.line_color = QColor(*line_color)
            # else:
            #     shape.line_color = generateColorByText(label)
            #
            # if fill_color:
            #     shape.fill_color = QColor(*fill_color)
            # else:
            #     shape.fill_color = generateColorByText(label)

            self.addLabel(shape)

        self.updateComboBox()
        self.canvas.loadShapes(s)

    def singleLabel(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapesToItems[shape]
        item.setText(shape.label)
        self.updateComboBox()

        # ADD:
        item = self.shapesToItemsbox[shape]
        item.setText(str([(int(p.x()), int(p.y())) for p in shape.points]))
        self.updateComboBox()

    def updateComboBox(self):
        # Get the unique labels and add them to the Combobox.
        itemsTextList = [str(self.labelList.item(i).text()) for i in range(self.labelList.count())]

        uniqueTextList = list(set(itemsTextList))
        # Add a null row for showing all the labels
        uniqueTextList.append("")
        uniqueTextList.sort()

        # self.comboBox.update_items(uniqueTextList)

    def saveLabels(self, annotationFilePath, mode='Auto'):
        # Mode is Auto means that labels will be loaded from self.result_dic totally, which is the output of ocr model
        annotationFilePath = ustr(annotationFilePath)

        def format_shape(s):
            # print('s in saveLabels is ',s)
            return dict(label=s.label,  # str
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        points=[(int(p.x()), int(p.y())) for p in s.points],  # QPonitF
                        # add chris
                        difficult=s.difficult)  # bool

        shapes = [] if mode == 'Auto' else \
            [format_shape(shape) for shape in self.canvas.shapes if shape.line_color != DEFAULT_LOCK_COLOR]
        # Can add differrent annotation formats here
        for box in self.result_dic:
            trans_dic = {"label": box[1][0], "points": box[0], 'difficult': False}
            if trans_dic["label"] == "" and mode == 'Auto':
                continue
            shapes.append(trans_dic)

        try:
            trans_dic = []
            for box in shapes:
                trans_dic.append(
                    {"transcription": box['label'], "points": box['points'], 'difficult': box['difficult']})
            self.PPlabel[annotationFilePath] = trans_dic
            if mode == 'Auto':
                self.Cachelabel[annotationFilePath] = trans_dic

            # else:
            #     self.labelFile.save(annotationFilePath, shapes, self.filePath, self.imageData,
            #                         self.lineColor.getRgb(), self.fillColor.getRgb())
            # print('Image:{0} -> Annotation:{1}'.format(self.filePath, annotationFilePath))
            return True
        except:
            self.errorMessage(u'Error saving label data', u'Error saving label data')
            return False

    def copySelectedShape(self):
        for shape in self.canvas.copySelectedShape():
            self.addLabel(shape)
        # fix copy and delete
        # self.shapeSelectionChanged(True)

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                selected_shapes.append(self.itemsToShapes[item])
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def keyListSelectionChanged(self):
        pass

    def keyListItemChanged(self):
        pass

    def boxSelectionChanged(self):
        if self._noSelectionSlot:
            # self.BoxList.scrollToItem(self.currentBox(), QAbstractItemView.PositionAtCenter)
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.BoxList.selectedItems():
                selected_shapes.append(self.itemsToShapesbox[item])
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        shape = self.itemsToShapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            # shape.line_color = generateColorByText(shape.label)
            self.setDirty()
        elif not ((item.checkState() == Qt.Unchecked) ^ (not shape.difficult)):
            shape.difficult = True if item.checkState() == Qt.Unchecked else False
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, True)  # item.checkState() == Qt.Checked
            # self.actions.save.setEnabled(True)

    # Callback functions:
    def newShape(self, value=True):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        if len(self.labelHist) > 0:
            self.labelDialog = LabelDialog(
                parent=self, listItem=self.labelHist)

        if value:
            text = self.labelDialog.popUp(text=self.prevLabelText)
            self.lastLabel = text
        else:
            text = self.prevLabelText

        if text is not None:
            self.prevLabelText = self.stringBundle.getString('tempLabel')
            # generate_color = generateColorByText(text)
            shape = self.canvas.setLastLabel(text, None, None)  # generate_color, generate_color
            self.addLabel(shape)
            if self.beginner():  # Switch to edit mode.
                self.canvas.setEditing(True)
                self.actions.create.setEnabled(True)
                self.actions.undoLastPoint.setEnabled(False)
                self.actions.undo.setEnabled(True)
            else:
                self.actions.editMode.setEnabled(True)
            self.setDirty()

        else:
            # self.canvas.undoLastLine()
            self.canvas.resetAllLines()

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)
        self.imageSlider.setValue(self.zoomWidget.value() + increment)  # set zoom slider value

    def zoomRequest(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scrollArea.width()
        h = self.scrollArea.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.itemsToShapes.items():
            self.canvas.setShapeVisible(shape, value)

    def loadFile(self, filePath=None):
        """Load the specified file, or the last opened file if None."""
        if self.dirty:
            self.mayContinue()
        self.resetState()
        self.canvas.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get(SETTING_FILENAME)

        # Make sure that filePath is a regular python string, rather than QString
        filePath = ustr(filePath)
        # Fix bug: An index error after select a directory when open a new file.
        unicodeFilePath = ustr(filePath)
        # unicodeFilePath = os.path.abspath(unicodeFilePath)
        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item

        if unicodeFilePath and self.fileListWidget.count() > 0:
            if unicodeFilePath in self.mImgList:
                index = self.mImgList.index(unicodeFilePath)
                fileWidgetItem = self.fileListWidget.item(index)
                print('unicodeFilePath is', unicodeFilePath)
                fileWidgetItem.setSelected(True)
                self.iconlist.clear()
                self.additems5(None)

                for i in range(5):
                    item_tooltip = self.iconlist.item(i).toolTip()
                    # print(i,"---",item_tooltip)
                    if item_tooltip == ustr(filePath):
                        titem = self.iconlist.item(i)
                        titem.setSelected(True)
                        self.iconlist.scrollToItem(titem)
                        break
            else:
                self.fileListWidget.clear()
                self.mImgList.clear()
                self.iconlist.clear()

        # if unicodeFilePath and self.iconList.count() > 0:
        #     if unicodeFilePath in self.mImgList:

        if unicodeFilePath and os.path.exists(unicodeFilePath):
            self.canvas.verified = False
            cvimg = cv2.imdecode(np.fromfile(unicodeFilePath, dtype=np.uint8), 1)
            height, width, depth = cvimg.shape
            cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
            image = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)

            if image.isNull():
                self.errorMessage(u'Error opening file',
                                  u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
                return False
            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            self.image = image
            self.filePath = unicodeFilePath
            self.canvas.loadPixmap(QPixmap.fromImage(image))

            if self.validFilestate(filePath) is True:
                self.setClean()
            else:
                self.dirty = False
                self.actions.save.setEnabled(True)
            if len(self.canvas.lockedShapes) != 0:
                self.actions.save.setEnabled(True)
                self.setDirty()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self.filePath)
            self.toggleActions(True)

            self.showBoundingBoxFromPPlabel(filePath)

            self.setWindowTitle(__appname__ + ' ' + filePath)

            # Default : select last item if there is at least one item
            if self.labelList.count():
                self.labelList.setCurrentItem(self.labelList.item(self.labelList.count() - 1))
                self.labelList.item(self.labelList.count() - 1).setSelected(True)

            # show file list image count
            select_indexes = self.fileListWidget.selectedIndexes()
            if len(select_indexes) > 0:
                self.fileDock.setWindowTitle(self.fileListName + f" ({select_indexes[0].row() + 1}"
                                                                  f"/{self.fileListWidget.count()})")
            # update show counting
            self.BoxListDock.setWindowTitle(self.BoxListDockName + f" ({self.BoxList.count()})")
            self.labelListDock.setWindowTitle(self.labelListDockName + f" ({self.labelList.count()})")

            self.canvas.setFocus(True)
            return True
        return False

    def showBoundingBoxFromPPlabel(self, filePath):
        width, height = self.image.width(), self.image.height()
        imgidx = self.getImglabelidx(filePath)
        shapes = []
        # box['ratio'] of the shapes saved in lockedShapes contains the ratio of the
        # four corner coordinates of the shapes to the height and width of the image
        for box in self.canvas.lockedShapes:
            if self.canvas.isInTheSameImage:
                shapes.append((box['transcription'], [[s[0] * width, s[1] * height] for s in box['ratio']],
                               DEFAULT_LOCK_COLOR, None, box['difficult']))
            else:
                shapes.append(('锁定框：待检测', [[s[0] * width, s[1] * height] for s in box['ratio']],
                               DEFAULT_LOCK_COLOR, None, box['difficult']))
        if imgidx in self.PPlabel.keys():
            for box in self.PPlabel[imgidx]:
                shapes.append((box['transcription'], box['points'], None, None, box['difficult']))

        self.loadLabels(shapes)
        self.canvas.verified = False

    def validFilestate(self, filePath):
        if filePath not in self.fileStatedict.keys():
            return None
        elif self.fileStatedict[filePath] == 1:
            return True
        else:
            return False

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull() \
                and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e - 110
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        else:
            settings = self.settings
            # If it loads images from dir, don't load it at the begining
            if self.dirname is None:
                settings[SETTING_FILENAME] = self.filePath if self.filePath else ''
            else:
                settings[SETTING_FILENAME] = ''

            settings[SETTING_WIN_SIZE] = self.size()
            settings[SETTING_WIN_POSE] = self.pos()
            settings[SETTING_WIN_STATE] = self.saveState()
            settings[SETTING_LINE_COLOR] = self.lineColor
            settings[SETTING_FILL_COLOR] = self.fillColor
            settings[SETTING_RECENT_FILES] = self.recentFiles
            settings[SETTING_ADVANCE_MODE] = not self._beginner
            if self.defaultSaveDir and os.path.exists(self.defaultSaveDir):
                settings[SETTING_SAVE_DIR] = ustr(self.defaultSaveDir)
            else:
                settings[SETTING_SAVE_DIR] = ''

            if self.lastOpenDir and os.path.exists(self.lastOpenDir):
                settings[SETTING_LAST_OPEN_DIR] = self.lastOpenDir
            else:
                settings[SETTING_LAST_OPEN_DIR] = ''

            settings[SETTING_PAINT_LABEL] = self.displayLabelOption.isChecked()
            settings[SETTING_DRAW_SQUARE] = self.drawSquaresOption.isChecked()
            settings.save()
            try:
                self.saveLabelFile()
            except:
                pass

    def loadRecent(self, filename):
        if self.mayContinue():
            print(filename, "======")
            self.loadFile(filename)

    def scanAllImages(self, folderPath):
        extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        images = []

        for file in os.listdir(folderPath):
            if file.lower().endswith(tuple(extensions)):
                relativePath = os.path.join(folderPath, file)
                path = ustr(os.path.abspath(relativePath))
                images.append(path)
        natural_sort(images, key=lambda x: x.lower())
        return images

    def openDirDialog(self, _value=False, dirpath=None, silent=False):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else '.'
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filePath) if self.filePath else '.'
        if silent != True:
            targetDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                                  '%s - Open Directory' % __appname__,
                                                                  defaultOpenDirPath,
                                                                  QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        else:
            targetDirPath = ustr(defaultOpenDirPath)
        self.lastOpenDir = targetDirPath
        self.importDirImages(targetDirPath)

    def openDatasetDirDialog(self):
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            if platform.system() == 'Windows':
                os.startfile(self.lastOpenDir)
            else:
                os.system('open ' + os.path.normpath(self.lastOpenDir))
            defaultOpenDirPath = self.lastOpenDir

        else:
            if self.lang == 'ch':
                self.msgBox.warning(self, "提示", "\n 原文件夹已不存在,请从新选择数据集路径!")
            else:
                self.msgBox.warning(self, "Warn",
                                    "\n The original folder no longer exists, please choose the data set path again!")

            self.actions.open_dataset_dir.setEnabled(False)
            defaultOpenDirPath = os.path.dirname(self.filePath) if self.filePath else '.'

    def importDirImages(self, dirpath, isDelete=False):
        if not self.mayContinue() or not dirpath:
            return
        if self.defaultSaveDir and self.defaultSaveDir != dirpath:
            self.saveLabelFile()

        if not isDelete:
            self.loadFilestate(dirpath)
            self.PPlabelpath = dirpath + '/Label.txt'
            self.PPlabel = self.loadLabelFile(self.PPlabelpath)
            self.Cachelabelpath = dirpath + '/Cache.cach'
            self.Cachelabel = self.loadLabelFile(self.Cachelabelpath)
            if self.Cachelabel:
                self.PPlabel = dict(self.Cachelabel, **self.PPlabel)
        self.lastOpenDir = dirpath
        self.dirname = dirpath

        self.defaultSaveDir = dirpath
        self.statusBar().showMessage('%s started. Annotation will be saved to %s' %
                                     (__appname__, self.defaultSaveDir))
        self.statusBar().show()

        self.filePath = None
        self.fileListWidget.clear()
        self.mImgList = self.scanAllImages(dirpath)
        self.mImgList5 = self.mImgList[:5]
        self.openNextImg()
        doneicon = newIcon('done')
        closeicon = newIcon('close')
        for imgPath in self.mImgList:
            filename = os.path.basename(imgPath)
            if self.validFilestate(imgPath) is True:
                item = QListWidgetItem(doneicon, filename)
            else:
                item = QListWidgetItem(closeicon, filename)
            self.fileListWidget.addItem(item)

        print('DirPath in importDirImages is', dirpath)
        self.iconlist.clear()
        self.additems5(dirpath)
        self.changeFileFolder = True
        self.haveAutoReced = False
        self.AutoRecognition.setEnabled(True)
        self.reRecogButton.setEnabled(True)
        self.actions.AutoRec.setEnabled(True)
        self.actions.reRec.setEnabled(True)
        self.actions.open_dataset_dir.setEnabled(True)
        self.actions.rotateLeft.setEnabled(True)
        self.actions.rotateRight.setEnabled(True)

        self.fileListWidget.setCurrentRow(0)  # set list index to first
        self.fileDock.setWindowTitle(self.fileListName + f" (1/{self.fileListWidget.count()})")  # show image count

    def openPrevImg(self, _value=False):
        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            return

        currIndex = self.mImgList.index(self.filePath)
        self.mImgList5 = self.mImgList[:5]
        if currIndex - 1 >= 0:
            filename = self.mImgList[currIndex - 1]
            self.mImgList5 = self.indexTo5Files(currIndex - 1)
            if filename:
                self.loadFile(filename)

    def openNextImg(self, _value=False):
        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        filename = None
        if self.filePath is None:
            filename = self.mImgList[0]
            self.mImgList5 = self.mImgList[:5]
        else:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex + 1 < len(self.mImgList):
                filename = self.mImgList[currIndex + 1]
                self.mImgList5 = self.indexTo5Files(currIndex + 1)
            else:
                self.mImgList5 = self.indexTo5Files(currIndex)
        if filename:
            print('file name in openNext is ', filename)
            self.loadFile(filename)

    def updateFileListIcon(self, filename):
        pass

    def saveFile(self, _value=False, mode='Manual'):
        # Manual mode is used for users click "Save" manually,which will change the state of the image
        if self.filePath:
            imgidx = self.getImglabelidx(self.filePath)
            self._saveFile(imgidx, mode=mode)

    def saveLockedShapes(self):
        self.canvas.lockedShapes = []
        self.canvas.selectedShapes = []
        for s in self.canvas.shapes:
            if s.line_color == DEFAULT_LOCK_COLOR:
                self.canvas.selectedShapes.append(s)
        self.lockSelectedShape()
        for s in self.canvas.shapes:
            if s.line_color == DEFAULT_LOCK_COLOR:
                self.canvas.selectedShapes.remove(s)
                self.canvas.shapes.remove(s)

    def _saveFile(self, annotationFilePath, mode='Manual'):
        if len(self.canvas.lockedShapes) != 0:
            self.saveLockedShapes()

        if mode == 'Manual':
            self.result_dic_locked = []
            img = cv2.imread(self.filePath)
            width, height = self.image.width(), self.image.height()
            for shape in self.canvas.lockedShapes:
                box = [[int(p[0] * width), int(p[1] * height)] for p in shape['ratio']]
                assert len(box) == 4
                result = [(shape['transcription'], 1)]
                result.insert(0, box)
                self.result_dic_locked.append(result)
            self.result_dic += self.result_dic_locked
            self.result_dic_locked = []
            if annotationFilePath and self.saveLabels(annotationFilePath, mode=mode):
                self.setClean()
                self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
                self.statusBar().show()
                currIndex = self.mImgList.index(self.filePath)
                item = self.fileListWidget.item(currIndex)
                item.setIcon(newIcon('done'))

                self.fileStatedict[self.filePath] = 1
                if len(self.fileStatedict) % self.autoSaveNum == 0:
                    self.saveFilestate()
                    self.savePPlabel(mode='Auto')

                self.fileListWidget.insertItem(int(currIndex), item)
                if not self.canvas.isInTheSameImage:
                    self.openNextImg()
                self.actions.saveRec.setEnabled(True)
                self.actions.saveLabel.setEnabled(True)

        elif mode == 'Auto':
            if annotationFilePath and self.saveLabels(annotationFilePath, mode=mode):
                self.setClean()
                self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
                self.statusBar().show()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def deleteImg(self):
        deletePath = self.filePath
        if deletePath is not None:
            deleteInfo = self.deleteImgDialog()
            if deleteInfo == QMessageBox.Yes:
                if platform.system() == 'Windows':
                    from win32com.shell import shell, shellcon
                    shell.SHFileOperation((0, shellcon.FO_DELETE, deletePath, None,
                                           shellcon.FOF_SILENT | shellcon.FOF_ALLOWUNDO | shellcon.FOF_NOCONFIRMATION,
                                           None, None))
                    # linux
                elif platform.system() == 'Linux':
                    cmd = 'trash ' + deletePath
                    os.system(cmd)
                    # macOS
                elif platform.system() == 'Darwin':
                    import subprocess
                    absPath = os.path.abspath(deletePath).replace('\\', '\\\\').replace('"', '\\"')
                    cmd = ['osascript', '-e',
                           'tell app "Finder" to move {the POSIX file "' + absPath + '"} to trash']
                    print(cmd)
                    subprocess.call(cmd, stdout=open(os.devnull, 'w'))

                if self.filePath in self.fileStatedict.keys():
                    self.fileStatedict.pop(self.filePath)
                imgidx = self.getImglabelidx(self.filePath)
                if imgidx in self.PPlabel.keys():
                    self.PPlabel.pop(imgidx)
                self.openNextImg()
                self.importDirImages(self.lastOpenDir, isDelete=True)

    def deleteImgDialog(self):
        yes, cancel = QMessageBox.Yes, QMessageBox.Cancel
        msg = u'The image will be deleted to the recycle bin'
        return QMessageBox.warning(self, u'Attention', msg, yes | cancel)

    def resetAll(self):
        self.settings.reset()
        self.close()
        proc = QProcess()
        proc.startDetached(os.path.abspath(__file__))

    def mayContinue(self):  #
        if not self.dirty:
            return True
        else:
            discardChanges = self.discardChangesDialog()
            if discardChanges == QMessageBox.No:
                return True
            elif discardChanges == QMessageBox.Yes:
                self.canvas.isInTheSameImage = True
                self.saveFile()
                self.canvas.isInTheSameImage = False
                return True
            else:
                return False

    def discardChangesDialog(self):
        yes, no, cancel = QMessageBox.Yes, QMessageBox.No, QMessageBox.Cancel
        if self.lang == 'ch':
            msg = u'您有未保存的变更, 您想保存再继续吗?\n点击 "No" 丢弃所有未保存的变更.'
        else:
            msg = u'You have unsaved changes, would you like to save them and proceed?\nClick "No" to undo all changes.'
        return QMessageBox.warning(self, u'Attention', msg, yes | no | cancel)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def chooseColor1(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            Shape.line_color = color
            self.canvas.setDrawingColor(color)
            self.canvas.update()
            self.setDirty()

    def deleteSelectedShape(self):
        self.remLabels(self.canvas.deleteSelected())
        self.actions.undo.setEnabled(True)
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)
        self.BoxListDock.setWindowTitle(self.BoxListDockName + f" ({self.BoxList.count()})")
        self.labelListDock.setWindowTitle(self.labelListDockName + f" ({self.labelList.count()})")

    def chshapeLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            for shape in self.canvas.selectedShapes: shape.line_color = color
            self.canvas.update()
            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            for shape in self.canvas.selectedShapes: shape.fill_color = color
            self.canvas.update()
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def loadPredefinedClasses(self, predefClassesFile):
        if os.path.exists(predefClassesFile) is True:
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.labelHist is None:
                        self.labelHist = [line]
                    else:
                        self.labelHist.append(line)

    def togglePaintLabelsOption(self):
        for shape in self.canvas.shapes:
            shape.paintLabel = self.displayLabelOption.isChecked()

    def toogleDrawSquare(self):
        self.canvas.setDrawingShapeToSquare(self.drawSquaresOption.isChecked())

    def additems(self, dirpath):
        for file in self.mImgList:
            pix = QPixmap(file)
            _, filename = os.path.split(file)
            filename, _ = os.path.splitext(filename)
            item = QListWidgetItem(QIcon(pix.scaled(100, 100, Qt.IgnoreAspectRatio, Qt.FastTransformation)),
                                   filename[:10])
            item.setToolTip(file)
            self.iconlist.addItem(item)

    def additems5(self, dirpath):
        for file in self.mImgList5:
            pix = QPixmap(file)
            _, filename = os.path.split(file)
            filename, _ = os.path.splitext(filename)
            pfilename = filename[:10]
            if len(pfilename) < 10:
                lentoken = 12 - len(pfilename)
                prelen = lentoken // 2
                bfilename = prelen * " " + pfilename + (lentoken - prelen) * " "
            # item = QListWidgetItem(QIcon(pix.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)),filename[:10])
            item = QListWidgetItem(QIcon(pix.scaled(100, 100, Qt.IgnoreAspectRatio, Qt.FastTransformation)), pfilename)
            # item.setForeground(QBrush(Qt.white))
            item.setToolTip(file)
            self.iconlist.addItem(item)
        owidth = 0
        for index in range(len(self.mImgList5)):
            item = self.iconlist.item(index)
            itemwidget = self.iconlist.visualItemRect(item)
            owidth += itemwidget.width()
        self.iconlist.setMinimumWidth(owidth + 50)

    def getImglabelidx(self, filePath):
        if platform.system() == 'Windows':
            spliter = '\\'
        else:
            spliter = '/'
        filepathsplit = filePath.split(spliter)[-2:]
        return filepathsplit[0] + '/' + filepathsplit[1]

    def autoRecognition(self):
        assert self.mImgList is not None
        print('Using model from ', self.model)

        uncheckedList = [i for i in self.mImgList if i not in self.fileStatedict.keys()]
        self.autoDialog = AutoDialog(parent=self, ocr=self.ocr, mImgList=uncheckedList, lenbar=len(uncheckedList))
        self.autoDialog.popUp()
        self.currIndex = len(self.mImgList) - 1
        self.loadFile(self.filePath)  # ADD
        self.haveAutoReced = True
        self.AutoRecognition.setEnabled(False)
        self.actions.AutoRec.setEnabled(False)
        self.setDirty()
        self.saveCacheLabel()

    def reRecognition(self):
        img = cv2.imread(self.filePath)
        # org_box = [dic['points'] for dic in self.PPlabel[self.getImglabelidx(self.filePath)]]
        if self.canvas.shapes:
            self.result_dic = []
            self.result_dic_locked = []  # result_dic_locked stores the ocr result of self.canvas.lockedShapes
            rec_flag = 0
            for shape in self.canvas.shapes:
                box = [[int(p.x()), int(p.y())] for p in shape.points]
                assert len(box) == 4
                img_crop = get_rotate_crop_image(img, np.array(box, np.float32))
                if img_crop is None:
                    msg = 'Can not recognise the detection box in ' + self.filePath + '. Please change manually'
                    QMessageBox.information(self, "Information", msg)
                    return
                result = self.ocr.ocr(img_crop, cls=True, det=False)
                if result[0][0] != '':
                    if shape.line_color == DEFAULT_LOCK_COLOR:
                        shape.label = result[0][0]
                        result.insert(0, box)
                        self.result_dic_locked.append(result)
                    else:
                        result.insert(0, box)
                        self.result_dic.append(result)
                else:
                    print('Can not recognise the box')
                    if shape.line_color == DEFAULT_LOCK_COLOR:
                        shape.label = result[0][0]
                        self.result_dic_locked.append([box, (self.noLabelText, 0)])
                    else:
                        self.result_dic.append([box, (self.noLabelText, 0)])
                try:
                    if self.noLabelText == shape.label or result[1][0] == shape.label:
                        print('label no change')
                    else:
                        rec_flag += 1
                except IndexError as e:
                    print('Can not recognise the box')
            if (len(self.result_dic) > 0 and rec_flag > 0) or self.canvas.lockedShapes:
                self.canvas.isInTheSameImage = True
                self.saveFile(mode='Auto')
                self.loadFile(self.filePath)
                self.canvas.isInTheSameImage = False
                self.setDirty()
            elif len(self.result_dic) == len(self.canvas.shapes) and rec_flag == 0:
                if self.lang == 'ch':
                    QMessageBox.information(self, "Information", "识别结果保持一致！")
                else:
                    QMessageBox.information(self, "Information", "The recognition result remains unchanged!")
            else:
                print('Can not recgonise in ', self.filePath)
        else:
            QMessageBox.information(self, "Information", "Draw a box!")

    def singleRerecognition(self):
        img = cv2.imread(self.filePath)
        for shape in self.canvas.selectedShapes:
            box = [[int(p.x()), int(p.y())] for p in shape.points]
            assert len(box) == 4
            img_crop = get_rotate_crop_image(img, np.array(box, np.float32))
            if img_crop is None:
                msg = 'Can not recognise the detection box in ' + self.filePath + '. Please change manually'
                QMessageBox.information(self, "Information", msg)
                return
            result = self.ocr.ocr(img_crop, cls=True, det=False)
            if result[0][0] != '':
                result.insert(0, box)
                print('result in reRec is ', result)
                if result[1][0] == shape.label:
                    print('label no change')
                else:
                    shape.label = result[1][0]
            else:
                print('Can not recognise the box')
                if self.noLabelText == shape.label:
                    print('label no change')
                else:
                    shape.label = self.noLabelText
            self.singleLabel(shape)
            self.setDirty()

    def autolcm(self):
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        self.panel = QLabel()
        self.panel.setText(self.stringBundle.getString('choseModelLg'))
        self.panel.setAlignment(Qt.AlignLeft)
        self.comboBox = QComboBox()
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItems(['Chinese & English', 'English', 'French', 'German', 'Korean', 'Japanese'])
        vbox.addWidget(self.panel)
        vbox.addWidget(self.comboBox)
        self.dialog = QDialog()
        self.dialog.resize(300, 100)
        self.okBtn = QPushButton(self.stringBundle.getString('ok'))
        self.cancelBtn = QPushButton(self.stringBundle.getString('cancel'))

        self.okBtn.clicked.connect(self.modelChoose)
        self.cancelBtn.clicked.connect(self.cancel)
        self.dialog.setWindowTitle(self.stringBundle.getString('choseModelLg'))

        hbox.addWidget(self.okBtn)
        hbox.addWidget(self.cancelBtn)

        vbox.addWidget(self.panel)
        vbox.addLayout(hbox)
        self.dialog.setLayout(vbox)
        self.dialog.setWindowModality(Qt.ApplicationModal)
        self.dialog.exec_()
        if self.filePath:
            self.AutoRecognition.setEnabled(True)
            self.actions.AutoRec.setEnabled(True)

    def modelChoose(self):
        print(self.comboBox.currentText())
        lg_idx = {'Chinese & English': 'ch', 'English': 'en', 'French': 'french', 'German': 'german',
                  'Korean': 'korean', 'Japanese': 'japan'}
        del self.ocr
        self.ocr = PaddleOCR(use_pdserving=False, use_angle_cls=True, det=True, cls=True, use_gpu=False,
                             lang=lg_idx[self.comboBox.currentText()])
        self.dialog.close()

    def cancel(self):
        self.dialog.close()

    def loadFilestate(self, saveDir):
        self.fileStatepath = saveDir + '/fileState.txt'
        self.fileStatedict = {}
        if not os.path.exists(self.fileStatepath):
            f = open(self.fileStatepath, 'w', encoding='utf-8')
        else:
            with open(self.fileStatepath, 'r', encoding='utf-8') as f:
                states = f.readlines()
                for each in states:
                    file, state = each.split('\t')
                    self.fileStatedict[file] = 1
                self.actions.saveLabel.setEnabled(True)
                self.actions.saveRec.setEnabled(True)

    def saveFilestate(self):
        with open(self.fileStatepath, 'w', encoding='utf-8') as f:
            for key in self.fileStatedict:
                f.write(key + '\t')
                f.write(str(self.fileStatedict[key]) + '\n')

    def loadLabelFile(self, labelpath):
        labeldict = {}
        if not os.path.exists(labelpath):
            f = open(labelpath, 'w', encoding='utf-8')

        else:
            with open(labelpath, 'r', encoding='utf-8') as f:
                data = f.readlines()
                for each in data:
                    file, label = each.split('\t')
                    if label:
                        label = label.replace('false', 'False')
                        label = label.replace('true', 'True')
                        labeldict[file] = eval(label)
                    else:
                        labeldict[file] = []
        return labeldict

    def savePPlabel(self, mode='Manual'):
        savedfile = [self.getImglabelidx(i) for i in self.fileStatedict.keys()]
        with open(self.PPlabelpath, 'w', encoding='utf-8') as f:
            for key in self.PPlabel:
                if key in savedfile and self.PPlabel[key] != []:
                    f.write(key + '\t')
                    f.write(json.dumps(self.PPlabel[key], ensure_ascii=False) + '\n')

        if mode == 'Manual':
            if self.lang == 'ch':
                msg = '已将检查过的图片标签保存在 ' + self.PPlabelpath + " 文件中"
            else:
                msg = 'Images that have been checked are saved in ' + self.PPlabelpath
            QMessageBox.information(self, "Information", msg)

    def saveCacheLabel(self):
        with open(self.Cachelabelpath, 'w', encoding='utf-8') as f:
            for key in self.Cachelabel:
                f.write(key + '\t')
                f.write(json.dumps(self.Cachelabel[key], ensure_ascii=False) + '\n')

    def saveLabelFile(self):
        self.saveFilestate()
        self.savePPlabel()

    def saveRecResult(self):
        if {} in [self.PPlabelpath, self.PPlabel, self.fileStatedict]:
            QMessageBox.information(self, "Information", "Check the image first")
            return

        rec_gt_dir = os.path.dirname(self.PPlabelpath) + '/rec_gt.txt'
        crop_img_dir = os.path.dirname(self.PPlabelpath) + '/crop_img/'
        ques_img = []
        if not os.path.exists(crop_img_dir):
            os.mkdir(crop_img_dir)

        with open(rec_gt_dir, 'w', encoding='utf-8') as f:
            for key in self.fileStatedict:
                idx = self.getImglabelidx(key)
                try:
                    img = cv2.imread(key)
                    for i, label in enumerate(self.PPlabel[idx]):
                        if label['difficult']: continue
                        img_crop = get_rotate_crop_image(img, np.array(label['points'], np.float32))
                        img_name = os.path.splitext(os.path.basename(idx))[0] + '_crop_' + str(i) + '.jpg'
                        cv2.imwrite(crop_img_dir + img_name, img_crop)
                        f.write('crop_img/' + img_name + '\t')
                        f.write(label['transcription'] + '\n')
                except Exception as e:
                    ques_img.append(key)
                    print("Can not read image ", e)
        if ques_img:
            QMessageBox.information(self,
                                    "Information",
                                    "The following images can not be saved, please check the image path and labels.\n"
                                    + "".join(str(i) + '\n' for i in ques_img))
        QMessageBox.information(self, "Information", "Cropped images have been saved in " + str(crop_img_dir))

    def speedChoose(self):
        if self.labelDialogOption.isChecked():
            self.canvas.newShape.disconnect()
            self.canvas.newShape.connect(partial(self.newShape, True))

        else:
            self.canvas.newShape.disconnect()
            self.canvas.newShape.connect(partial(self.newShape, False))

    def autoSaveFunc(self):
        if self.autoSaveOption.isChecked():
            self.autoSaveNum = 1  # Real auto_Save
            try:
                self.saveLabelFile()
            except:
                pass
            print('The program will automatically save once after confirming an image')
        else:
            self.autoSaveNum = 5  # Used for backup
            print('The program will automatically save once after confirming 5 images (default)')

    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.labelList.clear()
        self.BoxList.clear()
        self.loadShapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)
        print("loadShapes")  # 1

    def lockSelectedShape(self):
        """lock the selected shapes.

        Add self.selectedShapes to lock self.canvas.lockedShapes, 
        which holds the ratio of the four coordinates of the locked shapes
        to the width and height of the image
        """
        width, height = self.image.width(), self.image.height()

        def format_shape(s):
            return dict(label=s.label,  # str
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        ratio=[[int(p.x()) / width, int(p.y()) / height] for p in s.points],  # QPonitF
                        # add chris
                        difficult=s.difficult)  # bool

        # lock
        if len(self.canvas.lockedShapes) == 0:
            for s in self.canvas.selectedShapes:
                s.line_color = DEFAULT_LOCK_COLOR
                s.locked = True
            shapes = [format_shape(shape) for shape in self.canvas.selectedShapes]
            trans_dic = []
            for box in shapes:
                trans_dic.append({"transcription": box['label'], "ratio": box['ratio'], 'difficult': box['difficult']})
            self.canvas.lockedShapes = trans_dic
            self.actions.save.setEnabled(True)

        # unlock
        else:
            for s in self.canvas.shapes:
                s.line_color = DEFAULT_LINE_COLOR
            self.canvas.lockedShapes = []
            self.result_dic_locked = []
            self.setDirty()
            self.actions.save.setEnabled(True)


def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    # Tzutalin 201705+: Accept extra arguments to change predefined class file
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--lang", type=str, default='ch', nargs="?")
    arg_parser.add_argument("--gpu", type=str2bool, default=True, nargs="?")
    arg_parser.add_argument("--kie", type=str2bool, default=True, nargs="?")
    arg_parser.add_argument("--predefined_classes_file",
                            default=os.path.join(os.path.dirname(__file__), "data", "predefined_classes.txt"),
                            nargs="?")
    args = arg_parser.parse_args(argv[1:])

    win = MainWindow(lang=args.lang,
                     gpu=args.gpu,
                     kei_mode=args.kie,
                     default_predefined_class_file=args.predefined_classes_file)
    win.show()
    return app, win


def main():
    """construct main app and run it"""
    app, _win = get_main_app(sys.argv)
    return app.exec_()


if __name__ == '__main__':

    resource_file = './libs/resources.py'
    if not os.path.exists(resource_file):
        output = os.system('pyrcc5 -o libs/resources.py resources.qrc')
        assert output == 0, "operate the cmd have some problems ,please check  whether there is a in the lib " \
                            "directory resources.py "

    sys.exit(main())
