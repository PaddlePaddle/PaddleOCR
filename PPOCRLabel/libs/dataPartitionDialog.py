try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from libs.utils import newIcon

import time
import datetime
import json
import cv2
import numpy as np


BB = QDialogButtonBox

class DataPartitionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__()
        self.parnet = parent
        self.title = 'DATA PARTITION'

        self.train_ratio = 70
        self.val_ratio = 15
        self.test_ratio = 15
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setWindowModality(Qt.ApplicationModal)

        self.flag_accept = True

        if self.parnet.lang == 'ch':
            msg = "导出JSON前请保存所有图像的标注且关闭EXCEL!"
        else:
            msg = "Please save all the annotations and close the EXCEL before exporting JSON!"

        info_msg = QLabel(msg, self)
        info_msg.setWordWrap(True)
        info_msg.setStyleSheet("color: red")
        info_msg.setFont(QFont('Arial', 12))

        train_lbl = QLabel('Train split: ', self)
        train_lbl.setFont(QFont('Arial', 15))
        val_lbl = QLabel('Valid split: ', self)
        val_lbl.setFont(QFont('Arial', 15))
        test_lbl = QLabel('Test split: ', self)
        test_lbl.setFont(QFont('Arial', 15))

        self.train_input = QLineEdit(self)
        self.train_input.setFont(QFont('Arial', 15))
        self.val_input = QLineEdit(self)
        self.val_input.setFont(QFont('Arial', 15))
        self.test_input = QLineEdit(self)
        self.test_input.setFont(QFont('Arial', 15))

        self.train_input.setText(str(self.train_ratio))
        self.val_input.setText(str(self.val_ratio))
        self.test_input.setText(str(self.test_ratio))

        validator = QIntValidator(0, 100)
        self.train_input.setValidator(validator)
        self.val_input.setValidator(validator)
        self.test_input.setValidator(validator)

        gridlayout = QGridLayout()
        gridlayout.addWidget(info_msg, 0, 0, 1, 2)
        gridlayout.addWidget(train_lbl, 1, 0)
        gridlayout.addWidget(val_lbl, 2, 0)
        gridlayout.addWidget(test_lbl, 3, 0)
        gridlayout.addWidget(self.train_input, 1, 1)
        gridlayout.addWidget(self.val_input, 2, 1)
        gridlayout.addWidget(self.test_input, 3, 1)

        bb = BB(BB.Ok | BB.Cancel, Qt.Horizontal, self)
        bb.button(BB.Ok).setIcon(newIcon('done'))
        bb.button(BB.Cancel).setIcon(newIcon('undo'))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.cancel)
        gridlayout.addWidget(bb, 4, 0, 1, 2)

        self.setLayout(gridlayout)
        
        self.show()

    def validate(self):
        self.flag_accept = True
        self.accept()

    def cancel(self):
        self.flag_accept = False
        self.reject()
    
    def getStatus(self):
        return self.flag_accept

    def getDataPartition(self):
        self.train_ratio = int(self.train_input.text())
        self.val_ratio = int(self.val_input.text())
        self.test_ratio = int(self.test_input.text())

        return self.train_ratio, self.val_ratio, self.test_ratio

    def closeEvent(self, event):
        self.flag_accept = False
        self.reject()


