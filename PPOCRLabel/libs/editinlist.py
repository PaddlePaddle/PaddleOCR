import sys, time
from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class EditInList(QListWidget):
    def __init__(self):
        super(EditInList,self).__init__()
        # click to edit
        self.clicked.connect(self.item_clicked)  

    def item_clicked(self, modelindex: QModelIndex) -> None:
        self.edited_item = self.currentItem()
        self.closePersistentEditor(self.edited_item)
        item = self.item(modelindex.row())
        # time.sleep(0.2)
        self.edited_item = item
        self.openPersistentEditor(item)
        # time.sleep(0.2)
        self.editItem(item)

    def mouseDoubleClickEvent(self, event):
        # close edit
        for i in range(self.count()):
            self.closePersistentEditor(self.item(i))

    def leaveEvent(self, event):
        # close edit
        for i in range(self.count()):
            self.closePersistentEditor(self.item(i))