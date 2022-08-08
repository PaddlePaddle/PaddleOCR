# !/usr/bin/env python
# -*- coding: utf-8 -*-
from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import QListWidget


class EditInList(QListWidget):
    def __init__(self):
        super(EditInList, self).__init__()
        self.edited_item = None

    def item_clicked(self, modelindex: QModelIndex):
        try:
            if self.edited_item is not None:
                self.closePersistentEditor(self.edited_item)
        except:
            self.edited_item = self.currentItem()

        self.edited_item = self.item(modelindex.row())
        self.openPersistentEditor(self.edited_item)
        self.editItem(self.edited_item)

    def mouseDoubleClickEvent(self, event):
        pass

    def leaveEvent(self, event):
        # close edit
        for i in range(self.count()):
            self.closePersistentEditor(self.item(i))