from ..qt import QtWidgets, QtCore, Qt
from ..utils import RequiredAttr


__all__ = ['BaseWidget', 'Slider', 'ComboBox', 'CheckBox', 'Text', 'Button']


class BaseWidget(QtWidgets.QWidget):

    plugin = RequiredAttr("Widget is not attached to a Plugin.")

    def __init__(self, name, ptype=None, callback=None):
        super(BaseWidget, self).__init__()
        self.name = name
        self.ptype = ptype
        self.callback = callback
        self.plugin = None

    @property
    def val(self):
        msg = "Subclass of BaseWidget requires `val` property"
        raise NotImplementedError(msg)

    def _value_changed(self, value):
        self.callback(self.name, value)


class Text(BaseWidget):

    def __init__(self, name=None, text=''):
        super(Text, self).__init__(name)
        self._label = QtWidgets.QLabel()
        self.text = text
        self.layout = QtWidgets.QHBoxLayout(self)
        if name is not None:
            name_label = QtWidgets.QLabel()
            name_label.setText(name)
            self.layout.addWidget(name_label)
        self.layout.addWidget(self._label)

    @property
    def text(self):
        return self._label.text()

    @text.setter
    def text(self, text_str):
        self._label.setText(text_str)


class Slider(BaseWidget):
    """Slider widget for adjusting numeric parameters.

    Parameters
    ----------
    name : str
        Name of slider parameter. If this parameter is passed as a keyword
        argument, it must match the name of that keyword argument (spaces are
        replaced with underscores). In addition, this name is displayed as the
        name of the slider.
    low, high : float
        Range of slider values.
    value : float
        Default slider value. If None, use midpoint between `low` and `high`.
    value_type : {'float' | 'int'}, optional
        Numeric type of slider value.
    ptype : {'kwarg' | 'arg' | 'plugin'}, optional
        Parameter type.
    callback : callable f(widget_name, value), optional
        Callback function called in response to slider changes.
        *Note:* This function is typically set (overridden) when the widget is
        added to a plugin.
    orientation : {'horizontal' | 'vertical'}, optional
        Slider orientation.
    update_on : {'release' | 'move'}, optional
        Control when callback function is called: on slider move or release.
    """

    def __init__(self, name, low=0.0, high=1.0, value=None, value_type='float',
                 ptype='kwarg', callback=None, max_edit_width=60,
                 orientation='horizontal', update_on='release'):
        super(Slider, self).__init__(name, ptype, callback)

        if value is None:
            value = (high - low) / 2.

        # Set widget orientation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if orientation == 'vertical':
            self.slider = QtWidgets.QSlider(Qt.Vertical)
            align_text = QtCore.Qt.AlignHCenter
            align_value = QtCore.Qt.AlignHCenter
            self.layout = QtWidgets.QVBoxLayout(self)
        elif orientation == 'horizontal':
            self.slider = QtWidgets.QSlider(Qt.Horizontal)
            align_text = QtCore.Qt.AlignLeft
            align_value = QtCore.Qt.AlignRight
            self.layout = QtWidgets.QHBoxLayout(self)
        else:
            msg = "Unexpected value %s for 'orientation'"
            raise ValueError(msg % orientation)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Set slider behavior for float and int values.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if value_type == 'float':
            # divide slider into 1000 discrete values
            slider_max = 1000
            self._scale = float(high - low) / slider_max
            self.slider.setRange(0, slider_max)
            self.value_fmt = '%2.2f'
        elif value_type == 'int':
            self.slider.setRange(low, high)
            self.value_fmt = '%d'
        else:
            msg = "Expected `value_type` to be 'float' or 'int'; received: %s"
            raise ValueError(msg % value_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.value_type = value_type
        self._low = low
        self._high = high
        # Update slider position to default value
        self.val = value

        if update_on == 'move':
            self.slider.valueChanged.connect(self._on_slider_changed)
        elif update_on == 'release':
            self.slider.sliderReleased.connect(self._on_slider_changed)
        else:
            raise ValueError("Unexpected value %s for 'update_on'" % update_on)
        self.slider.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.name_label = QtWidgets.QLabel()
        self.name_label.setText(self.name)
        self.name_label.setAlignment(align_text)

        self.editbox = QtWidgets.QLineEdit()
        self.editbox.setMaximumWidth(max_edit_width)
        self.editbox.setText(self.value_fmt % self.val)
        self.editbox.setAlignment(align_value)
        self.editbox.editingFinished.connect(self._on_editbox_changed)

        self.layout.addWidget(self.name_label)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.editbox)

    def _on_slider_changed(self):
        """Call callback function with slider's name and value as parameters"""
        value = self.val
        self.editbox.setText(str(value)[:4])
        self.callback(self.name, value)

    def _on_editbox_changed(self):
        """Validate input and set slider value"""
        try:
            value = float(self.editbox.text())
        except ValueError:
            self._bad_editbox_input()
            return
        if not self._low <= value <= self._high:
            self._bad_editbox_input()
            return

        self.val = value
        self._good_editbox_input()
        self.callback(self.name, value)

    def _good_editbox_input(self):
        self.editbox.setStyleSheet("background-color: rgb(255, 255, 255)")

    def _bad_editbox_input(self):
        self.editbox.setStyleSheet("background-color: rgb(255, 200, 200)")

    @property
    def val(self):
        value = self.slider.value()
        if self.value_type == 'float':
            value = value * self._scale + self._low
        return value

    @val.setter
    def val(self, value):
        if self.value_type == 'float':
            value = (value - self._low) / self._scale
        self.slider.setValue(int(value))


class ComboBox(BaseWidget):
    """ComboBox widget for selecting among a list of choices.

    Parameters
    ----------
    name : str
        Name of ComboBox parameter. If this parameter is passed as a keyword
        argument, it must match the name of that keyword argument (spaces are
        replaced with underscores). In addition, this name is displayed as the
        name of the ComboBox.
    items: list of str
        Allowed parameter values.
    ptype : {'arg' | 'kwarg' | 'plugin'}, optional
        Parameter type.
    callback : callable f(widget_name, value), optional
        Callback function called in response to combobox changes.
        *Note:* This function is typically set (overridden) when the widget is
        added to a plugin.
    """

    def __init__(self, name, items, ptype='kwarg', callback=None):
        super(ComboBox, self).__init__(name, ptype, callback)

        self.name_label = QtWidgets.QLabel()
        self.name_label.setText(self.name)
        self.name_label.setAlignment(QtCore.Qt.AlignLeft)

        self._combo_box = QtWidgets.QComboBox()
        self._combo_box.addItems(list(items))

        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.addWidget(self.name_label)
        self.layout.addWidget(self._combo_box)

        self._combo_box.currentIndexChanged.connect(self._value_changed)

    @property
    def val(self):
        return self._combo_box.currentText()

    @property
    def index(self):
        return self._combo_box.currentIndex()

    @index.setter
    def index(self, i):
        self._combo_box.setCurrentIndex(i)


class CheckBox(BaseWidget):
    """CheckBox widget

    Parameters
    ----------
    name : str
        Name of CheckBox parameter. If this parameter is passed as a keyword
        argument, it must match the name of that keyword argument (spaces are
        replaced with underscores). In addition, this name is displayed as the
        name of the CheckBox.
    value: {False, True}, optional
        Initial state of the CheckBox.
    alignment: {'center','left','right'}, optional
        Checkbox alignment
    ptype : {'arg' | 'kwarg' | 'plugin'}, optional
        Parameter type
    callback : callable f(widget_name, value), optional
        Callback function called in response to checkbox changes.
        *Note:* This function is typically set (overridden) when the widget is
        added to a plugin.
    """

    def __init__(self, name, value=False, alignment='center', ptype='kwarg',
                 callback=None):
        super(CheckBox, self).__init__(name, ptype, callback)

        self._check_box = QtWidgets.QCheckBox()
        self._check_box.setChecked(value)
        self._check_box.setText(self.name)

        self.layout = QtWidgets.QHBoxLayout(self)
        if alignment == 'center':
            self.layout.setAlignment(QtCore.Qt.AlignCenter)
        elif alignment == 'left':
            self.layout.setAlignment(QtCore.Qt.AlignLeft)
        elif alignment == 'right':
            self.layout.setAlignment(QtCore.Qt.AlignRight)
        else:
            raise ValueError("Unexpected value %s for 'alignment'" % alignment)

        self.layout.addWidget(self._check_box)

        self._check_box.stateChanged.connect(self._value_changed)

    @property
    def val(self):
        return self._check_box.isChecked()

    @val.setter
    def val(self, i):
        self._check_box.setChecked(i)


class Button(BaseWidget):
    """Button which calls callback upon click.

    Parameters
    ----------
    name : str
        Name of button.
    callback : callable f()
        Function to call when button is clicked.
    """

    def __init__(self, name, callback):
        super(Button, self).__init__(self)
        self._button = QtWidgets.QPushButton(name)
        self._button.clicked.connect(callback)

        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.addWidget(self._button)
