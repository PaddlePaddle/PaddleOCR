# the module for the qt color_mixer plugin
from qtpy import QtCore
from qtpy.QtWidgets import (QWidget, QStackedWidget, QSlider, QGridLayout,
                            QLabel, QFrame, QComboBox, QRadioButton,
                            QPushButton)

from .util import ColorMixer


class IntelligentSlider(QWidget):
    ''' A slider that adds a 'name' attribute and calls a callback
    with 'name' as an argument to the registered callback.

    This allows you to create large groups of sliders in a loop,
    but still keep track of the individual events

    It also prints a label below the slider.

    The range of the slider is hardcoded from zero - 1000,
    but it supports a conversion factor so you can scale the results'''

    def __init__(self, name, a, b, callback):
        QWidget.__init__(self)
        self.name = name
        self.callback = callback
        self.a = a
        self.b = b
        self.manually_triggered = False

        self.slider = QSlider()
        self.slider.setRange(0, 1000)
        self.slider.setValue(500)
        self.slider.valueChanged.connect(self.slider_changed)

        self.name_label = QLabel()
        self.name_label.setText(self.name)
        self.name_label.setAlignment(QtCore.Qt.AlignCenter)

        self.value_label = QLabel()
        self.value_label.setText('%2.2f' % (self.slider.value() * self.a
                                             + self.b))
        self.value_label.setAlignment(QtCore.Qt.AlignCenter)

        self.layout = QGridLayout(self)
        self.layout.addWidget(self.name_label, 0, 0)
        self.layout.addWidget(self.slider, 1, 0, QtCore.Qt.AlignHCenter)
        self.layout.addWidget(self.value_label, 2, 0)

    # bind this to the valueChanged signal of the slider
    def slider_changed(self, val):
        val = self.val()
        self.value_label.setText(str(val)[:4])

        if not self.manually_triggered:
            self.callback(self.name, val)

    def set_conv_fac(self, a, b):
        self.a = a
        self.b = b

    def set_value(self, val):
        self.manually_triggered = True
        self.slider.setValue(int((val - self.b) / self.a))
        self.value_label.setText('%2.2f' % val)
        self.manually_triggered = False

    def val(self):
        return self.slider.value() * self.a + self.b


class MixerPanel(QFrame):
    '''A color mixer to hook up to an image.
    You pass the image you the panel to operate on
    and it operates on that image in place. You also
    pass a callback to be called to trigger a refresh.
    This callback is called every time the mixer modifies
    your image.'''
    def __init__(self, img):
        QFrame.__init__(self)
        # self.setFrameStyle(QFrame.Box | QFrame.Sunken)

        self.img = img
        self.mixer = ColorMixer(self.img)
        self.callback = None

        #---------------------------------------------------------------
        # ComboBox
        #---------------------------------------------------------------

        self.combo_box_entries = ['RGB Color', 'HSV Color',
                                  'Brightness/Contrast',
                                  'Gamma',
                                  'Gamma (Sigmoidal)']
        self.combo_box = QComboBox()
        for entry in self.combo_box_entries:
            self.combo_box.addItem(entry)
        self.combo_box.currentIndexChanged.connect(self.combo_box_changed)

        #---------------------------------------------------------------
        # RGB color sliders
        #---------------------------------------------------------------

        # radio buttons
        self.rgb_add = QRadioButton('Additive')
        self.rgb_mul = QRadioButton('Multiplicative')
        self.rgb_mul.toggled.connect(self.rgb_radio_changed)
        self.rgb_add.toggled.connect(self.rgb_radio_changed)

        # sliders
        rs = IntelligentSlider('R', 0.51, -255, self.rgb_changed)
        gs = IntelligentSlider('G', 0.51, -255, self.rgb_changed)
        bs = IntelligentSlider('B', 0.51, -255, self.rgb_changed)
        self.rs = rs
        self.gs = gs
        self.bs = bs

        self.rgb_widget = QWidget()
        self.rgb_widget.layout = QGridLayout(self.rgb_widget)
        self.rgb_widget.layout.addWidget(self.rgb_add, 0, 0, 1, 3)
        self.rgb_widget.layout.addWidget(self.rgb_mul, 1, 0, 1, 3)
        self.rgb_widget.layout.addWidget(self.rs, 2, 0)
        self.rgb_widget.layout.addWidget(self.gs, 2, 1)
        self.rgb_widget.layout.addWidget(self.bs, 2, 2)

        #---------------------------------------------------------------
        # HSV sliders
        #---------------------------------------------------------------
        # radio buttons
        self.hsv_add = QRadioButton('Additive')
        self.hsv_mul = QRadioButton('Multiplicative')
        self.hsv_mul.toggled.connect(self.hsv_radio_changed)
        self.hsv_mul.toggled.connect(self.hsv_radio_changed)

        # sliders
        hs = IntelligentSlider('H', 0.36, -180, self.hsv_changed)
        ss = IntelligentSlider('S', 0.002, 0, self.hsv_changed)
        vs = IntelligentSlider('V', 0.002, 0, self.hsv_changed)
        self.hs = hs
        self.ss = ss
        self.vs = vs

        self.hsv_widget = QWidget()
        self.hsv_widget.layout = QGridLayout(self.hsv_widget)
        self.hsv_widget.layout.addWidget(self.hsv_add, 0, 0, 1, 3)
        self.hsv_widget.layout.addWidget(self.hsv_mul, 1, 0, 1, 3)
        self.hsv_widget.layout.addWidget(self.hs, 2, 0)
        self.hsv_widget.layout.addWidget(self.ss, 2, 1)
        self.hsv_widget.layout.addWidget(self.vs, 2, 2)

        #---------------------------------------------------------------
        # Brightness/Contrast sliders
        #---------------------------------------------------------------
        # sliders
        cont = IntelligentSlider('x', 0.002, 0, self.bright_changed)
        bright = IntelligentSlider('+', 0.51, -255, self.bright_changed)
        self.cont = cont
        self.bright = bright

        # layout
        self.bright_widget = QWidget()
        self.bright_widget.layout = QGridLayout(self.bright_widget)
        self.bright_widget.layout.addWidget(self.cont, 0, 0)
        self.bright_widget.layout.addWidget(self.bright, 0, 1)

        #----------------------------------------------------------------------
        # Gamma Slider
        #----------------------------------------------------------------------
        gamma = IntelligentSlider('gamma', 0.005, 0, self.gamma_changed)
        self.gamma = gamma

        # layout
        self.gamma_widget = QWidget()
        self.gamma_widget.layout = QGridLayout(self.gamma_widget)
        self.gamma_widget.layout.addWidget(self.gamma, 0, 0)

        #---------------------------------------------------------------
        # Sigmoid Gamma sliders
        #---------------------------------------------------------------
        # sliders
        alpha = IntelligentSlider('alpha', 0.011, 1, self.sig_gamma_changed)
        beta = IntelligentSlider('beta', 0.012, 0, self.sig_gamma_changed)
        self.a_gamma = alpha
        self.b_gamma = beta

        # layout
        self.sig_gamma_widget = QWidget()
        self.sig_gamma_widget.layout = QGridLayout(self.sig_gamma_widget)
        self.sig_gamma_widget.layout.addWidget(self.a_gamma, 0, 0)
        self.sig_gamma_widget.layout.addWidget(self.b_gamma, 0, 1)

        #---------------------------------------------------------------
        # Buttons
        #---------------------------------------------------------------
        self.commit_button = QPushButton('Commit')
        self.commit_button.clicked.connect(self.commit_changes)
        self.revert_button = QPushButton('Revert')
        self.revert_button.clicked.connect(self.revert_changes)

        #---------------------------------------------------------------
        # Mixer Layout
        #---------------------------------------------------------------
        self.sliders = QStackedWidget()
        self.sliders.addWidget(self.rgb_widget)
        self.sliders.addWidget(self.hsv_widget)
        self.sliders.addWidget(self.bright_widget)
        self.sliders.addWidget(self.gamma_widget)
        self.sliders.addWidget(self.sig_gamma_widget)

        self.layout = QGridLayout(self)
        self.layout.addWidget(self.combo_box, 0, 0)
        self.layout.addWidget(self.sliders, 1, 0)
        self.layout.addWidget(self.commit_button, 2, 0)
        self.layout.addWidget(self.revert_button, 3, 0)

        #---------------------------------------------------------------
        # State Initialization
        #---------------------------------------------------------------

        self.combo_box.setCurrentIndex(0)
        self.rgb_mul.setChecked(True)
        self.hsv_mul.setChecked(True)

    def set_callback(self, callback):
        self.callback = callback

    def combo_box_changed(self, index):
        self.sliders.setCurrentIndex(index)
        self.reset()

    def rgb_radio_changed(self):
        self.reset()

    def hsv_radio_changed(self):
        self.reset()

    def reset(self):
        self.reset_sliders()
        self.mixer.set_to_stateimg()
        if self.callback:
            self.callback()

    def reset_sliders(self):
        # handle changing the conversion factors necessary
        if self.rgb_add.isChecked():
            self.rs.set_conv_fac(0.51, -255)
            self.rs.set_value(0)
            self.gs.set_conv_fac(0.51, -255)
            self.gs.set_value(0)
            self.bs.set_conv_fac(0.51, -255)
            self.bs.set_value(0)
        else:
            self.rs.set_conv_fac(0.002, 0)
            self.rs.set_value(1.)
            self.gs.set_conv_fac(0.002, 0)
            self.gs.set_value(1.)
            self.bs.set_conv_fac(0.002, 0)
            self.bs.set_value(1.)

        self.hs.set_value(0)
        if self.hsv_add.isChecked():
            self.ss.set_conv_fac(0.002, -1)
            self.ss.set_value(0)
            self.vs.set_conv_fac(0.002, -1)
            self.vs.set_value(0)
        else:
            self.ss.set_conv_fac(0.002, 0)
            self.ss.set_value(1.)
            self.vs.set_conv_fac(0.002, 0)
            self.vs.set_value(1.)

        self.bright.set_value(0)
        self.cont.set_value(1.)

        self.gamma.set_value(1)
        self.a_gamma.set_value(1)
        self.b_gamma.set_value(0.5)

    def rgb_changed(self, name, val):
        if name == 'R':
            channel = self.mixer.RED
        elif name == 'G':
            channel = self.mixer.GREEN
        else:
            channel = self.mixer.BLUE

        if self.rgb_mul.isChecked():
            self.mixer.multiply(channel, val)
        elif self.rgb_add.isChecked():
            self.mixer.add(channel, val)
        else:
            pass

        if self.callback:
            self.callback()

    def hsv_changed(self, name, val):
        h = self.hs.val()
        s = self.ss.val()
        v = self.vs.val()

        if self.hsv_mul.isChecked():
            self.mixer.hsv_multiply(h, s, v)
        elif self.hsv_add.isChecked():
            self.mixer.hsv_add(h, s, v)
        else:
            pass

        if self.callback:
            self.callback()

    def bright_changed(self, name, val):
        b = self.bright.val()
        c = self.cont.val()
        self.mixer.brightness(c, b)

        if self.callback:
            self.callback()

    def gamma_changed(self, name, val):
        self.mixer.gamma(val)

        if self.callback:
            self.callback()

    def sig_gamma_changed(self, name, val):
        ag = self.a_gamma.val()
        bg = self.b_gamma.val()
        self.mixer.sigmoid_gamma(ag, bg)

        if self.callback:
            self.callback()

    def commit_changes(self):
        self.mixer.commit_changes()
        self.reset_sliders()

    def revert_changes(self):
        self.mixer.revert()
        self.reset_sliders()

        if self.callback:
            self.callback()
