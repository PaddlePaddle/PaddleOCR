_qt_version = None
has_qt = True

try:
    from matplotlib.backends.qt_compat import QtGui, QtCore, QtWidgets, QT_RC_MAJOR_VERSION as _qt_version
except ImportError:
    try:
        from matplotlib.backends.qt4_compat import QtGui, QtCore
        QtWidgets = QtGui
        _qt_version = 4
    except ImportError:
        # Mock objects
        class QtGui_cls(object):
            QMainWindow = object
            QDialog = object
            QWidget = object

        class QtCore_cls(object):
            class Qt(object):
                 TopDockWidgetArea = None
                 BottomDockWidgetArea = None
                 LeftDockWidgetArea = None
                 RightDockWidgetArea = None

            def Signal(self, *args, **kwargs): 
                pass

        QtGui = QtWidgets = QtGui_cls()
        QtCore = QtCore_cls()

        has_qt = False

if _qt_version == 5:
    from matplotlib.backends.backend_qt5 import FigureManagerQT
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
elif _qt_version == 4:
    from matplotlib.backends.backend_qt4 import FigureManagerQT
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
else:
    FigureManagerQT = object
    FigureCanvasQTAgg = object

Qt = QtCore.Qt
Signal = QtCore.Signal
