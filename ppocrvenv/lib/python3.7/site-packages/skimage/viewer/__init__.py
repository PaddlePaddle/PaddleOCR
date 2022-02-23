from .._shared.utils import warn
from .._shared._dependency_checks import has_mpl
from .qt import has_qt

if not has_qt:
    warn('Viewer requires Qt')

if has_mpl:
    from .viewers import ImageViewer, CollectionViewer
else:
    ImageViewer, CollectionViewer = None, None
    warn('Viewer requires matplotlib', stacklevel=2)

__all__ = ['ImageViewer', 'CollectionViewer']
