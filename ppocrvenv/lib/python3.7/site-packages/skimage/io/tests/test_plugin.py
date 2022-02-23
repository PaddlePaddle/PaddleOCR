from contextlib import contextmanager
import numpy as np
import pytest

from skimage._shared._dependency_checks import has_mpl
from skimage import io
from skimage.io import manage_plugins


priority_plugin = 'pil'


def setup():
    io.use_plugin('pil')


def teardown_module():
    io.reset_plugins()


@contextmanager
def protect_preferred_plugins():
    """Contexts where `preferred_plugins` can be modified w/o side-effects."""
    preferred_plugins = manage_plugins.preferred_plugins.copy()
    try:
        yield
    finally:
        manage_plugins.preferred_plugins = preferred_plugins


def test_failed_use():
    with pytest.raises(ValueError):
        manage_plugins.use_plugin('asd')


@pytest.mark.skipif(not has_mpl, reason="matplotlib not installed")
def test_use_priority():
    manage_plugins.use_plugin(priority_plugin)
    plug, func = manage_plugins.plugin_store['imread'][0]
    np.testing.assert_equal(plug, priority_plugin)

    manage_plugins.use_plugin('matplotlib')
    plug, func = manage_plugins.plugin_store['imread'][0]
    np.testing.assert_equal(plug, 'matplotlib')


@pytest.mark.skipif(not has_mpl, reason="matplotlib not installed")
def test_load_preferred_plugins_all():
    from skimage.io._plugins import pil_plugin, matplotlib_plugin

    with protect_preferred_plugins():
        manage_plugins.preferred_plugins = {'all': ['pil'],
                                            'imshow': ['matplotlib']}
        manage_plugins.reset_plugins()

        for plugin_type in ('imread', 'imsave'):
            plug, func = manage_plugins.plugin_store[plugin_type][0]
            assert func == getattr(pil_plugin, plugin_type)
        plug, func = manage_plugins.plugin_store['imshow'][0]
        assert func == getattr(matplotlib_plugin, 'imshow')


@pytest.mark.skipif(not has_mpl, reason="matplotlib not installed")
def test_load_preferred_plugins_imread():
    from skimage.io._plugins import pil_plugin, matplotlib_plugin

    with protect_preferred_plugins():
        manage_plugins.preferred_plugins['imread'] = ['pil']
        manage_plugins.reset_plugins()

        plug, func = manage_plugins.plugin_store['imread'][0]
        assert func == pil_plugin.imread
        plug, func = manage_plugins.plugin_store['imshow'][0]
        assert func == matplotlib_plugin.imshow, func.__module__
