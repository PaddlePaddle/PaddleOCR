import pytest

import matplotlib.pyplot as plt


pytest.importorskip("matplotlib.backends.backend_macosx",
                    reason="These are mac only tests")


@pytest.mark.backend('macosx')
def test_cached_renderer():
    # Make sure that figures have an associated renderer after
    # a fig.canvas.draw() call
    fig = plt.figure(1)
    fig.canvas.draw()
    assert fig._cachedRenderer is not None

    fig = plt.figure(2)
    fig.draw_without_rendering()
    assert fig._cachedRenderer is not None
