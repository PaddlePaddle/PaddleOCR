from functools import reduce
import numpy as np
from ..draw import polygon
from .._shared.version_requirements import require


LEFT_CLICK = 1
RIGHT_CLICK = 3


def _mask_from_vertices(vertices, shape, label):
    mask = np.zeros(shape, dtype=int)
    pr = [y for x, y in vertices]
    pc = [x for x, y in vertices]
    rr, cc = polygon(pr, pc, shape)
    mask[rr, cc] = label
    return mask


@require("matplotlib", ">=3.0.3")
def _draw_polygon(ax, vertices, alpha=0.4):
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt

    polygon = Polygon(vertices, closed=True)
    p = PatchCollection([polygon], match_original=True, alpha=alpha)
    polygon_object = ax.add_collection(p)
    plt.draw()
    return polygon_object


@require("matplotlib", ">=3.0.3")
def manual_polygon_segmentation(image, alpha=0.4, return_all=False):
    """Return a label image based on polygon selections made with the mouse.

    Parameters
    ----------
    image : (M, N[, 3]) array
        Grayscale or RGB image.

    alpha : float, optional
        Transparency value for polygons drawn over the image.

    return_all : bool, optional
        If True, an array containing each separate polygon drawn is returned.
        (The polygons may overlap.) If False (default), latter polygons
        "overwrite" earlier ones where they overlap.

    Returns
    -------
    labels : array of int, shape ([Q, ]M, N)
        The segmented regions. If mode is `'separate'`, the leading dimension
        of the array corresponds to the number of regions that the user drew.

    Notes
    -----
    Use left click to select the vertices of the polygon
    and right click to confirm the selection once all vertices are selected.

    Examples
    --------
    >>> from skimage import data, future, io
    >>> camera = data.camera()
    >>> mask = future.manual_polygon_segmentation(camera)  # doctest: +SKIP
    >>> io.imshow(mask)  # doctest: +SKIP
    >>> io.show()  # doctest: +SKIP
    """
    import matplotlib
    import matplotlib.pyplot as plt

    list_of_vertex_lists = []
    polygons_drawn = []

    temp_list = []
    preview_polygon_drawn = []

    if image.ndim not in (2, 3):
        raise ValueError('Only 2D grayscale or RGB images are supported.')

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    ax.imshow(image, cmap="gray")
    ax.set_axis_off()

    def _undo(*args, **kwargs):
        if list_of_vertex_lists:
            list_of_vertex_lists.pop()
            # Remove last polygon from list of polygons...
            last_poly = polygons_drawn.pop()
            # ... then from the plot
            last_poly.remove()
            fig.canvas.draw_idle()

    undo_pos = fig.add_axes([0.85, 0.05, 0.075, 0.075])
    undo_button = matplotlib.widgets.Button(undo_pos, u'\u27F2')
    undo_button.on_clicked(_undo)

    def _extend_polygon(event):
        # Do not record click events outside axis or in undo button
        if event.inaxes is None or event.inaxes is undo_pos:
            return
        # Do not record click events when toolbar is active
        if fig.canvas.manager.toolbar._active is not None:
            return

        if event.button == LEFT_CLICK:  # Select vertex
            temp_list.append([event.xdata, event.ydata])
            # Remove previously drawn preview polygon if any.
            if preview_polygon_drawn:
                poly = preview_polygon_drawn.pop()
                poly.remove()

            # Preview polygon with selected vertices.
            polygon = _draw_polygon(ax, temp_list, alpha=(alpha / 1.4))
            preview_polygon_drawn.append(polygon)

        elif event.button == RIGHT_CLICK:  # Confirm the selection
            if not temp_list:
                return

            # Store the vertices of the polygon as shown in preview.
            # Redraw polygon and store it in polygons_drawn so that
            # `_undo` works correctly.
            list_of_vertex_lists.append(temp_list[:])
            polygon_object = _draw_polygon(ax, temp_list, alpha=alpha)
            polygons_drawn.append(polygon_object)

            # Empty the temporary variables.
            preview_poly = preview_polygon_drawn.pop()
            preview_poly.remove()
            del temp_list[:]

            plt.draw()

    fig.canvas.mpl_connect('button_press_event', _extend_polygon)

    plt.show(block=True)

    labels = (_mask_from_vertices(vertices, image.shape[:2], i)
              for i, vertices in enumerate(list_of_vertex_lists, start=1))
    if return_all:
        return np.stack(labels)
    else:
        return reduce(np.maximum, labels, np.broadcast_to(0, image.shape[:2]))


@require("matplotlib", ">=3.0.3")
def manual_lasso_segmentation(image, alpha=0.4, return_all=False):
    """Return a label image based on freeform selections made with the mouse.

    Parameters
    ----------
    image : (M, N[, 3]) array
        Grayscale or RGB image.

    alpha : float, optional
        Transparency value for polygons drawn over the image.

    return_all : bool, optional
        If True, an array containing each separate polygon drawn is returned.
        (The polygons may overlap.) If False (default), latter polygons
        "overwrite" earlier ones where they overlap.

    Returns
    -------
    labels : array of int, shape ([Q, ]M, N)
        The segmented regions. If mode is `'separate'`, the leading dimension
        of the array corresponds to the number of regions that the user drew.

    Notes
    -----
    Press and hold the left mouse button to draw around each object.

    Examples
    --------
    >>> from skimage import data, future, io
    >>> camera = data.camera()
    >>> mask = future.manual_lasso_segmentation(camera)  # doctest: +SKIP
    >>> io.imshow(mask)  # doctest: +SKIP
    >>> io.show()  # doctest: +SKIP
    """
    import matplotlib
    import matplotlib.pyplot as plt

    list_of_vertex_lists = []
    polygons_drawn = []

    if image.ndim not in (2, 3):
        raise ValueError('Only 2D grayscale or RGB images are supported.')

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    ax.imshow(image, cmap="gray")
    ax.set_axis_off()

    def _undo(*args, **kwargs):
        if list_of_vertex_lists:
            list_of_vertex_lists.pop()
            # Remove last polygon from list of polygons...
            last_poly = polygons_drawn.pop()
            # ... then from the plot
            last_poly.remove()
            fig.canvas.draw_idle()

    undo_pos = fig.add_axes([0.85, 0.05, 0.075, 0.075])
    undo_button = matplotlib.widgets.Button(undo_pos, u'\u27F2')
    undo_button.on_clicked(_undo)

    def _on_lasso_selection(vertices):
        if len(vertices) < 3:
            return
        list_of_vertex_lists.append(vertices)
        polygon_object = _draw_polygon(ax, vertices, alpha=alpha)
        polygons_drawn.append(polygon_object)
        plt.draw()

    matplotlib.widgets.LassoSelector(ax, _on_lasso_selection)

    plt.show(block=True)

    labels = (_mask_from_vertices(vertices, image.shape[:2], i)
              for i, vertices in enumerate(list_of_vertex_lists, start=1))
    if return_all:
        return np.stack(labels)
    else:
        return reduce(np.maximum, labels, np.broadcast_to(0, image.shape[:2]))
