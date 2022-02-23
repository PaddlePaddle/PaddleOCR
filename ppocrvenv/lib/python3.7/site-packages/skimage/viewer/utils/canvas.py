

class BlitManager(object):

    """Object that manages blits on an axes"""

    def __init__(self, ax):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.canvas.mpl_connect('draw_event', self.on_draw_event)
        self.ax = ax
        self.background = None
        self.artists = []

    def add_artists(self, artists):
        self.artists.extend(artists)
        self.redraw()

    def remove_artists(self, artists):
        for artist in artists:
            self.artists.remove(artist)

    def on_draw_event(self, event=None):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.draw_artists()

    def redraw(self):
        if self.background is not None:
            self.canvas.restore_region(self.background)
            self.draw_artists()
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def draw_artists(self):
        for artist in self.artists:
            self.ax.draw_artist(artist)


class EventManager(object):

    """Object that manages events on a canvas"""

    def __init__(self, ax):
        self.canvas = ax.figure.canvas
        self.connect_event('button_press_event', self.on_mouse_press)
        self.connect_event('key_press_event', self.on_key_press)
        self.connect_event('button_release_event', self.on_mouse_release)
        self.connect_event('motion_notify_event', self.on_move)
        self.connect_event('scroll_event', self.on_scroll)

        self.tools = []
        self.active_tool = None

    def connect_event(self, name, handler):
        self.canvas.mpl_connect(name, handler)

    def attach(self, tool):
        self.tools.append(tool)
        self.active_tool = tool

    def detach(self, tool):
        self.tools.remove(tool)
        if self.tools:
            self.active_tool = self.tools[-1]
        else:
            self.active_tool = None

    def on_mouse_press(self, event):
        for tool in self.tools:
            if not tool.ignore(event) and tool.hit_test(event):
                self.active_tool = tool
                break
        if self.active_tool and not self.active_tool.ignore(event):
            self.active_tool.on_mouse_press(event)
            return
        for tool in reversed(self.tools):
            if not tool.ignore(event):
                self.active_tool = tool
                tool.on_mouse_press(event)
                return

    def on_key_press(self, event):
        tool = self._get_tool(event)
        if tool is not None:
            tool.on_key_press(event)

    def _get_tool(self, event):
        if not self.tools or self.active_tool.ignore(event):
            return None
        return self.active_tool

    def on_mouse_release(self, event):
        tool = self._get_tool(event)
        if tool is not None:
            tool.on_mouse_release(event)

    def on_move(self, event):
        tool = self._get_tool(event)
        if tool is not None:
            tool.on_move(event)

    def on_scroll(self, event):
        tool = self._get_tool(event)
        if tool is not None:
            tool.on_scroll(event)
