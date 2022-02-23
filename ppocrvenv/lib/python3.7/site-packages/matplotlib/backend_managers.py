from matplotlib import _api, cbook, widgets
import matplotlib.backend_tools as tools


class ToolEvent:
    """Event for tool manipulation (add/remove)."""
    def __init__(self, name, sender, tool, data=None):
        self.name = name
        self.sender = sender
        self.tool = tool
        self.data = data


class ToolTriggerEvent(ToolEvent):
    """Event to inform that a tool has been triggered."""
    def __init__(self, name, sender, tool, canvasevent=None, data=None):
        super().__init__(name, sender, tool, data)
        self.canvasevent = canvasevent


class ToolManagerMessageEvent:
    """
    Event carrying messages from toolmanager.

    Messages usually get displayed to the user by the toolbar.
    """
    def __init__(self, name, sender, message):
        self.name = name
        self.sender = sender
        self.message = message


class ToolManager:
    """
    Manager for actions triggered by user interactions (key press, toolbar
    clicks, ...) on a Figure.

    Attributes
    ----------
    figure : `.Figure`
    keypresslock : `~matplotlib.widgets.LockDraw`
        `.LockDraw` object to know if the `canvas` key_press_event is locked.
    messagelock : `~matplotlib.widgets.LockDraw`
        `.LockDraw` object to know if the message is available to write.
    """

    def __init__(self, figure=None):

        self._key_press_handler_id = None

        self._tools = {}
        self._keys = {}
        self._toggled = {}
        self._callbacks = cbook.CallbackRegistry()

        # to process keypress event
        self.keypresslock = widgets.LockDraw()
        self.messagelock = widgets.LockDraw()

        self._figure = None
        self.set_figure(figure)

    @property
    def canvas(self):
        """Canvas managed by FigureManager."""
        if not self._figure:
            return None
        return self._figure.canvas

    @property
    def figure(self):
        """Figure that holds the canvas."""
        return self._figure

    @figure.setter
    def figure(self, figure):
        self.set_figure(figure)

    def set_figure(self, figure, update_tools=True):
        """
        Bind the given figure to the tools.

        Parameters
        ----------
        figure : `.Figure`
        update_tools : bool, default: True
            Force tools to update figure.
        """
        if self._key_press_handler_id:
            self.canvas.mpl_disconnect(self._key_press_handler_id)
        self._figure = figure
        if figure:
            self._key_press_handler_id = self.canvas.mpl_connect(
                'key_press_event', self._key_press)
        if update_tools:
            for tool in self._tools.values():
                tool.figure = figure

    def toolmanager_connect(self, s, func):
        """
        Connect event with string *s* to *func*.

        Parameters
        ----------
        s : str
            The name of the event. The following events are recognized:

            - 'tool_message_event'
            - 'tool_removed_event'
            - 'tool_added_event'

            For every tool added a new event is created

            - 'tool_trigger_TOOLNAME', where TOOLNAME is the id of the tool.

        func : callable
            Callback function for the toolmanager event with signature::

                def func(event: ToolEvent) -> Any

        Returns
        -------
        cid
            The callback id for the connection. This can be used in
            `.toolmanager_disconnect`.
        """
        return self._callbacks.connect(s, func)

    def toolmanager_disconnect(self, cid):
        """
        Disconnect callback id *cid*.

        Example usage::

            cid = toolmanager.toolmanager_connect('tool_trigger_zoom', onpress)
            #...later
            toolmanager.toolmanager_disconnect(cid)
        """
        return self._callbacks.disconnect(cid)

    def message_event(self, message, sender=None):
        """Emit a `ToolManagerMessageEvent`."""
        if sender is None:
            sender = self

        s = 'tool_message_event'
        event = ToolManagerMessageEvent(s, sender, message)
        self._callbacks.process(s, event)

    @property
    def active_toggle(self):
        """Currently toggled tools."""
        return self._toggled

    def get_tool_keymap(self, name):
        """
        Return the keymap associated with the specified tool.

        Parameters
        ----------
        name : str
            Name of the Tool.

        Returns
        -------
        list of str
            List of keys associated with the tool.
        """

        keys = [k for k, i in self._keys.items() if i == name]
        return keys

    def _remove_keys(self, name):
        for k in self.get_tool_keymap(name):
            del self._keys[k]

    def update_keymap(self, name, key):
        """
        Set the keymap to associate with the specified tool.

        Parameters
        ----------
        name : str
            Name of the Tool.
        key : str or list of str
            Keys to associate with the tool.
        """
        if name not in self._tools:
            raise KeyError(f'{name} not in Tools')
        self._remove_keys(name)
        if isinstance(key, str):
            key = [key]
        for k in key:
            if k in self._keys:
                _api.warn_external(
                    f'Key {k} changed from {self._keys[k]} to {name}')
            self._keys[k] = name

    def remove_tool(self, name):
        """
        Remove tool named *name*.

        Parameters
        ----------
        name : str
            Name of the tool.
        """

        tool = self.get_tool(name)
        tool.destroy()

        # If it's a toggle tool and toggled, untoggle
        if getattr(tool, 'toggled', False):
            self.trigger_tool(tool, 'toolmanager')

        self._remove_keys(name)

        s = 'tool_removed_event'
        event = ToolEvent(s, self, tool)
        self._callbacks.process(s, event)

        del self._tools[name]

    def add_tool(self, name, tool, *args, **kwargs):
        """
        Add *tool* to `ToolManager`.

        If successful, adds a new event ``tool_trigger_{name}`` where
        ``{name}`` is the *name* of the tool; the event is fired every time the
        tool is triggered.

        Parameters
        ----------
        name : str
            Name of the tool, treated as the ID, has to be unique.
        tool : class_like, i.e. str or type
            Reference to find the class of the Tool to added.

        Notes
        -----
        args and kwargs get passed directly to the tools constructor.

        See Also
        --------
        matplotlib.backend_tools.ToolBase : The base class for tools.
        """

        tool_cls = self._get_cls_to_instantiate(tool)
        if not tool_cls:
            raise ValueError('Impossible to find class for %s' % str(tool))

        if name in self._tools:
            _api.warn_external('A "Tool class" with the same name already '
                               'exists, not added')
            return self._tools[name]

        if name == 'cursor' and tool_cls != tools.SetCursorBase:
            _api.warn_deprecated("3.5",
                                 message="Overriding ToolSetCursor with "
                                 f"{tool_cls.__qualname__} was only "
                                 "necessary to provide the .set_cursor() "
                                 "method, which is deprecated since "
                                 "%(since)s and will be removed "
                                 "%(removal)s. Please report this to the "
                                 f"{tool_cls.__module__} author.")

        tool_obj = tool_cls(self, name, *args, **kwargs)
        self._tools[name] = tool_obj

        if tool_cls.default_keymap is not None:
            self.update_keymap(name, tool_cls.default_keymap)

        # For toggle tools init the radio_group in self._toggled
        if isinstance(tool_obj, tools.ToolToggleBase):
            # None group is not mutually exclusive, a set is used to keep track
            # of all toggled tools in this group
            if tool_obj.radio_group is None:
                self._toggled.setdefault(None, set())
            else:
                self._toggled.setdefault(tool_obj.radio_group, None)

            # If initially toggled
            if tool_obj.toggled:
                self._handle_toggle(tool_obj, None, None, None)
        tool_obj.set_figure(self.figure)

        self._tool_added_event(tool_obj)
        return tool_obj

    def _tool_added_event(self, tool):
        s = 'tool_added_event'
        event = ToolEvent(s, self, tool)
        self._callbacks.process(s, event)

    def _handle_toggle(self, tool, sender, canvasevent, data):
        """
        Toggle tools, need to untoggle prior to using other Toggle tool.
        Called from trigger_tool.

        Parameters
        ----------
        tool : `.ToolBase`
        sender : object
            Object that wishes to trigger the tool.
        canvasevent : Event
            Original Canvas event or None.
        data : object
            Extra data to pass to the tool when triggering.
        """

        radio_group = tool.radio_group
        # radio_group None is not mutually exclusive
        # just keep track of toggled tools in this group
        if radio_group is None:
            if tool.name in self._toggled[None]:
                self._toggled[None].remove(tool.name)
            else:
                self._toggled[None].add(tool.name)
            return

        # If the tool already has a toggled state, untoggle it
        if self._toggled[radio_group] == tool.name:
            toggled = None
        # If no tool was toggled in the radio_group
        # toggle it
        elif self._toggled[radio_group] is None:
            toggled = tool.name
        # Other tool in the radio_group is toggled
        else:
            # Untoggle previously toggled tool
            self.trigger_tool(self._toggled[radio_group],
                              self,
                              canvasevent,
                              data)
            toggled = tool.name

        # Keep track of the toggled tool in the radio_group
        self._toggled[radio_group] = toggled

    def _get_cls_to_instantiate(self, callback_class):
        # Find the class that corresponds to the tool
        if isinstance(callback_class, str):
            # FIXME: make more complete searching structure
            if callback_class in globals():
                callback_class = globals()[callback_class]
            else:
                mod = 'backend_tools'
                current_module = __import__(mod,
                                            globals(), locals(), [mod], 1)

                callback_class = getattr(current_module, callback_class, False)
        if callable(callback_class):
            return callback_class
        else:
            return None

    def trigger_tool(self, name, sender=None, canvasevent=None, data=None):
        """
        Trigger a tool and emit the ``tool_trigger_{name}`` event.

        Parameters
        ----------
        name : str
            Name of the tool.
        sender : object
            Object that wishes to trigger the tool.
        canvasevent : Event
            Original Canvas event or None.
        data : object
            Extra data to pass to the tool when triggering.
        """
        tool = self.get_tool(name)
        if tool is None:
            return

        if sender is None:
            sender = self

        self._trigger_tool(name, sender, canvasevent, data)

        s = 'tool_trigger_%s' % name
        event = ToolTriggerEvent(s, sender, tool, canvasevent, data)
        self._callbacks.process(s, event)

    def _trigger_tool(self, name, sender=None, canvasevent=None, data=None):
        """Actually trigger a tool."""
        tool = self.get_tool(name)

        if isinstance(tool, tools.ToolToggleBase):
            self._handle_toggle(tool, sender, canvasevent, data)

        # Important!!!
        # This is where the Tool object gets triggered
        tool.trigger(sender, canvasevent, data)

    def _key_press(self, event):
        if event.key is None or self.keypresslock.locked():
            return

        name = self._keys.get(event.key, None)
        if name is None:
            return
        self.trigger_tool(name, canvasevent=event)

    @property
    def tools(self):
        """A dict mapping tool name -> controlled tool."""
        return self._tools

    def get_tool(self, name, warn=True):
        """
        Return the tool object with the given name.

        For convenience, this passes tool objects through.

        Parameters
        ----------
        name : str or `.ToolBase`
            Name of the tool, or the tool itself.
        warn : bool, default: True
            Whether a warning should be emitted it no tool with the given name
            exists.

        Returns
        -------
        `.ToolBase` or None
            The tool or None if no tool with the given name exists.
        """
        if isinstance(name, tools.ToolBase) and name.name in self._tools:
            return name
        if name not in self._tools:
            if warn:
                _api.warn_external(f"ToolManager does not control tool {name}")
            return None
        return self._tools[name]
