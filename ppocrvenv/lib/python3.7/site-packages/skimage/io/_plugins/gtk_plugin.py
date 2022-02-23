from .util import prepare_for_display, window_manager, GuiLockError

try:
    # we try to acquire the gui lock first
    # or else the gui import might trample another
    # gui's pyos_inputhook.
    window_manager.acquire('gtk')
except GuiLockError as gle:
    print(gle)
else:
    try:
        import gtk
    except ImportError:
        print('pygtk libraries not installed.')
        print('plugin not loaded.')
        window_manager._release('gtk')
    else:

        class ImageWindow(gtk.Window):
            def __init__(self, arr, mgr):
                gtk.Window.__init__(self)
                self.mgr = mgr
                self.mgr.add_window(self)

                self.connect("destroy", self.destroy)

                width = arr.shape[1]
                height = arr.shape[0]
                rstride = arr.strides[0]
                pb = gtk.gdk.pixbuf_new_from_data(arr.data,
                                                  gtk.gdk.COLORSPACE_RGB,
                                                  False, 8, width, height,
                                                  rstride)
                self.img = gtk.Image()
                self.img.set_from_pixbuf(pb)

                self.add(self.img)
                self.img.show()

            def destroy(self, widget, data=None):
                self.mgr.remove_window(self)

        def imshow(arr):
            arr = prepare_for_display(arr)

            iw = ImageWindow(arr, window_manager)
            iw.show()

        def _app_show():
            if window_manager.has_windows():
                window_manager.register_callback(gtk.main_quit)
                gtk.main()
            else:
                print('no images to display')
