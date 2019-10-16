import gi, time
gi.require_version('Gtk', '3.0')
gi.require_version('Wnck', '3.0')

from gi.repository import Gtk, Wnck, GdkX11


class ActivateWindow:

    def __init__(self):
        # Gtk.init([])
        # Gtk.init
        # Gtk.gdk,threads_init()

        # gi.threads_init()
        self.screen = Wnck.Screen.get_default()
        self.screen.force_update()
        self.ETS_window = None
        self.ETS_window_coords = None

    def activate(self):
        for window in self.screen.get_windows():
            if window.get_name() == 'Euro Truck Simulator 2':
                # now = GdkX11.x11_get_server_time(window)
                now = Gtk.get_current_event_time()
                # window.activate(now)

    def get_ets_window_coords(self):
        for window in self.screen.get_windows():
            if window.get_name() == 'Euro Truck Simulator 2':
                self.ETS_window = window
                self.ETS_window_coords = window.get_geometry()