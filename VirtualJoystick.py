import uinput
import time
import multiprocessing
from ActivateWindow import *

class VirtualJoystick(multiprocessing.Process):

    ETS_window_coords = None

    def __init__(self, inq,  _name="My controller", _bustype=0x0003, _vendor=0x046d, _product=0xc39b, _version=0x0111, _events=None):
        print("Virtual joystick initialised")
        self.winact = ActivateWindow()
        self.winact.get_ets_window_coords()
        self.ETS_window_coords = self.winact.ETS_window_coords

        multiprocessing.Process.__init__(self)
        self._inq = inq
        self.deamon = True

        self.speed = 0
        self.pause = 1
        if _events is None:
            _events = (
                uinput.BTN_JOYSTICK,
                uinput.ABS_X + (-32767, 32767, 0, 0),
                uinput.ABS_Y + (0, 65535, 0, 0),  # acceleration
                uinput.ABS_Z + (0, 65535, 0, 0),  # brake
                (1, 41),                             # tylda
                uinput.KEY_A,
                uinput.KEY_B,
                uinput.KEY_C,
                uinput.KEY_D,
                uinput.KEY_E,
                uinput.KEY_F,
                uinput.KEY_G,
                uinput.KEY_H,
                uinput.KEY_I,
                uinput.KEY_J,
                uinput.KEY_K,
                uinput.KEY_L,
                uinput.KEY_M,
                uinput.KEY_N,
                uinput.KEY_O,
                uinput.KEY_P,
                uinput.KEY_Q,
                uinput.KEY_R,
                uinput.KEY_S,
                uinput.KEY_T,
                uinput.KEY_U,
                uinput.KEY_V,
                uinput.KEY_W,
                uinput.KEY_X,
                uinput.KEY_Y,
                uinput.KEY_1,
                uinput.KEY_2,
                uinput.KEY_3,
                uinput.KEY_4,
                uinput.KEY_5,
                uinput.KEY_6,
                uinput.KEY_7,
                uinput.KEY_8,
                uinput.KEY_9,
                uinput.KEY_0,
                uinput.KEY_ENTER,
            )
        self.device = uinput.Device(name=_name, bustype=_bustype, vendor=_vendor, product=_product, events=_events)
        self.emit(1, 1, 2)
        self.device.emit(uinput.BTN_JOYSTICK, 1)
        time.sleep(5)
        for i in range(-32000, 32000, 100):
            self.emit(i, accel=None, brake=None)

        for i in range(0, 64000, 100):
            self.emit(3, i, brake=None)

        for i in range(0, 64000, 100):
            self.emit(4, None, i)


    def emit(self, angle, accel=None, brake=None):
        self.winact.activate()
        if accel:
            self.device.emit(uinput.ABS_Y, accel, syn=True)
        if brake:
            self.device.emit(uinput.ABS_Z, brake, syn=True)

        self.device.emit(uinput.ABS_X, angle, syn=True)

    def write_in_console(self, text):
        # self.winact.activate()
        time.sleep(0.1)
        self.device.emit_click((1, 41), syn=True)
        time.sleep(0.1)
        for letter in text:
            self.device.emit_click(uinput._CHAR_MAP[letter], syn=True)
        time.sleep(0.1)
        self.device.emit_click(uinput.KEY_ENTER, syn=True)

    def get_speed(self):
        with open("/home/jarcyk/.steam/steam/steamapps/common/Euro Truck Simulator 2/bin/linux_x64/telemetry.log", "r") as f:
            reading = f.readlines()
            #print(reading)
            params = reading.split(",")
            self.speed = 3.6 * float(params[0])
            self.pause = int(params[1])
            # print("Speed: ",self.speed, "Pause state: ", self.pause)

    def run(self):
        while True:
            if not self._inq.empty():
                # self.get_speed()
                control_vector = self._inq.get()
                #print(control_vector)
                if control_vector[0]:
                    #print("Set steering wheel to angle: ", control_vector[0])
                    self.emit(control_vector[0], control_vector[1], control_vector[2])