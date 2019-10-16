import VirtualJoystick
import ScreenCapture
import Display
import Process


import multiprocessing
from threading import Thread, Event
import queue
import concurrent.futures
import time

import cv2

capturer2process = multiprocessing.Queue()
process2display = multiprocessing.Queue()
display2control = multiprocessing.Queue()


controller = VirtualJoystick.VirtualJoystick(display2control)
capturer = ScreenCapture.ScreenCapture(controller.ETS_window_coords[1], controller.ETS_window_coords[0], controller.ETS_window_coords[2], controller.ETS_window_coords[3], capturer2process)
process = Process.Process(capturer2process, process2display)
displayer = Display.Display(process2display, display2control)

controller.start()
#
capturer.start()
process.start()
displayer.start()
