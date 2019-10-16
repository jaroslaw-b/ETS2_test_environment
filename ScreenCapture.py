import mss
import cv2
import numpy as np
import multiprocessing
import time
from PIL import Image

class ScreenCapture(multiprocessing.Process):
    img = None

    def __init__(self, _top, _left, _width, _height, outq):
        self.monitor = {"top": _top+30, "left": _left, "width": _width, "height": _height-30}

        multiprocessing.Process.__init__(self)
        self._outq = outq
        self.deamon = True

    def run(self):
        while True:
            q_size = self._outq.qsize()
            time.sleep(q_size*1)
            start_time = time.time()
            with mss.mss() as sct:
                data = sct.grab(self.monitor)
                sct.close()
            process_time = time.time() - start_time
            #print(process_time)
            image = np.array(data)
            self._outq.put(image)
            # self.img = np.array(sct.grab())

    def show_image(self, queue, event):
        while not event.is_set() or not queue.empty():
            # queue.get().show()
            cv2.imshow("ETS2", queue.get())

