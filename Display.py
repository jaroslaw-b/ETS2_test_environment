import multiprocessing
import cv2

from PIL import Image


class Display(multiprocessing.Process):

    def __init__(self, inq, outq):
        multiprocessing.Process.__init__(self)
        self._inq = inq
        self._outq = outq
        self.daemon = True

    def run(self):
        while True:
            if not self._inq.empty():
                queue_gets = self._inq.get()                
                cv2.imshow("Display Results", queue_gets[0])
                if cv2.waitKey(25) == ord("q"):
                    break
                self._outq.put(queue_gets[1])
