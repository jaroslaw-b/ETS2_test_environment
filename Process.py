import multiprocessing, cv2
import numpy as np
import time

class Process(multiprocessing.Process):

    def region_of_interest(self, img, vertices):  # not
        mask = np.zeros_like(img)
        ignore_mask_color = 255

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def get_speed(self):
        with open("/home/" + self.user_name + "/.steam/steam/steamapps/common/Euro Truck Simulator 2/bin/linux_x64/telemetry.log", "r") as f:
            reading = f.readlines()
            params = reading[0].split(",")
            self.speed = 3.6 * float(params[0])
            self.pause = int(params[1])
            # print("Speed: ", int(self.speed), "Pause state: ", self.pause)
            # print(self.pause)

    def process_image(self, image):
        self.get_speed()
        if self.pause == 1:
            return [image, [None, 0, 0]]
        ###############################################
        #user code starts here















        #user code ends here
        ###############################################
        return [image, [None, 0, 0]]


    def __init__(self, inq, outq, _user_name = "lsriw"):
        multiprocessing.Process.__init__(self)
        self._inq = inq
        self._outq = outq
        self.daemon = True
        self.speed = 0
        self.pause = 0
        self.user_name = _user_name

    def run(self):
        while True:
            if not self._inq.empty():
                img = self._inq.get()
                #print("Queue size: ", self._inq.qsize())
                [img, control] = self.process_image(img)

                self._outq.put([img, control])