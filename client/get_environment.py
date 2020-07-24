import cv2
import numpy as np
import pyautogui
from time import sleep


class GetEnv:
    def __init__(self, inDim, outDim):
        self.inDim = inDim
        self.outDim = outDim

    def takeImage(self, num, processing):
        Freq = 28 / num
        step = 0
        for i in range(28):
            step += 1
            if step % Freq == 0:
                image = pyautogui.screenshot()
                image = cv2.cvtColor(np.array(image),
                                     cv2.COLOR_RGB2GRAY)
                dim = self.outDim
                inDim = self.inDim
                # print(inDim[0], ainDim[1], inDim[2], inDim[3])
                img = cv2.resize(image[inDim[0]:inDim[1], inDim[2]:inDim[3]], dim, interpolation=cv2.INTER_AREA)
                if processing == "inverse":
                    img = cv2.bitwise_not(img)
                return img
            sleep(1 / 28)


