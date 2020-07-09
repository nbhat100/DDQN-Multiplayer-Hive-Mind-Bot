import cv2
import numpy as np
import pyautogui
from time import sleep
from matplotlib import pyplot as plt


class GetEnv:
    def __init__(self, inDim, outDim):
        self.inDim = inDim
        self.outDim = outDim

    def takeImage(self, num, processing):
        out = []
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
                print(inDim[0], inDim[1], inDim[2], inDim[3])
                img = cv2.resize(image[inDim[0]:inDim[1], inDim[2]:inDim[3]], dim, interpolation=cv2.INTER_AREA)
                if processing == "inverse":
                    img = cv2.bitwise_not(img)
                out.append(img)
                print(img.shape)
            sleep(1 / 28)
        print(len(out))
        return out

    def step(self, action):
        if action == 1:
            pyautogui.press('left')
        elif action == 2:
            pyautogui.press('up')
        elif action == 3:
            pyautogui.press('down')
        elif action == 4:
            pyautogui.press('right')
        elif action == 5:
            pyautogui.press('enter')
        else:
            sleep(1 / 28)

