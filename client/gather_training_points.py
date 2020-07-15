from get_environment import GetEnv
from time import sleep
import cv2
import pytesseract


class readReward():
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r'C:\\Tesseract-OCR\\tesseract.exe'

    def readPoints(self, inImg):
        img = cv2.imread(inImg)
        string = str(pytesseract.image_to_string)
        if string.isdigit():
            return pytesseract.image_to_string(img)
        else:
            return 0

    def mainLoop(self, inDim, outDim):
        fetchTrain = GetEnv(inDim=inDim, outDim=outDim)
        # sleep(5)
        image = fetchTrain.takeImage(1, "Inverse")
        # print(len(image))
        cv2.imwrite("points.png", image)
        return self.readPoints("points.png")
