from get_environment import GetEnv
from time import sleep
import cv2
import pytesseract


class readReward():
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r'C:\\Tesseract-OCR\\tesseract.exe'
    def containsDigits(self, string):
        for character in string:
            if character.isdigit():
                return True
        return False
    def readPoints(self, inImg):
        img = cv2.imread(inImg)
        string = str(pytesseract.image_to_string(img))
        print(f"string: {string}")
        if not self.containsDigits(string):
            print("123 not in string")
            return 0
        else:
            integer = int(''.join(c for c in string if c.isdigit()))
            return integer

    def mainLoop(self, inDim, outDim):
        fetchTrain = GetEnv(inDim=inDim, outDim=outDim)
        # sleep(5)
        image = fetchTrain.takeImage(1, "inverse")
        # print(len(image))
        cv2.imwrite("points.png", image)
        return self.readPoints("points.png")


