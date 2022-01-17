import numpy as np
import cv2


# 读取一张图片，并将它转换成灰度图
def getGrayImage(address: str) -> np.ndarray:
    img = cv2.imread(address)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
