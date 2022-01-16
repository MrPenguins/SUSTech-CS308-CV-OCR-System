import numpy as np
import cv2

# 读取一张图片，并将它转换成灰度图
def getGrayImage(address: str):
    img = cv2.imread(address)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def main():
    img = getGrayImage("Sample_Picture.png")
    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(img)


if __name__ == '__main__':
    main()
