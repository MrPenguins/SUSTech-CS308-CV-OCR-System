import cv2
from preprocess import *
from segmentation import *


def main():
    img = getGrayImage("Sample_Picture.png")
    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(img)
    lp=line_projection(img)
    draw_line_projection_graph(lp)


if __name__ == '__main__':
    main()
