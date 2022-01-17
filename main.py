import cv2
from preprocess import *
from segmentation import *


def main():
    img = getGrayImage("Sample_Picture.png")
    # cv2.namedWindow("Image")
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    lp = line_projection(img)
    draw_line_projection_graph(lp)
    lines = line_segmentation(lp)
    print(lines)
    cp = character_projection(img, lines[2])
    draw_character_projection_graph(cp)
    characters=character_segmentation(cp)
    print(len(characters))


if __name__ == '__main__':
    main()
