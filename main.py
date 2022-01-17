import cv2
import os

import numpy as np

from preprocess import *
from segmentation import *
from visulization import *


def image_character_segmentation(image_addr: str) -> list:
    character_list = []
    img = getGrayImage(image_addr)
    lp = line_projection(img)
    lines = line_segmentation(lp)
    for line in lines:
        cp = character_projection(img, line)
        # draw_character_projection_graph(cp)
        characters = character_segmentation(cp)
        for character in characters:
            character_list.append((line[0], line[1], character[0], character[1]))

    return character_list


def main():
    t = image_character_segmentation("Sample2.png")
    print(t)
    rectangle_characters("Sample2.png", t)
    image = cv2.imread("Sample2.png")
    c = t[1]
    # cv2.namedWindow("Image")
    # cv2.imshow("Image", image[c[0]: c[1], c[2]: c[3]])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    for c in t:
        # cv2.namedWindow("Image")
        # cv2.imshow("Image", resize_character_image(image[c[0]: c[1], c[2]: c[3]]))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(os.path.join('./tmp/t.png'), resize_character_image(image[c[0]: c[1], c[2]: c[3]]))
        # TODO call your function to get corresponding letters


def resize_character_image(image: np.ndarray) -> np.ndarray:
    m = max(image.shape[0], image.shape[1])
    height_plus = (m - image.shape[0]) // 2 + 2
    width_plus = (m - image.shape[1]) // 2 + 2
    image = np.pad(image, ((height_plus, height_plus), (width_plus, width_plus), (0, 0)), 'constant',
                   constant_values=255)
    return image


if __name__ == '__main__':
    main()
