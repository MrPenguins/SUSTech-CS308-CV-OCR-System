import cv2
import os
from preprocess import *
from segmentation import *
from visulization import *

from translate import translate

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
    t = image_character_segmentation("test2.png")
    print(t)
    rectangle_characters("test2.png", t)
    image = cv2.imread("test2.png")
    c = t[1]
    # cv2.namedWindow("Image")
    # cv2.imshow("Image", image[c[0]: c[1], c[2]: c[3]])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    for c in t:
        # cv2.namedWindow("Image")
        # cv2.imshow("Image", image[c[0]: c[1], c[2]: c[3]])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(os.path.join('./tmp/t.png'), image[c[0]: c[1], c[2]: c[3]])
        # TODO call your function to get corresponding letters
        now_letter_char = translate(os.path.join('./tmp/t.png'))
        print(now_letter_char,end="")

if __name__ == '__main__':
    main()
