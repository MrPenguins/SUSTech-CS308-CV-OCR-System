import cv2
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
    t = image_character_segmentation("Sample_Picture.png")
    print(t)
    rectangle_characters("Sample_Picture.png", t)


if __name__ == '__main__':
    main()
