from matplotlib import pyplot as plt
import cv2
import numpy as np


def draw_line_projection_graph(lp: np.ndarray):
    plt.barh(range(0, lp.shape[0]), lp, height=1)
    plt.show()


def draw_character_projection_graph(cp: np.ndarray):
    plt.bar(range(0, cp.shape[0]), cp, width=1)
    plt.show()


def rectangle_characters(image_addr: str, character_list):
    character_list=iter(character_list)
    img = cv2.imread(image_addr)
    for character_rec_index in character_list:
        cv2.rectangle(img, (character_rec_index[2], character_rec_index[0]),
                      (character_rec_index[3], character_rec_index[1]), (0, 255, 0), 1)
    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
