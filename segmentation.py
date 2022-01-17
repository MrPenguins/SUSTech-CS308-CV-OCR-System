import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List

POINT_THRESHOLD = 100  # 小于该值，判定改像素点有文字内容
LINE_HEIGHT_MIN = 5  # 最小行高
CHARACTER_WIDTH_MIN = 1  # 最小字宽
LINE_COUNT_MIN = 1
CHARACTER_COUNT_MIN = 1


def line_projection(image: np.ndarray) -> np.ndarray:
    lp = np.zeros(image.shape[0], dtype=int)
    for i in range(image.shape[0]):
        lp[i] = np.sum(image[i] < POINT_THRESHOLD)
    return lp


def draw_line_projection_graph(lp: np.ndarray):
    plt.barh(range(0, lp.shape[0]), lp, height=1)
    plt.show()


def line_segmentation(lp: np.ndarray):
    lines = []
    find_start = False
    start = 0
    end = 0
    for i in range(lp.shape[0] - 1):
        if not find_start and lp[i] > LINE_COUNT_MIN:
            start = i
            find_start = True
        elif lp[i + 1] < LINE_COUNT_MIN < lp[i] and i - start > LINE_HEIGHT_MIN:
            end = i
            lines.append((start, end))
            find_start = False

    return lines


def character_projection(image: np.ndarray, line: tuple) -> np.ndarray:
    start = line[0]
    end = line[1]
    image_line = image[start:end]
    cp = np.zeros(image_line.shape[1], dtype=int)
    for i in range(image_line.shape[1]):
        cp[i] = np.sum(image_line[:, i] < POINT_THRESHOLD)
    return cp


def draw_character_projection_graph(cp: np.ndarray):
    plt.bar(range(0, cp.shape[0]), cp, width=1)
    plt.show()


def character_segmentation(cp: np.ndarray):
    characters = []
    find_start = False
    start = 0
    for i in range(cp.shape[0] - 1):
        if not find_start and cp[i] > CHARACTER_COUNT_MIN:
            start = i
            find_start = True
        elif cp[i + 1] < CHARACTER_COUNT_MIN < cp[i] and i - start > CHARACTER_WIDTH_MIN:
            end = i
            characters.append((start, end))
            find_start = False

    return characters
