import numpy as np
import cv2
import matplotlib.pyplot as plt

POINT_THRESHOLD = 100  # 小于该值，判定改像素点有文字内容
LINE_HEIGHT_MIN = 5  # 最小行高
CHARACTER_WIDTH_MIN = 5  # 最小字宽
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
