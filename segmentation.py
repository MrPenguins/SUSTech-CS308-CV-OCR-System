import numpy as np
import cv2
import matplotlib.pyplot as plt

THRESHOLD = 100


def line_projection(image: np.ndarray) -> np.ndarray:
    lp = np.zeros(image.shape[0], dtype=int)
    for i in range(image.shape[0]):
        lp[i] = np.sum(image[i] < THRESHOLD)
    return lp


def draw_line_projection_graph(lp: np.ndarray):
    plt.barh(range(0, lp.shape[0]), lp, height=1)
    plt.show()
