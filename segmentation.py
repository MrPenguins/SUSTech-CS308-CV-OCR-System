import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List

POINT_THRESHOLD = 100  # 小于该值，判定改像素点有文字内容
LINE_HEIGHT_MIN = 5  # 最小行高 真实识别 5    生成训练集 20
CHARACTER_WIDTH_MIN = 1  # 最小字宽
LINE_COUNT_MIN = 1
CHARACTER_COUNT_MIN = 1
WORD_SPACE_RATIO = 1.5


def line_projection(image: np.ndarray) -> np.ndarray:
    lp = np.zeros(image.shape[0], dtype=int)
    for i in range(image.shape[0]):
        lp[i] = np.sum(image[i] < POINT_THRESHOLD)
    return lp


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
            lines.append((start - 1, end + 1))
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


def character_segmentation(cp: np.ndarray):
    characters = []
    find_start = False
    start = 0
    for i in range(cp.shape[0] - 1):
        if not find_start and cp[i] > CHARACTER_COUNT_MIN:
            start = i
            find_start = True
        elif find_start and cp[i + 1] < CHARACTER_COUNT_MIN and i - start > CHARACTER_WIDTH_MIN:
            end = i
            characters.append((start - 1, end + 1))
            find_start = False
    characters = split_long_character(characters)
    characters = split_words(characters)
    return characters


def split_long_character(characters):
    new_characters = []
    sum = 0
    for character in characters:
        sum += character[1] - character[0]
    avg = sum / len(characters)
    for character in characters:
        if character[1] - character[0] > 2 * avg:
            length = character[1] - character[0]
            new_characters.append((character[0], int(character[0] + length / 2) + 1))
            new_characters.append((int(character[0] + length / 2) - 1, character[1]))
        else:
            new_characters.append((character[0], character[1]))
    return new_characters


def split_words(characters):
    new_characters = []
    sum = 0
    for i in range(len(characters) - 1):
        sum += characters[i + 1][0] - characters[i][1]
    avg = sum / (len(characters) - 1)
    for i in range(len(characters) - 1):
        new_characters.append(characters[i])
        if characters[i + 1][0] - characters[i][1] > avg * WORD_SPACE_RATIO:
            new_characters.append(" ")
    new_characters.append(characters[len(characters) - 1])
    return new_characters
