import os
import random

from PIL import Image, ImageFont, ImageDraw

FONT_SIZE_RANGE = [30, 30]
IMAGE_WIDTH_RANGE = [1000, 1000]
IMAGE_HEIGHT_RANGE = [1000, 1000]
LINE_SPACING_RANGE = [30, 30]

FONT_DICT = ["Lato-Black.ttf"]


def get_random_font_addr():
    return FONT_DICT[random.randint(0, len(FONT_DICT) - 1)]


def generate_test_image():
    current_path = os.getcwd()
    font_size = random.randint(FONT_SIZE_RANGE[0], FONT_SIZE_RANGE[1])
    image_size = (random.randint(IMAGE_WIDTH_RANGE[0], IMAGE_WIDTH_RANGE[1]),
                  random.randint(IMAGE_HEIGHT_RANGE[0], IMAGE_HEIGHT_RANGE[1]))
    image = Image.new("RGB", image_size, (255, 255, 255))
    image_draw = ImageDraw.Draw(image)
    with open('./test_txt/test.txt') as textFile:
        lines = textFile.readlines()
    line_location = 20
    for i in range(0, len(lines)):
        font = ImageFont.truetype(os.path.join('./Font', get_random_font_addr()), font_size)
        image_draw.text([10, line_location], lines[i], font=font, fill="#000000")
        line_location += font_size + random.randint(LINE_SPACING_RANGE[0], LINE_SPACING_RANGE[1])
    image.save('test.png')


generate_test_image()
