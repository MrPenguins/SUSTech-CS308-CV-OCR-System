import os
import random

from PIL import Image, ImageFont, ImageDraw

# 训练参数1
# FONT_SIZE_RANGE = [30, 30] #
# IMAGE_WIDTH_RANGE = [28, 28]
# IMAGE_HEIGHT_RANGE = [28, 28]
# LETTER_LOCATION_RANGE_HOR = [-2, 16]
# LETTER_LOCATION_RANGE_VER = [-7, 2]

# FONT_SIZE_RANGE = [30, 30] #
# IMAGE_WIDTH_RANGE = [28, 28]
# IMAGE_HEIGHT_RANGE = [28, 28]
# LETTER_LOCATION_RANGE_HOR = [-2, 16]
# LETTER_LOCATION_RANGE_VER = [-7, 2]

# 这组参数效果好 基本上填满了整个框
# FONT_SIZE_RANGE = [32, 32] #
# IMAGE_WIDTH_RANGE = [28, 28]
# IMAGE_HEIGHT_RANGE = [28, 28]
# LETTER_LOCATION_RANGE_HOR = [4, 4]
# LETTER_LOCATION_RANGE_VER = [-7, -7]

# 上面的参数加入随机性 训练参数2
FONT_SIZE_RANGE = [32, 32]  #
IMAGE_WIDTH_RANGE = [28, 28]
IMAGE_HEIGHT_RANGE = [28, 28]
LETTER_LOCATION_RANGE_HOR = [2, 6]
LETTER_LOCATION_RANGE_VER = [-9, -5]

FONT_DICT = ["TimesNewRoman.ttf", "Lato/Lato-Black.ttf"]


def get_random_font():
    return FONT_DICT[random.randint(0, len(FONT_DICT) - 1)]


def generate_character_image(letter: str, font_size: int, image_size: tuple, letter_location: tuple):
    image = Image.new("RGB", image_size, (255, 255, 255))
    image_draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("./Font/Lato/Lato-Black.ttf", font_size)
    image_draw.text(letter_location, letter, font=font, fill="#000000")
    # image.show()
    # image.save("t.png")
    return image


def generate_random_image(epoch: int):
    current_path = os.getcwd()
    font_size = random.randint(FONT_SIZE_RANGE[0], FONT_SIZE_RANGE[1])
    image_size = (random.randint(IMAGE_WIDTH_RANGE[0], IMAGE_WIDTH_RANGE[1]),
                  random.randint(IMAGE_HEIGHT_RANGE[0], IMAGE_HEIGHT_RANGE[1]))
    letter_location = (random.randint(LETTER_LOCATION_RANGE_HOR[0], LETTER_LOCATION_RANGE_HOR[1]),
                       random.randint(LETTER_LOCATION_RANGE_VER[0], LETTER_LOCATION_RANGE_VER[1]))
    for i in range(65, 91):
        image = generate_character_image(chr(i), font_size, image_size, letter_location)
        path = os.path.join(current_path, 'datasets/two_104000/Test_png', chr(i), 'u' + str(epoch) + '.png')
        image.save(path)

    for i in range(97, 123):
        image = generate_character_image(chr(i), font_size, image_size, letter_location)
        path = os.path.join(current_path, 'datasets/two_104000/Test_png', chr(i - 32), 'l' + str(epoch) + '.png')
        image.save(path)


def generate_folders():
    current_path = os.getcwd()
    for i in range(65, 91):
        os.mkdir(current_path + '/datasets/two_104000/Test_png/' + chr(i))

    # for i in range(65, 91):
    #     os.mkdir(current_path + '\\data\\UpperCase\\' + chr(i))
    #
    # for i in range(97, 123):
    #     os.mkdir(current_path + '\\data\\LowerCase\\' + chr(i))


generate_folders()
for i in range(0, 200):
    generate_random_image(i)
