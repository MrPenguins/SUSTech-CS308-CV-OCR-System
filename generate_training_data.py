import os
import random
import cv2

from PIL import Image, ImageFont, ImageDraw
from main import image_character_segmentation, resize_character_image

# 训练参数1
FONT_SIZE_RANGE = [30, 30] #
IMAGE_WIDTH_RANGE = [28, 28]
IMAGE_HEIGHT_RANGE = [28, 28]
LETTER_LOCATION_RANGE_HOR = [-2, 16]
LETTER_LOCATION_RANGE_VER = [-7, 2]

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
# FONT_SIZE_RANGE = [50, 50]  #
# IMAGE_WIDTH_RANGE = [200, 200]
# IMAGE_HEIGHT_RANGE = [200, 200]
# LETTER_LOCATION_RANGE_HOR = [50, 50]
# LETTER_LOCATION_RANGE_VER = [50, 50]

FONT_ADDR = os.path.join('./Font')


def get_random_font_addr():
    for root, dirs, files in os.walk(FONT_ADDR):
        return os.path.join(root, files[random.randint(0, len(files) - 1)])


def generate_character_image(letter: str, font_size: int, image_size: tuple, letter_location: tuple):
    image = Image.new("RGB", image_size, (255, 255, 255))
    image_draw = ImageDraw.Draw(image)
    font_addr = get_random_font_addr()
    font = ImageFont.truetype(font_addr, font_size)
    print(font_addr)
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
        path = os.path.join(current_path, 'datasets/xun1_duoziti/Test_png', chr(i), 'u' + str(epoch) + '.png')
        image.save(path)
        # training_data_process(path)

    for i in range(97, 123):
        image = generate_character_image(chr(i), font_size, image_size, letter_location)
        path = os.path.join(current_path, 'datasets/xun1_duoziti/Test_png', chr(i - 32), 'l' + str(epoch) + '.png')
        image.save(path)
        # training_data_process(path)


def training_data_process(image_path: os.path):
    image = cv2.imread(image_path)
    cl = image_character_segmentation(image_path)
    print(len(cl))
    print(image_path)
    location = cl[0]
    cv2.imwrite(image_path,
                cv2.resize((resize_character_image(image[location[0]: location[1], location[2]: location[3]])),
                           (28, 28)))


def generate_folders():
    current_path = os.getcwd()
    for i in range(65, 91):
        os.mkdir(current_path + '/datasets/xun1_duoziti/Test_png/' + chr(i))

    # for i in range(65, 91):
    #     os.mkdir(current_path + '\\data\\UpperCase\\' + chr(i))
    #
    # for i in range(97, 123):
    #     os.mkdir(current_path + '\\data\\LowerCa se\\' + chr(i))


generate_folders()
for i in range(0, 200):
    generate_random_image(i)
