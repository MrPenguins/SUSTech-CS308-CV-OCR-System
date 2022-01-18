from translate import translate

import os

from preprocess import *
from segmentation import *
from visulization import *
from result_analysis import *


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

        character_list.append("\n")

    return character_list


def main():
    with open('./test_txt/Alice.txt') as textFile:
        standard_text = textFile.read()
    t = image_character_segmentation("Alice.png")
    print(t)
    rectangle_characters("Alice.png", t)
    image = cv2.imread("Alice.png")
    c = t[0]
    # cv2.namedWindow("Image")
    # cv2.imshow("Image", image[c[0]: c[1], c[2]: c[3]])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    result = ""
    for c in t:
        # cv2.namedWindow("Image")
        # cv2.imshow("Image", resize_character_image(image[c[0]: c[1], c[2]: c[3]]))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if c == "\n":
            result += c
        else:
            cv2.imwrite(os.path.join('./tmp/t.png'), resize_character_image(image[c[0]: c[1], c[2]: c[3]]))
            # TODO call your function to get corresponding letters
            now_letter_char = translate(os.path.join('./tmp/t.png'))
            result += now_letter_char
    print("Original text:")
    print(standard_text)
    print("Result text:")
    print(result)
    print("Accuracy:")
    print(accuracy(standard_text, result))


def resize_character_image(image: np.ndarray) -> np.ndarray:
    m = max(image.shape[0], image.shape[1])
    height_plus = (m - image.shape[0]) // 2 + 1
    width_plus = (m - image.shape[1]) // 2 + 1
    image = np.pad(image, ((height_plus, height_plus), (width_plus, width_plus), (0, 0)), 'constant',
                   constant_values=255)
    return image


if __name__ == '__main__':
    main()
