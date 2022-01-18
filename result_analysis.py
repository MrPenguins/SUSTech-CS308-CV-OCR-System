import difflib


def accuracy(text1: str, text2: str) -> float:
    text1 = text1.lower()
    text2 = text2.lower()

    sequence = difflib.SequenceMatcher(None, text1, text2)
    return sequence.ratio()


# def generate_test_image

# print(accuracy("asdfghjkl", "asdfhjkl"))
