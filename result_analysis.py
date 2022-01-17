import difflib


def accuracy(text1: str, text2: str) -> float:
    sequence = difflib.SequenceMatcher(None, text1, text2)
    return sequence.ratio()


# print(accuracy("asdfghjkl", "asdfhjkl"))
