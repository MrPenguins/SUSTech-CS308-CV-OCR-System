import difflib
import Levenshtein

def accuracy(text1: str, text2: str) -> float:
    text1 = text1.lower()
    text1 = text1.replace(" ", "")
    text1 = text1.replace("\n", "")
    text2 = text2.lower()
    text2 = text2.replace("\n","")
    text2 = text2.replace(" ","")

    # sequence = difflib.SequenceMatcher(None, text1, text2).quick_ratio()
    sequence = Levenshtein.ratio(text2,text1)

    return sequence


# def generate_test_image

# print(accuracy("asdfghjkl", "asdfhjkl"))
