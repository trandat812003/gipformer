import re
import unicodedata


def normalize_text(text):
    text = str(text).lower().strip()

    # bỏ dấu câu
    text = re.sub(r"[^\w\s]", " ", text)

    # normalize unicode
    text = unicodedata.normalize("NFC", text)

    # bỏ space thừa
    text = re.sub(r"\s+", " ", text).strip()

    return text
