import re
import unicodedata


def normalize_text(text):
    text = str(text).upper().strip()

    # bỏ dấu câu
    text = re.sub(r"[^\w\s]", " ", text)

    # normalize unicode
    text = unicodedata.normalize("NFC", text)

    # bỏ space thừa
    text = re.sub(r"\s+", " ", text).strip()

    return text


def pre_process(text: str):
    text = re.sub(r"[\.,!?;:\"()\[\]{}\-](?!\w)", " ", text) # remove punc, ignore a.b with look ahead (?!\w)
    text = text.upper()
    return text