import re


R_PUNCT_PATTERN = re.compile(r"(\(|¡|¿)(\s+)")
L_PUNCT_PATTERN = re.compile(r"(\s+)(,|\.|:|;|\?|!|\))")


def fix_punctuation(text: str) -> str:
    text = R_PUNCT_PATTERN.sub(r"\1", text)
    return L_PUNCT_PATTERN.sub(r"\2", text)
