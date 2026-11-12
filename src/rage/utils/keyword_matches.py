"""Locate the original text behind FastEmbed BM25 matches."""

import re

from fastembed.sparse.bm25 import Bm25


def keyword_match_ranges(
    text: str, model: Bm25, query_token_ids: set[int]
) -> list[tuple[int, int]]:
    """Return half-open UTF-16 ranges of words matching the query vector."""
    ranges = []
    previous_end = 0
    utf16_end = 0
    # FastEmbed splits on non-word characters. Keep the original offsets and
    # let the active BM25 model normalize, filter, and stem each word.
    for word in re.finditer(r"\w+", text):
        tokens = model.tokenizer.tokenize(word[0])
        if not any(
            model.compute_token_id(stem) in query_token_ids
            for stem in model._stem(tokens)
        ):
            continue

        start, end = word.span()
        utf16_start = utf16_end + len(text[previous_end:start].encode("utf-16-le")) // 2
        utf16_end = utf16_start + len(word[0].encode("utf-16-le")) // 2
        ranges.append((utf16_start, utf16_end))
        previous_end = end

    return ranges
