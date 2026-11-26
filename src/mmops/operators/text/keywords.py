from __future__ import annotations

import re
from collections.abc import Sequence

import pyarrow as pa

from ..._decorators import batch, operator, row

KEYWORD_MATCH_DTYPE = pa.list_(pa.string())


@operator
class KeywordMatcher:
    """Match a configured keyword set with worker-local compiled regexes.

    Each configured keyword appears at most once in the result, in configuration
    order, even when it occurs several times in the input text.
    """

    def __init__(
        self,
        keywords: Sequence[str],
        *,
        case_sensitive: bool = False,
        whole_word: bool = False,
    ) -> None:
        if isinstance(keywords, str) or not keywords:
            raise ValueError("keywords must contain at least one string")

        self.keywords = tuple(keywords)
        self.case_sensitive = case_sensitive
        self.whole_word = whole_word

        seen: set[str] = set()
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled: list[tuple[str, re.Pattern[str]]] = []
        for keyword in self.keywords:
            if not isinstance(keyword, str) or not keyword:
                raise ValueError("keywords must contain only non-empty strings")
            identity = keyword if case_sensitive else keyword.casefold()
            if identity in seen:
                raise ValueError(f"duplicate keyword under matching rules: {keyword!r}")
            seen.add(identity)

            pattern = re.escape(keyword)
            if whole_word:
                pattern = rf"(?<!\w){pattern}(?!\w)"
            compiled.append((keyword, re.compile(pattern, flags)))
        self._patterns = tuple(compiled)

    def _match_text(self, text: str | None) -> list[str] | None:
        if text is None:
            return None
        return [keyword for keyword, pattern in self._patterns if pattern.search(text)]

    @row(return_dtype=KEYWORD_MATCH_DTYPE)
    def match(self, text: str | None) -> list[str] | None:
        """Return configured keywords found in one text value."""

        return self._match_text(text)

    @batch(return_dtype=KEYWORD_MATCH_DTYPE, batch_size=256)
    def match_batch(self, text: pa.Array) -> pa.Array:
        """Return configured keywords found in each text value.

        The compiled Python regex engine requires Python strings, so this method
        crosses the Arrow scalar boundary once per row without materializing an
        intermediate table or accepting Daft-specific values.
        """

        matches = [self._match_text(value.as_py()) for value in text]
        return pa.array(matches, type=KEYWORD_MATCH_DTYPE)
