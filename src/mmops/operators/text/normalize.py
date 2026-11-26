from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from ..._decorators import batch

_NORMALIZATION_FORMS = frozenset({"NFC", "NFKC", "NFD", "NFKD"})


@batch(return_dtype=pa.string(), batch_size=1024)
def normalize_text(
    text: pa.Array,
    form: str = "NFKC",
    lowercase: bool = False,
    collapse_whitespace: bool = True,
    strip: bool = True,
) -> pa.Array:
    """Normalize a string array with Arrow compute kernels.

    Null values remain null. Unicode normalization is applied before optional
    lowercasing, whitespace collapsing, and trimming.
    """

    normalized_form = form.upper()
    if normalized_form not in _NORMALIZATION_FORMS:
        supported = ", ".join(sorted(_NORMALIZATION_FORMS))
        raise ValueError(
            f"unsupported Unicode normalization form {form!r}; "
            f"expected one of: {supported}"
        )

    result = pc.utf8_normalize(text, form=normalized_form)
    if lowercase:
        result = pc.utf8_lower(result)
    if collapse_whitespace:
        result = pc.replace_substring_regex(
            result,
            pattern=r"\s+",
            replacement=" ",
        )
    if strip:
        result = pc.utf8_trim_whitespace(result)
    return result
