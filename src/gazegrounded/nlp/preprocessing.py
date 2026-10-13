"""Light-touch text cleaning for short post-task feedback.

Deliberately minimal: lowercase, normalise quotes/whitespace, drop punctuation but KEEP
apostrophes and negations ("didn't", "not"), and do not remove stop-words or stem. In short UX
comments, words like "not", "no" and "where" carry the meaning ("not where I expected").
"""

from __future__ import annotations

import re

import pandas as pd

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
_PUNCT = re.compile(r"[^\w\s']")
_SPACE = re.compile(r"\s+")

# Token pattern for vectorisers: words incl. internal apostrophes ("didn't"), length >= 1.
TOKEN_PATTERN = r"(?u)\b\w[\w']*\b"


def clean_text(text: object) -> str:
    """Return a cleaned string ('' for missing values)."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).translate(_QUOTES).lower()
    s = _PUNCT.sub(" ", s)
    s = re.sub(r"(?<!\w)'|'(?!\w)", " ", s)  # strip quote marks but keep "didn't"
    return _SPACE.sub(" ", s).strip()


def prepare_feedback(trials: pd.DataFrame) -> pd.DataFrame:
    """Add ``feedback_clean`` and ``has_feedback`` columns."""
    out = trials.copy()
    out["feedback_clean"] = out["feedback_text"].map(clean_text)
    out["has_feedback"] = out["feedback_clean"].str.len() > 0
    return out
