"""Light text normalization for extracted PDF content."""
from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Collapse noisy whitespace while keeping paragraph breaks."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
