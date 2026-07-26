"""Shared text-quality heuristics.

Used by both candidate collection and candidate ranking so a junk line is
rejected wherever it enters, rather than only at the first gate.
"""

from __future__ import annotations

import re

MIN_LENGTH = 5
MIN_ALPHA_RATIO = 0.6
MAX_MIXED_CASE_RATIO = 0.34


def looks_like_words(line: str) -> bool:
    """Reject OCR noise while keeping legitimately messy business names.

    Handwritten pages produce lines like 'fel tolnall Ante latalentl A rer = OI'
    and 'PS eee Z ie Qlva'. Requiring a couple of word-shaped tokens plus
    mostly-alphabetic content filters those without discarding real names such as
    'Ent rek (B) Sdn.Bhd.'.
    """
    stripped = line.strip()
    if len(stripped) < MIN_LENGTH:
        return False

    letters = sum(1 for ch in stripped if ch.isalpha())
    if letters < MIN_LENGTH or letters < len(stripped) * MIN_ALPHA_RATIO:
        return False

    tokens = re.findall(r"[A-Za-z][A-Za-z.'&-]*", stripped)
    solid = [token for token in tokens if len(token) >= 3]
    if len(solid) < 2:
        # A single long word can still be a name ('DETOXICARE'), but a scatter of
        # short fragments is almost always noise.
        return len(solid) == 1 and len(solid[0]) >= 8 and len(tokens) <= 2

    # Noise alternates case mid-word; real headers are caps or title case.
    odd = sum(1 for token in solid if re.search(r"[a-z][A-Z]", token))
    return odd <= len(solid) * MAX_MIXED_CASE_RATIO
