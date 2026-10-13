"""Scanpaths as AOI sequences: visits, transitions and simple sequence comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aoi_sequence(fixations: pd.DataFrame, drop_outside: bool = True) -> list[str | None]:
    """AOI id of each fixation in temporal order (``None`` = outside all AOIs)."""
    # Missing AOIs may arrive as None or NaN depending on how the table was built/read.
    seq = [a if pd.notna(a) else None for a in fixations.sort_values("start_ms")["aoi_id"]]
    return [a for a in seq if a is not None] if drop_outside else seq


def collapse_visits(sequence: list[str | None]) -> list[tuple[str | None, int]]:
    """Merge consecutive fixations on the same AOI into visits: ``[(aoi, n_fixations), ...]``.

    ``A A B A`` -> ``[(A, 2), (B, 1), (A, 1)]``: two visits to A, so A has one *revisit*.
    """
    visits: list[tuple[str | None, int]] = []
    for aoi in sequence:
        if visits and visits[-1][0] == aoi:
            visits[-1] = (aoi, visits[-1][1] + 1)
        else:
            visits.append((aoi, 1))
    return visits


def transition_matrix(sequence: list[str | None], aoi_ids: list[str] | None = None) -> pd.DataFrame:
    """Count matrix of transitions between *different* consecutive AOIs (rows = from)."""
    visits = [a for a, _ in collapse_visits(sequence)]
    labels = aoi_ids or sorted({a for a in visits if a is not None})
    mat = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for a, b in zip(visits[:-1], visits[1:], strict=False):
        if a in mat.index and b in mat.columns:
            mat.loc[a, b] += 1
    return mat


def transition_probabilities(counts: pd.DataFrame) -> pd.DataFrame:
    """Row-normalise a transition count matrix (rows with no transitions stay 0)."""
    totals = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(totals, axis=0).fillna(0.0)


def scanpath_length_px(fixations: pd.DataFrame) -> float:
    """Total Euclidean distance between successive fixation centroids."""
    f = fixations.sort_values("start_ms")
    if len(f) < 2:
        return 0.0
    dx = np.diff(f["centroid_x"].to_numpy(dtype=float))
    dy = np.diff(f["centroid_y"].to_numpy(dtype=float))
    return float(np.hypot(dx, dy).sum())


def format_scanpath(
    sequence: list[str | None], labels: dict[str, str] | None = None, max_len: int = 12
) -> str:
    """Readable visit sequence, e.g. ``Header → Order summary → Place order``."""
    names = [
        (labels or {}).get(a, a) if a is not None else "·" for a, _ in collapse_visits(sequence)
    ]
    if len(names) > max_len:
        names = names[: max_len - 1] + ["…", names[-1]]
    return " → ".join(str(n) for n in names)


def levenshtein(a: list, b: list) -> int:
    """Edit distance between two AOI visit sequences (classic string-edit scanpath comparison)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized_scanpath_similarity(a: list, b: list) -> float:
    """``1 - levenshtein / max(len)`` over collapsed visit sequences, in [0, 1]."""
    va = [x for x, _ in collapse_visits(a)]
    vb = [x for x, _ in collapse_visits(b)]
    if not va and not vb:
        return 1.0
    return 1.0 - levenshtein(va, vb) / max(len(va), len(vb))
