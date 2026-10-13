"""Formatting helpers and the generated Markdown report.

The report fills in *numbers* automatically but never writes interpretations of real data:
for real studies the "Interpretation" fields are left for the researcher. For synthetic
demo data every section says so explicitly.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from gazegrounded.utils import SYNTHETIC_BANNER, ensure_dir


def fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    return "< .001" if p < 0.001 else f"= {p:.3f}".replace("0.", ".", 1)


def fmt_num(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{digits}f}"


def format_test(res: dict) -> str:
    """One-line APA-like summary, e.g. ``rmcorr r = 0.41 [0.22, 0.57], p < .001, N = 120``."""
    parts = [f"{res.get('method', res.get('test', ''))}"]
    stat_name = {
        "spearman": "rho",
        "rmcorr": "r",
        "spearman_participant_means": "rho",
        "wilcoxon": "W",
        "ttest": "t",
    }.get(res.get("method", res.get("test")), "stat")
    parts.append(f"{stat_name} = {fmt_num(res.get('statistic'))}")
    ci = res.get("ci95")
    if (
        isinstance(ci, list | tuple)
        and len(ci) == 2
        and not any(isinstance(c, float) and math.isnan(c) for c in ci)
    ):
        parts[-1] += f" [{fmt_num(ci[0])}, {fmt_num(ci[1])}]"
    parts.append(f"p {fmt_p(res.get('p_value'))}")
    p_holm = res.get("p_holm")
    if isinstance(p_holm, float) and not math.isnan(p_holm):
        parts.append(f"p_Holm {fmt_p(p_holm)}")
    if isinstance(res.get("effect_size_name"), str):
        parts.append(f"{res['effect_size_name']} = {fmt_num(res.get('effect_size'))}")
    parts.append(f"N = {res.get('n')}")
    return ", ".join(parts)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(
                format(v, floatfmt)
                if isinstance(v, float) and not math.isnan(v)
                else ("n/a" if isinstance(v, float) else str(v))
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


EXPERIMENT_FIELDS = (
    "Hypothesis",
    "Independent variables",
    "Dependent variables",
    "Dataset",
    "Split strategy",
    "Model",
    "Metrics",
    "Result",
    "Interpretation",
    "Limitation",
)


def experiment_block(title: str, fields: dict[str, str], synthetic: bool) -> str:
    """Render one experiment in the fixed template (see reports/experiment_template.md)."""
    lines = [f"### {title}", ""]
    for key in EXPERIMENT_FIELDS:
        value = fields.get(key, "")
        if key == "Interpretation" and not value:
            value = (
                "Not interpretable — synthetic demo data generated with these effects " "built in."
                if synthetic
                else "_To be written by the researcher after inspecting the results._"
            )
        lines.append(f"**{key}.** {value}")
        lines.append("")
    return "\n".join(lines)


def write_report(path: str | Path, title: str, sections: list[str], synthetic: bool) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    header = [
        f"# {title}",
        "",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} by `gazegrounded.cli evaluate`._",
        "",
    ]
    if synthetic:
        header += [
            f"> **{SYNTHETIC_BANNER}.** Every number below comes from simulated data whose "
            "patterns were put there by the generator. They demonstrate that the "
            "pipeline runs; they are not findings.",
            "",
        ]
    p.write_text("\n".join(header + sections) + "\n")
    return p
