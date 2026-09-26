from __future__ import annotations

"""
Interpretation-facing report figures built from existing result CSVs.

This module intentionally reads from ``results/*.csv`` instead of recomputing
analysis. It keeps polished, report-oriented figures separate from diagnostic
pipeline plots and writes a small manifest describing each generated figure.
"""

import argparse
import ast
import re
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from .logging_utils import get_logger

logger = get_logger(__name__)

INK = "#1f2933"
MUTED = "#64748b"
GRID = "#e5e7eb"
BLUE = "#2563eb"
TEAL = "#0f766e"
RUST = "#b45309"
RED = "#b91c1c"
GREEN = "#15803d"
AMBER = "#d97706"
GRAY = "#94a3b8"
PALE = "#f8fafc"

PORTFOLIO_FIGURE_FILES = {
    "report_portfolio_aging_rejuvenation.png": "aging_rejuvenation_signal.png",
    "report_portfolio_multimodal_evidence.png": "multimodal_evidence_architecture.png",
    "report_portfolio_estimability_guardrail.png": "estimability_guardrail.png",
}


@dataclass(frozen=True)
class FigureRecord:
    figure: str
    path: Path
    source_tables: str
    status: str
    message: str


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def _read_table(results_dir: Path, filename: str) -> pd.DataFrame:
    path = results_dir / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return pd.DataFrame()


def _bool_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    values = df[column]
    if values.dtype == bool:
        return values.fillna(default)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _numeric(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _first_text(df: pd.DataFrame, column: str, default: str = "") -> str:
    if df.empty or column not in df.columns:
        return default
    values = df[column].dropna().astype(str).str.strip()
    values = values.loc[values != ""]
    return str(values.iloc[0]) if not values.empty else default


def _save(fig: plt.Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _status_figure(
    *,
    out_path: Path,
    title: str,
    message: str,
    source: str,
) -> FigureRecord:
    _set_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axis("off")
    ax.add_patch(
        plt.Rectangle((0.03, 0.18), 0.94, 0.64, transform=ax.transAxes, color=PALE, ec=GRID)
    )
    ax.text(0.06, 0.72, title, transform=ax.transAxes, fontsize=15, weight="bold", color=INK)
    ax.text(
        0.06,
        0.5,
        message,
        transform=ax.transAxes,
        fontsize=10.5,
        color=INK,
        va="center",
        wrap=True,
    )
    ax.text(0.06, 0.22, f"Source: {source}", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    _save(fig, out_path)
    return FigureRecord(out_path.name, out_path, source, "status", message)


def _draw_card(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    edge_color: str = BLUE,
    face_color: str = PALE,
    title_color: str = INK,
    body_color: str = MUTED,
    body_offset: float = 0.13,
) -> None:
    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=face_color,
        edgecolor=edge_color,
        linewidth=1.5,
    )
    ax.add_patch(card)
    ax.text(
        x + 0.018,
        y + height - 0.055,
        title,
        transform=ax.transAxes,
        color=title_color,
        fontsize=9.2,
        weight="bold",
        va="top",
    )
    ax.text(
        x + 0.018,
        y + height - body_offset,
        body,
        transform=ax.transAxes,
        color=body_color,
        fontsize=8.0,
        va="top",
        linespacing=1.25,
    )


def _draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    linestyle: str = "-",
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": 1.4,
            "linestyle": linestyle,
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )


def _source_label(*names: str) -> str:
    return ";".join(names)


def _display_reason_code(value: object) -> str:
    text = str(value or "").strip()
    labels = {
        "OK": "OK",
        "LOW_SAMPLE_SIZE_CAUTION": "low sample size",
        "CONFIG_DISABLED": "disabled",
        "PLASMA_BULK_ANIMAL_LINKAGE_MISSING": "plasma-bulk link missing",
        "PLASMA_LINKAGE_CONFIDENCE_MISSING": "linkage confidence missing",
        "PLASMA_ANIMAL_LINKAGE_COLLISION": "plasma-animal linkage collision",
        "INSUFFICIENT_LINKED_ANIMALS": "insufficient linked animals",
        "OMIX007582_SENTRIX_SAMPLE_SHEET_MISSING": "Mammal40 sample sheet missing",
        "OMIX009284_PBMC_ONLY_FOR_ATTRIBUTION": "PBMC-only for attribution",
        "DATA_FILE_UNAVAILABLE": "source unavailable",
        "NON_ESTIMABLE_UNSPECIFIED": "not estimable",
    }
    return labels.get(text, text.replace("_", " ").lower())


def _display_author_key(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    parts = [part.strip() for part in text.split(";") if part.strip()]
    labels = []
    for part in parts:
        if part in {"not_required", ""}:
            labels.append("not required")
        elif "omix007581" in part:
            labels.append("plasma sample -> animal key")
        elif "omix007580" in part:
            labels.append("bulk sample -> animal/tissue key")
        elif "omix007582" in part or "sentrix" in part.lower():
            labels.append("Mammal40 Sentrix sample sheet")
        elif "exosome_preparation" in part:
            labels.append("exosome donor/cargo/recipient map")
        elif "direct_exosome" in part:
            labels.append("direct exosome uptake/cargo-transfer key")
        elif "qc_exclusion" in part:
            labels.append("QC/exclusion tables")
        elif "treatment_era" in part:
            labels.append("stage vs treatment-era control key")
        elif part == "review_required":
            labels.append("manual review required")
        else:
            labels.append(part.replace("_", " "))
    return " + ".join(dict.fromkeys(labels))


def _wrapped(value: object, width: int) -> str:
    return textwrap.fill(str(value or ""), width=width, break_long_words=False)


def _parse_ci(value: object) -> tuple[float, float]:
    if value is None:
        return np.nan, np.nan
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)) and len(value) >= 2:
        return (
            float(pd.to_numeric(pd.Series([value[0]]), errors="coerce").iloc[0]),
            float(pd.to_numeric(pd.Series([value[1]]), errors="coerce").iloc[0]),
        )
    if pd.isna(value):
        return np.nan, np.nan
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return np.nan, np.nan
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            return float(parsed[0]), float(parsed[1])
    except (SyntaxError, ValueError, TypeError):
        pass
    matches = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)
    if len(matches) >= 2:
        return float(matches[0]), float(matches[1])
    return np.nan, np.nan


def _ci_crosses_zero(low: float, high: float) -> bool:
    return bool(np.isfinite(low) and np.isfinite(high) and low <= 0 <= high)


def _numeric_max(df: pd.DataFrame, column: str, default: float = 0.0) -> float:
    values = _numeric(df, column, default=np.nan).dropna()
    if values.empty:
        return default
    return float(values.max())


def _is_named_protein(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text and text != "--" and text.lower() not in {"nan", "none"})


def _protein_category(protein: object) -> str:
    symbol = str(protein or "").strip().upper()
    if not symbol or symbol == "--":
        return "Unannotated"
    if symbol.startswith("COL") or symbol in {
        "POSTN",
        "FN1",
        "TNC",
        "THBS1",
        "SPARC",
        "VCAN",
        "LAMA1",
        "LAMA2",
        "LAMA3",
        "LAMA4",
        "LAMA5",
        "LAMB1",
        "LAMB2",
        "LAMB3",
    }:
        return "Extracellular matrix"
    if (
        symbol.startswith(("IL", "CCL", "CXCL", "HLA", "S100", "CD"))
        or symbol in {"B2M", "BPI", "CRP", "MIF", "TNF", "TNFAIP3"}
    ):
        return "Immune/inflammatory"
    if (
        symbol.startswith(("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "CF", "SERPIN"))
        or symbol in {"F2", "FGA", "FGB", "FGG", "PLG", "PROC", "PROS1", "VWF"}
    ):
        return "Complement/coagulation"
    if symbol.startswith(("PRDX", "GPX", "SOD")) or symbol in {"HMOX1", "TXN", "TXNDC5", "CAT"}:
        return "Oxidative stress"
    if symbol.startswith(("ACT", "TUB", "MYH", "MYL", "KRT")) or symbol in {"VIM", "MVP", "CAPN2"}:
        return "Cytoskeleton/adhesion"
    if (
        symbol.startswith(("MAPK", "AKT", "PRK", "RHO", "SIRT"))
        or symbol in {"MTOR", "ROCK1", "FOXO3", "FOXO3A", "INSR", "IGF1R"}
    ):
        return "Signaling/metabolism"
    return "Other named proteins"


def _add_effect_uncertainty_for_plot(work: pd.DataFrame, effect_col: str = "effect") -> pd.DataFrame:
    out = work.copy()
    effect = pd.to_numeric(out[effect_col], errors="coerce")
    ci_low = pd.to_numeric(out.get("ci_low_num", out.get("ci_low")), errors="coerce")
    ci_high = pd.to_numeric(out.get("ci_high_num", out.get("ci_high")), errors="coerce")
    ci_width = ci_high - ci_low
    finite_ci = ci_low.notna() & ci_high.notna() & (ci_width > 0)

    if "ci_crosses_zero" in out.columns:
        crosses_zero = _bool_series(out, "ci_crosses_zero", default=False)
    else:
        crosses_zero = finite_ci & (ci_low <= 0) & (ci_high >= 0)

    if "signal_to_uncertainty" in out.columns:
        score = pd.to_numeric(out["signal_to_uncertainty"], errors="coerce")
    else:
        score = effect.abs() / ci_width.replace(0, np.nan)
        score = score.where(finite_ci)

    direction = pd.Series("unresolved", index=out.index, dtype="object")
    direction.loc[effect < 0] = "younger_shift"
    direction.loc[effect > 0] = "older_shift"
    if "effect_direction" in out.columns:
        existing = out["effect_direction"].dropna().astype(str)
        direction.loc[existing.index] = existing

    out["ci_crosses_zero_plot"] = crosses_zero.fillna(False)
    out["signal_to_uncertainty_plot"] = score
    out["effect_direction_plot"] = direction
    return out


def _select_primary_treatment_contrast(df: pd.DataFrame) -> pd.DataFrame:
    """Select the prespecified SRC-versus-vehicle rows when contrasts are explicit."""
    if "is_primary" in df.columns:
        return df.loc[_bool_series(df, "is_primary", default=False)].copy()
    if "contrast" in df.columns:
        labels = df["contrast"].astype(str)
        primary = labels.isin({"SRC_vs_vehicle", "GES_vs_Veh"})
        if primary.any():
            return df.loc[primary].copy()
    return df.copy()


def plot_tissue_rejuvenation_forest(results_dir: Path, out_dir: Path) -> FigureRecord:
    source = "rejuvenation_by_tissue.csv"
    df = _read_table(results_dir, source)
    out_path = out_dir / "report_tissue_rejuvenation_forest.png"
    if df.empty:
        return _status_figure(
            out_path=out_path,
            title="Tissue Rejuvenation Summary",
            message="No tissue-level rejuvenation table was available.",
            source=source,
        )

    work = df.loc[_bool_series(df, "estimable", default=True)].copy()
    work = _select_primary_treatment_contrast(work)
    effect_col = "mean_effect" if "mean_effect" in work.columns else "effect_median"
    if effect_col not in work.columns or "tissue" not in work.columns:
        return _status_figure(
            out_path=out_path,
            title="Tissue Rejuvenation Summary",
            message="The tissue table is missing tissue or effect columns required for a forest plot.",
            source=source,
        )

    work["effect"] = _numeric(work, effect_col)
    work["ci_low_num"] = _numeric(work, "ci_low")
    work["ci_high_num"] = _numeric(work, "ci_high")
    work["n_used_num"] = _numeric(work, "n_used", default=0).fillna(0).astype(int)
    work = _add_effect_uncertainty_for_plot(work, effect_col="effect")
    work = work.dropna(subset=["effect"]).sort_values("effect")
    if work.empty:
        reason = _first_text(df, "reason_code", "NO_ESTIMABLE_FEATURES")
        key = _first_text(df, "missing_author_key", "not_required")
        return _status_figure(
            out_path=out_path,
            title="Tissue Rejuvenation Summary",
            message=f"No estimable tissue effect rows. reason_code={reason}; missing_author_key={key}",
            source=source,
        )

    _set_style()
    height = max(4.8, 0.38 * len(work) + 2.0)
    fig, ax = plt.subplots(figsize=(8.8, height))
    y = np.arange(len(work))
    effects = work["effect"].to_numpy(float)
    crosses_zero = work["ci_crosses_zero_plot"].astype(bool).to_numpy()
    sign_colors = np.where(effects < 0, TEAL, RUST)
    face_colors = np.where(crosses_zero, "white", sign_colors)
    ax.scatter(effects, y, s=58, color=face_colors, edgecolors=sign_colors, linewidths=1.1, zorder=3)
    if work["ci_low_num"].notna().any() and work["ci_high_num"].notna().any():
        low_err = effects - work["ci_low_num"].to_numpy(float)
        high_err = work["ci_high_num"].to_numpy(float) - effects
        ok = np.isfinite(low_err) & np.isfinite(high_err)
        ax.errorbar(
            effects[ok],
            y[ok],
            xerr=np.vstack([low_err[ok], high_err[ok]]),
            fmt="none",
            ecolor=MUTED,
            elinewidth=1.3,
            capsize=3,
            zorder=2,
        )
    ax.axvline(0, color=INK, linewidth=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{row.tissue}  (n={int(row.n_used_num)})"
            for row in work[["tissue", "n_used_num"]].itertuples(index=False)
        ]
    )
    ax.set_xlabel("Treatment effect on delta age (years; negative = younger)")
    ax.set_title("Tissue-Level Delta-Age Effects: SRC vs Vehicle")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.0,
        -0.16,
        "Open markers have pointwise 95% bootstrap intervals crossing zero; multiplicity-adjusted p-values remain in the source table.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8.5,
    )
    _save(fig, out_path)
    n_uncertain = int(crosses_zero.sum())
    return FigureRecord(
        out_path.name,
        out_path,
        source,
        "ok",
        f"{len(work)} tissues plotted; {n_uncertain} CIs cross zero",
    )


def plot_top_tissue_priority(results_dir: Path, out_dir: Path, top_n: int = 8) -> FigureRecord:
    source = "rejuvenation_by_tissue.csv"
    df = _read_table(results_dir, source)
    out_path = out_dir / "report_top_tissue_priority.png"
    if df.empty:
        return _status_figure(
            out_path=out_path,
            title="Tissue Prioritization",
            message="No tissue-level rejuvenation table was available.",
            source=source,
        )

    work = df.loc[_bool_series(df, "estimable", default=True)].copy()
    work = _select_primary_treatment_contrast(work)
    effect_col = "mean_effect" if "mean_effect" in work.columns else "effect_median"
    if effect_col not in work.columns or "tissue" not in work.columns:
        return _status_figure(
            out_path=out_path,
            title="Tissue Prioritization",
            message="The tissue table is missing tissue or effect columns required for prioritization.",
            source=source,
        )

    work["effect"] = _numeric(work, effect_col)
    work["ci_low_num"] = _numeric(work, "ci_low")
    work["ci_high_num"] = _numeric(work, "ci_high")
    work["n_used_num"] = _numeric(work, "n_used", default=0).fillna(0).astype(int)
    work = _add_effect_uncertainty_for_plot(work, effect_col="effect")
    work = work.dropna(subset=["effect"])
    if work.empty:
        reason = _first_text(df, "reason_code", "NO_ESTIMABLE_FEATURES")
        return _status_figure(
            out_path=out_path,
            title="Tissue Prioritization",
            message=f"No finite estimable tissue effects were available. reason_code={reason}",
            source=source,
        )

    top_n = max(1, int(top_n))
    work["priority_for_plot"] = work["signal_to_uncertainty_plot"].fillna(work["effect"].abs())
    younger_candidates = (
        work.loc[work["effect"] < 0]
        .sort_values(["priority_for_plot", "effect"], ascending=[False, True])
        .head(top_n)
        .sort_values("effect")
    )
    older_candidates = (
        work.loc[work["effect"] > 0]
        .sort_values(["priority_for_plot", "effect"], ascending=[False, False])
        .head(top_n)
        .sort_values("effect", ascending=False)
    )

    _set_style()
    total_rows = max(len(younger_candidates) + len(older_candidates), 1)
    fig, axes = plt.subplots(2, 1, figsize=(10.8, max(7.2, 0.42 * total_rows + 2.4)))
    panels = [
        (axes[0], younger_candidates, "Nominal younger-shifted candidates", TEAL),
        (axes[1], older_candidates, "Nominal older-shifted candidates", RUST),
    ]
    for ax, subset, title, color in panels:
        y = np.arange(len(subset))
        effects = subset["effect"].to_numpy(float)
        crosses_zero = subset["ci_crosses_zero_plot"].astype(bool).to_numpy()
        bar_colors = np.where(crosses_zero, GRAY, color)
        bars = ax.barh(y, effects, color=bar_colors, alpha=0.88)
        for bar, crosses in zip(bars, crosses_zero):
            if crosses:
                bar.set_edgecolor(color)
                bar.set_linewidth(1.1)
        if subset["ci_low_num"].notna().any() and subset["ci_high_num"].notna().any():
            low_err = effects - subset["ci_low_num"].to_numpy(float)
            high_err = subset["ci_high_num"].to_numpy(float) - effects
            ok = np.isfinite(low_err) & np.isfinite(high_err)
            ax.errorbar(
                effects[ok],
                y[ok],
                xerr=np.vstack([low_err[ok], high_err[ok]]),
                fmt="none",
                ecolor=INK,
                elinewidth=1.1,
                capsize=2.5,
                zorder=3,
            )
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{row.tissue} (n={int(row.n_used_num)})" for row in subset.itertuples(index=False)])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Delta-age treatment effect")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.subplots_adjust(hspace=0.55)
    fig.suptitle("Tissue Prioritization: Nominal Effects Ranked by Precision", x=0.02, ha="left", fontsize=15, weight="bold")
    fig.text(
        0.02,
        0.02,
        "Rank uses |effect| / CI width; grey bars cross zero and should not be treated as tissue-specific evidence.",
        color=MUTED,
        fontsize=8.5,
    )
    _save(fig, out_path)
    n_uncertain = int(
        pd.concat(
            [
                younger_candidates["ci_crosses_zero_plot"].astype(bool),
                older_candidates["ci_crosses_zero_plot"].astype(bool),
            ]
        ).sum()
    )
    return FigureRecord(
        out_path.name,
        out_path,
        source,
        "ok",
        f"{len(younger_candidates)} younger-shifted and {len(older_candidates)} older-shifted candidates plotted; {n_uncertain} CIs cross zero",
    )


def plot_exosome_alignment_summary(results_dir: Path, out_dir: Path) -> FigureRecord:
    source = "exosome_alignment_summary.csv"
    df = _read_table(results_dir, source)
    out_path = out_dir / "report_exosome_alignment_summary.png"
    if df.empty:
        return _status_figure(
            out_path=out_path,
            title="Exosome-Alignment Evidence",
            message="No exosome-alignment summary table was available.",
            source=source,
        )

    estimable = df.loc[_bool_series(df, "estimable", default=False)].copy()
    if estimable.empty:
        reason = _first_text(df, "reason_code", "NON_ESTIMABLE_UNSPECIFIED")
        key = _first_text(df, "missing_author_key", "review_required")
        return _status_figure(
            out_path=out_path,
            title="Exosome-Alignment Evidence",
            message=(
                "Exosome alignment is not estimable from the current result tables. "
                f"reason_code={reason}; missing_author_key={key}"
            ),
            source=source,
        )

    estimable["similarity"] = _numeric(estimable, "mean_standardized_effect_similarity")
    estimable["ci_low_num"] = _numeric(estimable, "ci_low")
    estimable["ci_high_num"] = _numeric(estimable, "ci_high")
    estimable["spearman_rho_num"] = _numeric(estimable, "spearman_rho")
    estimable["signed_concordance"] = _numeric(estimable, "fraction_signed_concordant")
    estimable["n_common"] = _numeric(estimable, "n_common_tissues", default=0).fillna(0).astype(int)
    estimable = estimable.dropna(subset=["similarity"]).sort_values("similarity")
    if estimable.empty:
        return _status_figure(
            out_path=out_path,
            title="Exosome-Alignment Evidence",
            message="Alignment rows were estimable but did not contain finite similarity metrics.",
            source=source,
        )

    _set_style()
    fig, ax = plt.subplots(figsize=(8.8, max(4.8, 0.5 * len(estimable) + 2.0)))
    y = np.arange(len(estimable))
    colors = np.where(estimable["spearman_rho_num"].fillna(0) >= 0, TEAL, AMBER)
    ax.barh(y, estimable["similarity"], color=colors, alpha=0.86)
    if estimable["ci_low_num"].notna().any() and estimable["ci_high_num"].notna().any():
        x = estimable["similarity"].to_numpy(float)
        low = x - estimable["ci_low_num"].to_numpy(float)
        high = estimable["ci_high_num"].to_numpy(float) - x
        ok = np.isfinite(low) & np.isfinite(high)
        ax.errorbar(x[ok], y[ok], xerr=np.vstack([low[ok], high[ok]]), fmt="none", ecolor=INK, capsize=3)
    labels = []
    for row in estimable.itertuples(index=False):
        labels.append(
            f"{row.contrast}  n={int(row.n_common)}  rho={row.spearman_rho_num:.2f}  signed={row.signed_concordance:.2f}"
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(1.0, float(estimable["similarity"].max()) * 1.15))
    ax.set_xlabel("Standardized effect similarity")
    ax.set_title("Cross-Species Exosome-Alignment Support")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.0,
        -0.16,
        "Higher values indicate stronger standardized similarity; color encodes the sign of rank correlation.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8.5,
    )
    _save(fig, out_path)
    return FigureRecord(out_path.name, out_path, source, "ok", f"{len(estimable)} contrasts plotted")


def plot_exosome_alignment_by_tissue(results_dir: Path, out_dir: Path) -> FigureRecord:
    source = "exosome_alignment_by_tissue.csv"
    df = _read_table(results_dir, source)
    out_path = out_dir / "report_exosome_alignment_by_tissue.png"
    if df.empty:
        return _status_figure(
            out_path=out_path,
            title="Exosome Alignment by Tissue",
            message="No tissue-level exosome-alignment table was available.",
            source=source,
        )

    required = {"contrast", "primate_tissue", "macaque_effect", "mouse_effect"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required.difference(df.columns)))
        return _status_figure(
            out_path=out_path,
            title="Exosome Alignment by Tissue",
            message=f"The alignment table is missing required columns: {missing}",
            source=source,
        )

    work = df.loc[_bool_series(df, "estimable", default=False)].copy()
    weak_context_count = 0
    if "include_in_primary" in work.columns:
        primary_mask = _bool_series(work, "include_in_primary", default=False)
        weak_context_count = int((~primary_mask).sum())
        work = work.loc[primary_mask].copy()
    if work.empty:
        reason = _first_text(df, "reason_code", "NON_ESTIMABLE_UNSPECIFIED")
        key = _first_text(df, "missing_author_key", "review_required")
        return _status_figure(
            out_path=out_path,
            title="Exosome Alignment by Tissue",
            message=f"No estimable tissue-level alignment rows. reason_code={reason}; missing_author_key={key}",
            source=source,
        )

    work["macaque_effect_num"] = _numeric(work, "macaque_effect")
    work["mouse_effect_num"] = _numeric(work, "mouse_effect")
    work["signed_concordance_num"] = _numeric(work, "signed_concordance", default=0).fillna(0)
    work["similarity_num"] = _numeric(work, "standardized_effect_similarity")
    work["residual_num"] = _numeric(work, "residual_component")
    if work["residual_num"].isna().all() and work["similarity_num"].notna().any():
        work["residual_num"] = 1.0 - work["similarity_num"]
    work = work.dropna(subset=["residual_num", "macaque_effect_num", "mouse_effect_num"])
    if work.empty:
        return _status_figure(
            out_path=out_path,
            title="Exosome Alignment by Tissue",
            message="Tissue-level alignment rows were present but did not contain finite effect-mismatch metrics.",
            source=source,
        )

    work["label"] = work["contrast"].astype(str) + " | " + work["primate_tissue"].astype(str)
    work = work.sort_values(["contrast", "signed_concordance_num", "residual_num"], ascending=[True, True, False])
    colors = np.where(work["signed_concordance_num"] >= 0.5, TEAL, RED)

    _set_style()
    fig, ax = plt.subplots(figsize=(11.2, max(5.2, 0.46 * len(work) + 2.0)))
    y = np.arange(len(work))
    ax.barh(y, work["residual_num"], color=colors, alpha=0.86)
    for yi, row in enumerate(work.itertuples(index=False)):
        text = f"macaque {row.macaque_effect_num:+.2f}; mouse {row.mouse_effect_num:+.2f}"
        ax.text(row.residual_num + max(work["residual_num"].max() * 0.02, 0.03), yi, text, va="center", fontsize=8.2, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(work["label"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Effect mismatch (standardized residual component; lower = closer alignment)")
    ax.set_title("Exosome Alignment by Tissue: Concordant and Discordant Drivers")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=TEAL, label="signed concordant"),
            plt.Rectangle((0, 0), 1, 1, color=RED, label="signed discordant"),
        ],
        frameon=False,
        loc="lower right",
    )
    ax.text(
        0.0,
        -0.14,
        (
            "This is cross-species mechanism support, not direct primate causal mediation. "
            + (
                f"{weak_context_count} weak-context pair(s) remain in the result table as sensitivity only."
                if weak_context_count
                else ""
            )
        ),
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8.5,
    )
    _save(fig, out_path)
    n_discordant = int((work["signed_concordance_num"] < 0.5).sum())
    return FigureRecord(
        out_path.name,
        out_path,
        source,
        "ok",
        (
            f"{len(work)} primary tissue rows plotted; {n_discordant} discordant; "
            f"{weak_context_count} weak-context sensitivity rows excluded"
        ),
    )


def plot_estimability_status(results_dir: Path, out_dir: Path) -> FigureRecord:
    source = _source_label("estimability_report.csv", "linkage_qc_report.csv")
    estimability = _read_table(results_dir, "estimability_report.csv")
    out_path = out_dir / "report_estimability_status.png"
    if estimability.empty:
        return _status_figure(
            out_path=out_path,
            title="Linked-Mediation Estimability",
            message="No estimability report was available.",
            source=source,
        )

    row = estimability.iloc[0]
    metrics = [
        ("All linked animals", "n_overlap_animal_ids", "min_overlap_animals"),
        ("Treated linked animals", "n_overlap_treated_animals", "min_treated_overlap"),
        ("Control linked animals", "n_overlap_control_animals", "min_control_overlap"),
        ("Mediation samples", "n_overlap_animal_ids", "min_samples_for_mediation"),
    ]
    values = []
    minimums = []
    labels = []
    for label, value_col, min_col in metrics:
        values.append(float(pd.to_numeric(pd.Series([row.get(value_col)]), errors="coerce").iloc[0] or 0))
        minimums.append(float(pd.to_numeric(pd.Series([row.get(min_col)]), errors="coerce").iloc[0] or 0))
        labels.append(label)

    tier = str(row.get("tier", "unknown"))
    reason_code = str(row.get("reason_code", ""))
    missing_key = str(row.get("missing_author_key", ""))
    can_do = str(row.get("can_do_mediation", "False")).lower() in {"true", "1", "yes"}
    colors = [GREEN if v >= m and m > 0 else RED for v, m in zip(values, minimums)]

    _set_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, alpha=0.88, label="Observed")
    ax.scatter(x, minimums, color=INK, marker="D", s=46, label="Required")
    for i, (v, m) in enumerate(zip(values, minimums)):
        ax.text(i, max(v, m) + max(values + minimums + [1]) * 0.04, f"{int(v)}/{int(m)}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Animal count")
    ax.set_title(f"Linked-Mediation Gate: {tier} ({'estimable' if can_do else 'not estimable'})")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    note = f"reason_code={reason_code or 'OK'}"
    if missing_key and missing_key.lower() not in {"nan", "none"}:
        note += f"; missing_author_key={missing_key}"
    ax.text(0.0, -0.24, note, transform=ax.transAxes, color=MUTED, fontsize=8.3, wrap=True)
    _save(fig, out_path)
    return FigureRecord(out_path.name, out_path, source, "ok", note)


def plot_mediation_uncertainty(results_dir: Path, out_dir: Path) -> FigureRecord:
    source = "mediation_summary.csv"
    df = _read_table(results_dir, source)
    out_path = out_dir / "report_mediation_uncertainty.png"
    if df.empty:
        return _status_figure(
            out_path=out_path,
            title="Mediation Uncertainty",
            message="No mediation summary table was available.",
            source=source,
        )

    estimable = df.loc[_bool_series(df, "estimable", default=False)].copy()
    if estimable.empty:
        reason = _first_text(df, "reason_code", "NON_ESTIMABLE_UNSPECIFIED")
        key = _first_text(df, "missing_author_key", "review_required")
        return _status_figure(
            out_path=out_path,
            title="Mediation Uncertainty",
            message=f"Linked mediation is not estimable. reason_code={reason}; missing_author_key={key}",
            source=source,
        )

    row = estimable.iloc[0]
    metric_specs = [
        ("ACME", "ACME_CI", "ACME"),
        ("ADE", "ADE_CI", "ADE"),
        ("Total", "Total_CI", "Total effect"),
        ("PropMediated", "PropMediated_CI", "Proportion mediated"),
    ]
    metrics: list[dict[str, object]] = []
    for value_col, ci_col, label in metric_specs:
        value = pd.to_numeric(pd.Series([row.get(value_col)]), errors="coerce").iloc[0]
        low, high = _parse_ci(row.get(ci_col))
        metrics.append(
            {
                "label": label,
                "value": float(value) if pd.notna(value) else np.nan,
                "ci_low": low,
                "ci_high": high,
                "crosses_zero": _ci_crosses_zero(low, high),
            }
        )
    finite_metrics = [item for item in metrics if np.isfinite(float(item["value"]))]
    if not finite_metrics:
        return _status_figure(
            out_path=out_path,
            title="Mediation Uncertainty",
            message="Mediation was marked estimable, but no finite mediation metrics were present.",
            source=source,
        )

    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.3), gridspec_kw={"width_ratios": [1.35, 1.0]})
    panels = [
        (axes[0], finite_metrics[:3], "Direct, indirect, and total effects", True),
        (axes[1], finite_metrics[3:], "Proportion mediated", False),
    ]
    unstable_labels: list[str] = []
    for ax, panel_metrics, title, show_y_labels in panels:
        if not panel_metrics:
            ax.axis("off")
            continue
        y = np.arange(len(panel_metrics))
        values = np.array([float(item["value"]) for item in panel_metrics], dtype=float)
        lows = np.array([float(item["ci_low"]) for item in panel_metrics], dtype=float)
        highs = np.array([float(item["ci_high"]) for item in panel_metrics], dtype=float)
        crosses = np.array([bool(item["crosses_zero"]) for item in panel_metrics], dtype=bool)
        colors = np.where(crosses, RED, TEAL)
        ax.scatter(values, y, s=70, color=colors, zorder=3)
        ok = np.isfinite(lows) & np.isfinite(highs)
        if ok.any():
            ax.errorbar(
                values[ok],
                y[ok],
                xerr=np.vstack([values[ok] - lows[ok], highs[ok] - values[ok]]),
                fmt="none",
                ecolor=INK,
                capsize=3,
                elinewidth=1.2,
                zorder=2,
            )
        for item in panel_metrics:
            if bool(item["crosses_zero"]):
                unstable_labels.append(str(item["label"]))
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [str(item["label"]) for item in panel_metrics] if show_y_labels else []
        )
        if not show_y_labels:
            ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    tier = str(row.get("tier", "unknown"))
    n_overlap = int(pd.to_numeric(pd.Series([row.get("n_overlap_animal_ids", row.get("n_used", 0))]), errors="coerce").fillna(0).iloc[0])
    fig.subplots_adjust(top=0.78, bottom=0.15, wspace=0.24)
    fig.suptitle("Mediation Uncertainty in the Linked Subset", x=0.02, ha="left", fontsize=15, weight="bold")
    warning = (
        "Warning: confidence intervals crossing zero indicate statistically unstable mediation estimates. "
        "The proportion-mediated estimate is especially fragile when the total effect is near zero."
        if unstable_labels
        else "No plotted confidence interval crossed zero; causal interpretation still depends on the linked-mediation assumptions."
    )
    fig.text(0.02, 0.86, f"Gate: {tier}; linked animals: n={n_overlap}", color=MUTED, fontsize=9.2)
    fig.text(0.02, 0.02, warning, color=RED if unstable_labels else MUTED, fontsize=8.6)
    _save(fig, out_path)
    message = "estimable but unstable; CIs cross zero for " + ", ".join(dict.fromkeys(unstable_labels)) if unstable_labels else "estimable; CIs do not cross zero"
    return FigureRecord(out_path.name, out_path, source, "ok", message)


def plot_plasma_biomarkers(results_dir: Path, out_dir: Path, top_n: int = 15) -> FigureRecord:
    source = "plasma_biomarkers.csv"
    df = _read_table(results_dir, source)
    out_path = out_dir / "report_plasma_biomarker_signed_stability.png"
    if df.empty:
        return _status_figure(
            out_path=out_path,
            title="Plasma Biomarker Candidates",
            message="No plasma biomarker table was available.",
            source=source,
        )
    is_contrast_table = "log2_fold_change" in df.columns
    if not is_contrast_table and ("protein" not in df.columns or "spearman_r" not in df.columns):
        return _status_figure(
            out_path=out_path,
            title="Plasma Protein Signals",
            message="The plasma table is missing treatment-effect or legacy association columns.",
            source=source,
        )

    work = df.copy()
    if is_contrast_table and "is_primary" in work.columns:
        primary = _bool_series(work, "is_primary", default=False)
        if primary.any():
            work = work.loc[primary].copy()
    if "gene_name" in work.columns:
        work["protein"] = work["gene_name"].astype("string").fillna("").str.strip()
        if "protein_accession" in work.columns:
            fallback = work["protein_accession"].astype("string").fillna("").str.strip()
            work["protein"] = work["protein"].where(work["protein"].ne(""), fallback)
    else:
        work["protein"] = work["protein"].astype(str).str.strip()
    work = work.loc[work["protein"].ne("") & work["protein"].ne("--")].copy()
    work["rho"] = _numeric(work, "log2_fold_change" if is_contrast_table else "spearman_r")
    work["abs_rho"] = work["rho"].abs()
    work["stable"] = _bool_series(work, "stable_association", default=False)
    if is_contrast_table:
        work["rho_ci_low_num"] = _numeric(work, "bootstrap_ci_low").fillna(_numeric(work, "ci_low"))
        work["rho_ci_high_num"] = _numeric(work, "bootstrap_ci_high").fillna(_numeric(work, "ci_high"))
    else:
        work["rho_ci_low_num"] = _numeric(work, "rho_ci_low")
        work["rho_ci_high_num"] = _numeric(work, "rho_ci_high")
    stable = work.loc[work["stable"]].copy()
    if not stable.empty:
        work = stable
    work = work.dropna(subset=["rho"]).sort_values("abs_rho", ascending=False).head(top_n)
    work = work.sort_values("rho")
    if work.empty:
        return _status_figure(
            out_path=out_path,
            title="Plasma Protein Signals",
            message="No named plasma proteins had finite treatment-effect estimates.",
            source=source,
        )

    _set_style()
    fig, ax = plt.subplots(figsize=(9.2, max(5.2, 0.36 * len(work) + 2.0)))
    y = np.arange(len(work))
    colors = np.where(work["rho"] < 0, TEAL, RUST)
    ax.barh(y, work["rho"], color=colors, alpha=0.88)
    if work["rho_ci_low_num"].notna().any() and work["rho_ci_high_num"].notna().any():
        x = work["rho"].to_numpy(float)
        low = x - work["rho_ci_low_num"].to_numpy(float)
        high = work["rho_ci_high_num"].to_numpy(float) - x
        ok = np.isfinite(low) & np.isfinite(high)
        ax.errorbar(x[ok], y[ok], xerr=np.vstack([low[ok], high[ok]]), fmt="none", ecolor=INK, capsize=2.5)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(work["protein"].tolist())
    ax.set_xlabel(
        "Adjusted log2 fold-change (GES - vehicle)"
        if is_contrast_table
        else "Signed Spearman association with legacy plasma state"
    )
    subtitle = "stable associations only" if not stable.empty else "top named associations"
    plot_title = (
        "Treatment-Associated Plasma Protein Signals"
        if is_contrast_table
        else "Legacy Plasma Protein Associations"
    )
    ax.set_title(f"{plot_title} ({subtitle})")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.0,
        -0.16,
        (
            "Effects are within-plasma treatment associations, not rejuvenation biomarkers or therapeutic targets."
            if is_contrast_table
            else "Legacy signed associations are shown only for historical output compatibility."
        ),
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8.5,
    )
    _save(fig, out_path)
    return FigureRecord(out_path.name, out_path, source, "ok", f"{len(work)} proteins plotted")


def plot_plasma_biomarker_categories(results_dir: Path, out_dir: Path, top_n: int = 30) -> FigureRecord:
    source = "plasma_biomarkers.csv"
    df = _read_table(results_dir, source)
    out_path = out_dir / "report_plasma_biomarker_categories.png"
    if df.empty:
        return _status_figure(
            out_path=out_path,
            title="Plasma Biomarker Categories",
            message="No plasma biomarker table was available.",
            source=source,
        )
    is_contrast_table = "log2_fold_change" in df.columns
    if not is_contrast_table and ("protein" not in df.columns or "spearman_r" not in df.columns):
        return _status_figure(
            out_path=out_path,
            title="Plasma Protein Categories",
            message="The plasma table is missing treatment-effect or legacy association columns.",
            source=source,
        )

    work = df.copy()
    if is_contrast_table and "is_primary" in work.columns:
        primary = _bool_series(work, "is_primary", default=False)
        if primary.any():
            work = work.loc[primary].copy()
    if "gene_name" in work.columns:
        work["protein"] = work["gene_name"].astype("string").fillna("").str.strip()
        if "protein_accession" in work.columns:
            fallback = work["protein_accession"].astype("string").fillna("").str.strip()
            work["protein"] = work["protein"].where(work["protein"].ne(""), fallback)
    else:
        work["protein"] = work["protein"].astype(str).str.strip()
    work = work.loc[work["protein"].map(_is_named_protein)].copy()
    work["rho"] = _numeric(work, "log2_fold_change" if is_contrast_table else "spearman_r")
    work["abs_rho"] = work["rho"].abs()
    work["stable"] = _bool_series(work, "stable_association", default=False)
    if work["stable"].any():
        work = work.loc[work["stable"]].copy()
    work = work.dropna(subset=["rho"]).sort_values("abs_rho", ascending=False).head(max(1, int(top_n)))
    if work.empty:
        return _status_figure(
            out_path=out_path,
            title="Plasma Protein Categories",
            message="No named plasma proteins had finite treatment-effect estimates.",
            source=source,
        )

    annotation_col = next(
        (
            col
            for col in ["pathway_category", "protein_category", "pathway", "category"]
            if col in work.columns and work[col].notna().any()
        ),
        None,
    )
    if annotation_col:
        work["category"] = work[annotation_col].fillna("Unannotated").astype(str)
        annotation_note = f"Categories use the provided `{annotation_col}` column."
    else:
        work["category"] = work["protein"].map(_protein_category)
        annotation_note = "Categories are heuristic protein-symbol groupings, not formal pathway enrichment."
    negative_label = "lower in GES" if is_contrast_table else "negative signed association"
    positive_label = "higher in GES" if is_contrast_table else "positive signed association"
    work["signed_direction"] = np.where(work["rho"] < 0, negative_label, positive_label)
    counts = (
        work.groupby(["category", "signed_direction"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    for col in [negative_label, positive_label]:
        if col not in counts.columns:
            counts[col] = 0
    counts["total"] = counts[negative_label] + counts[positive_label]
    counts = counts.sort_values("total", ascending=True)

    _set_style()
    fig, ax = plt.subplots(figsize=(9.8, max(4.8, 0.48 * len(counts) + 2.2)))
    y = np.arange(len(counts))
    negative = counts[negative_label].to_numpy(int)
    positive = counts[positive_label].to_numpy(int)
    ax.barh(y, -negative, color=TEAL, alpha=0.88, label=negative_label)
    ax.barh(y, positive, color=RUST, alpha=0.88, label=positive_label)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index.tolist())
    max_count = max(int(max(negative.max(initial=0), positive.max(initial=0))), 1)
    ax.set_xlim(-(max_count + 1), max_count + 1)
    ax.set_xlabel("Number of top named plasma proteins")
    ax.set_title(
        "Plasma Protein Categories by GES-Vehicle Effect"
        if is_contrast_table
        else "Legacy Plasma Protein Categories by Signed Association"
    )
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.0,
        -0.17,
        annotation_note,
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8.5,
    )
    _save(fig, out_path)
    return FigureRecord(out_path.name, out_path, source, "ok", f"{len(work)} proteins summarized; {annotation_note}")


def plot_oriented_plasma_age_axis(
    results_dir: Path,
    out_dir: Path,
    top_n_loadings: int = 12,
) -> FigureRecord:
    sources = _source_label(
        "plasma_age_axis_scores.csv",
        "plasma_age_axis_summary.csv",
        "plasma_age_axis_loadings.csv",
        "plasma_age_axis_delta_age_correlation.csv",
    )
    scores = _read_table(results_dir, "plasma_age_axis_scores.csv")
    summary = _read_table(results_dir, "plasma_age_axis_summary.csv")
    loadings = _read_table(results_dir, "plasma_age_axis_loadings.csv")
    correlation = _read_table(results_dir, "plasma_age_axis_delta_age_correlation.csv")
    out_path = out_dir / "report_oriented_plasma_age_axis.png"
    if scores.empty or "plasma_age_axis_score" not in scores.columns or "group" not in scores.columns:
        reason = _first_text(summary, "reason_code", "DATA_FILE_UNAVAILABLE")
        return _status_figure(
            out_path=out_path,
            title="Oriented Plasma Aging Axis",
            message=f"No oriented plasma age-axis score table was available. reason_code={reason}",
            source=sources,
        )

    work = scores.copy()
    if "estimable" in work.columns:
        work = work.loc[_bool_series(work, "estimable", default=True)].copy()
    work["axis_score"] = _numeric(work, "plasma_age_axis_score")
    work["group_clean"] = work["group"].astype(str).str.replace(r"^O_", "", regex=True).str.strip()
    work = work.dropna(subset=["axis_score"])
    if work.empty:
        return _status_figure(
            out_path=out_path,
            title="Oriented Plasma Aging Axis",
            message="The score table was present but had no finite estimable axis scores.",
            source=sources,
        )

    load = loadings.copy()
    if not load.empty and "loading" in load.columns and "protein" in load.columns:
        if "estimable" in load.columns:
            load = load.loc[_bool_series(load, "estimable", default=True)].copy()
        load["protein"] = load["protein"].astype(str).str.strip()
        load["loading_num"] = _numeric(load, "loading")
        load["abs_loading_num"] = _numeric(load, "abs_loading", default=np.nan)
        load["abs_loading_num"] = load["abs_loading_num"].fillna(load["loading_num"].abs())
        named = load.loc[load["protein"].map(_is_named_protein)].copy()
        if len(named) >= 3:
            load = named
        load = (
            load.dropna(subset=["loading_num"])
            .sort_values("abs_loading_num", ascending=False)
            .head(max(1, int(top_n_loadings)))
            .sort_values("loading_num")
        )
    else:
        load = pd.DataFrame()

    shift_label = _first_text(summary, "treated_shift_label", "not_estimated").replace("_", " ")
    fraction = _numeric(summary, "treated_fraction_of_old_young_gap").dropna()
    p_value = _numeric(summary, "treated_vs_old_permutation_p_value").dropna()
    explained = _numeric(summary, "pc1_explained_variance_ratio").dropna()
    old_gap_crosses_zero = False
    if not summary.empty and "old_young_gap_ci_crosses_zero" in summary.columns:
        old_gap_crosses_zero = bool(_bool_series(summary, "old_young_gap_ci_crosses_zero", default=False).any())
    elif not summary.empty:
        gap_low = _numeric(summary, "old_vs_young_gap_ci_low").dropna()
        gap_high = _numeric(summary, "old_vs_young_gap_ci_high").dropna()
        if not gap_low.empty and not gap_high.empty:
            old_gap_crosses_zero = bool(float(gap_low.iloc[0]) <= 0 <= float(gap_high.iloc[0]))
    if not fraction.empty:
        fraction_text = f"{float(fraction.iloc[0]):.2f}"
        if old_gap_crosses_zero:
            fraction_text += " (unstable denominator)"
    else:
        fraction_text = "NA"
    p_text = f"{float(p_value.iloc[0]):.3g}" if not p_value.empty else "NA"
    explained_text = f"{100 * float(explained.iloc[0]):.1f}%" if not explained.empty else "NA"

    corr_estimable = bool(not correlation.empty and _bool_series(correlation, "estimable", default=False).any())
    if corr_estimable:
        corr_row = correlation.loc[_bool_series(correlation, "estimable", default=False)].iloc[0]
        rho = pd.to_numeric(pd.Series([corr_row.get("spearman_rho")]), errors="coerce").iloc[0]
        ci_low = pd.to_numeric(pd.Series([corr_row.get("ci_low")]), errors="coerce").iloc[0]
        ci_high = pd.to_numeric(pd.Series([corr_row.get("ci_high")]), errors="coerce").iloc[0]
        n_animals = int(pd.to_numeric(pd.Series([corr_row.get("n_animals", corr_row.get("n_used", 0))]), errors="coerce").fillna(0).iloc[0])
        corr_p = pd.to_numeric(pd.Series([corr_row.get("permutation_p_value")]), errors="coerce").iloc[0]
        corr_text = f"Linked delta-age correlation: rho={rho:.2f} [{ci_low:.2f}, {ci_high:.2f}], p={corr_p:.3g}, n={n_animals}."
    else:
        reason = _first_text(correlation, "reason_code", "NON_ESTIMABLE_UNSPECIFIED")
        corr_text = f"Linked delta-age correlation not estimable; reason_code={reason}."

    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.2), gridspec_kw={"width_ratios": [1.05, 1.2]})
    ax = axes[0]
    group_order = {"Y": 0, "V": 1, "WT": 2, "GES": 3}
    groups = sorted(work["group_clean"].dropna().unique().tolist(), key=lambda x: (group_order.get(str(x), 99), str(x)))
    positions = np.arange(len(groups))
    box_data = [work.loc[work["group_clean"].eq(group), "axis_score"].to_numpy(float) for group in groups]
    ax.boxplot(
        box_data,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.4},
        boxprops={"facecolor": PALE, "edgecolor": INK, "linewidth": 1.0},
        whiskerprops={"color": MUTED, "linewidth": 1.0},
        capprops={"color": MUTED, "linewidth": 1.0},
    )
    for pos, values, group in zip(positions, box_data, groups):
        offsets = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.array([0.0])
        color = {"Y": TEAL, "GES": BLUE, "V": RUST, "WT": AMBER}.get(str(group), GRAY)
        ax.scatter(np.full(len(values), pos) + offsets, values, s=36, color=color, edgecolors="white", linewidths=0.7, zorder=3)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Oriented plasma PC1 score\n(higher = older-like)")
    ax.set_title("Plasma Age-State Axis by Group")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    if load.empty:
        ax2.axis("off")
        ax2.text(0.04, 0.55, "No finite named protein loadings were available.", transform=ax2.transAxes, color=MUTED)
        ax2.set_title("Top PC1 Loadings")
    else:
        y = np.arange(len(load))
        values = load["loading_num"].to_numpy(float)
        colors = np.where(values < 0, TEAL, RUST)
        ax2.barh(y, values, color=colors, alpha=0.88)
        ax2.axvline(0, color=INK, linewidth=1)
        ax2.set_yticks(y)
        ax2.set_yticklabels(load["protein"].tolist())
        ax2.set_xlabel("PC1 loading after age orientation")
        ax2.set_title("Proteins Driving the Oriented Axis")
        ax2.grid(axis="x", color=GRID, linewidth=0.8)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    fig.suptitle("Oriented Plasma Aging Axis", x=0.02, ha="left", fontsize=15, weight="bold")
    fig.text(
        0.02,
        0.89,
        f"PC1 variance={explained_text}; GES shift label={shift_label}; treated fraction of old-young gap={fraction_text}; GES vs old-control p={p_text}.",
        color=MUTED,
        fontsize=9.0,
    )
    fig.text(0.02, 0.045, corr_text, color=MUTED, fontsize=8.6)
    fig.text(
        0.02,
        0.015,
        "This axis is age-state support from plasma proteomics. It is not a causal decomposition of exosome versus cell-intrinsic effects.",
        color=MUTED,
        fontsize=8.4,
    )
    fig.subplots_adjust(top=0.82, bottom=0.17, wspace=0.42)
    _save(fig, out_path)
    return FigureRecord(
        out_path.name,
        out_path,
        sources,
        "ok",
        f"{len(work)} plasma samples plotted; {len(load)} loadings shown; {corr_text}",
    )


def plot_sensitivity_robustness(results_dir: Path, out_dir: Path) -> FigureRecord:
    source = "sensitivity_summary.csv"
    df = _read_table(results_dir, source)
    out_path = out_dir / "report_sensitivity_robustness.png"
    if df.empty or "effect" not in df.columns:
        return _status_figure(
            out_path=out_path,
            title="Sensitivity and Robustness",
            message="No sensitivity table with effect estimates was available.",
            source=source,
        )

    work = df.loc[_bool_series(df, "estimable", default=False)].copy()
    work["effect_num"] = _numeric(work, "effect")
    work = work.dropna(subset=["effect_num"])
    if work.empty:
        return _status_figure(
            out_path=out_path,
            title="Sensitivity and Robustness",
            message="Sensitivity analyses were present but no finite estimable effects were available.",
            source=source,
        )

    control_order = {
        "SRC_vs_vehicle": 0,
        "WTC_vs_vehicle": 1,
        "SRC_vs_WTC": 2,
        "primary": 10,
        "no_vehicle_no_wt": 11,
        "oc_only": 12,
        "consensus": 13,
    }
    work["scenario_text"] = work["scenario"].astype(str)
    work["scenario_order"] = work["scenario_text"].map(control_order).fillna(99).astype(int)
    work["top_n"] = pd.to_numeric(work["scenario_text"].str.extract(r"(\d+)")[0], errors="coerce")

    control = work.loc[
        work["analysis_type"].isin(
            ["animal_level_treatment_contrast", "control_set", "control_set_summary"]
        )
    ].copy()
    control = control.sort_values(["scenario_order", "effect_num"])
    plasma = work.loc[work["analysis_type"].eq("top_features")].copy()
    plasma = plasma.sort_values("top_n", na_position="last")

    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.6, max(5.0, 0.56 * max(len(control), len(plasma), 1) + 2.0)))

    def draw_panel(
        ax: plt.Axes,
        subset: pd.DataFrame,
        *,
        title: str,
        xlabel: str,
        label_map: dict[str, str] | None = None,
        show_ci: bool = False,
        pc1_panel: bool = False,
    ) -> None:
        if subset.empty:
            ax.axis("off")
            ax.text(0.05, 0.55, "No estimable rows.", transform=ax.transAxes, color=MUTED)
            ax.set_title(title)
            return
        y = np.arange(len(subset))
        effects = subset["effect_num"].to_numpy(float)
        colors = np.where(effects < 0, TEAL, RUST)
        ax.scatter(effects, y, color=colors, s=58, zorder=3)
        for yi, effect in zip(y, effects):
            ax.plot([0, effect], [yi, yi], color=GRID, linewidth=2, zorder=1)
        if show_ci and {"ci_low", "ci_high"}.issubset(subset.columns):
            lows = _numeric(subset, "ci_low").to_numpy(float)
            highs = _numeric(subset, "ci_high").to_numpy(float)
            ok = np.isfinite(lows) & np.isfinite(highs)
            if ok.any():
                crosses = (lows <= 0) & (highs >= 0)
                ax.errorbar(
                    effects[ok],
                    y[ok],
                    xerr=np.vstack([effects[ok] - lows[ok], highs[ok] - effects[ok]]),
                    fmt="none",
                    ecolor=INK,
                    capsize=3,
                    elinewidth=1.1,
                    zorder=2,
                )
                open_mask = ok & crosses
                if open_mask.any():
                    ax.scatter(
                        effects[open_mask],
                        y[open_mask],
                        facecolors="white",
                        edgecolors=np.where(effects[open_mask] < 0, TEAL, RUST),
                        linewidths=1.2,
                        s=64,
                        zorder=4,
                    )
        labels = subset["scenario_text"].map(label_map or {}).fillna(subset["scenario_text"]).tolist()
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    draw_panel(
        axes[0],
        control,
        title="Animal-Level Treatment Contrasts",
        xlabel="Secondary global delta-age contrast (years)",
        label_map={
            "SRC_vs_vehicle": "SRC - vehicle (primary)",
            "WTC_vs_vehicle": "WTC - vehicle",
            "SRC_vs_WTC": "SRC - WTC",
            "primary": "Primary controls",
            "no_vehicle_no_wt": "No vehicle/WT controls",
            "oc_only": "Old untreated controls only",
            "consensus": "Direction consensus",
        },
        show_ci=True,
    )
    draw_panel(
        axes[1],
        plasma,
        title="Plasma Top-Feature Sensitivity",
        xlabel="Median plasma-state difference: GES - plasma controls (z-score)",
        label_map={
            "plasma_top_20": "Top 20 proteins",
            "plasma_top_50": "Top 50 proteins",
            "plasma_top_100": "Top 100 proteins",
        },
        pc1_panel=True,
    )
    fig.subplots_adjust(bottom=0.24, top=0.82, wspace=0.34)
    fig.suptitle("Sensitivity Analysis: Robustness Checks by Analysis Family", x=0.02, ha="left", fontsize=15, weight="bold")
    fig.text(
        0.02,
        0.045,
        "Panels use different units and should not be compared by magnitude. "
        "For plasma PC1, sign is arbitrary; top-k consistency is the useful diagnostic.",
        color=MUTED,
        fontsize=8.5,
    )
    _save(fig, out_path)
    return FigureRecord(
        out_path.name,
        out_path,
        source,
        "ok",
        f"{len(control)} treatment/comparator scenarios and {len(plasma)} plasma top-feature scenarios plotted",
    )


def plot_evidence_ladder(results_dir: Path, out_dir: Path) -> FigureRecord:
    sources = _source_label(
        "rejuvenation_by_tissue.csv",
        "plasma_biomarkers.csv",
        "estimability_report.csv",
        "exosome_alignment_summary.csv",
        "mediation_summary.csv",
    )
    out_path = out_dir / "report_evidence_ladder.png"
    rejuvenation = _read_table(results_dir, "rejuvenation_by_tissue.csv")
    plasma = _read_table(results_dir, "plasma_biomarkers.csv")
    estimability = _read_table(results_dir, "estimability_report.csv")
    alignment = _read_table(results_dir, "exosome_alignment_summary.csv")
    mediation = _read_table(results_dir, "mediation_summary.csv")

    rows: list[dict[str, str | int]] = []
    if rejuvenation.empty or not _bool_series(rejuvenation, "estimable", default=False).any():
        rows.append(
            {
                "level": 1,
                "claim": "Macaque age and treatment-effect analysis",
                "status": "Not estimable",
                "detail": "No estimable tissue-level rejuvenation table.",
            }
        )
    else:
        rejuv_estimable = rejuvenation.loc[_bool_series(rejuvenation, "estimable", default=False)].copy()
        rejuv_estimable = _select_primary_treatment_contrast(rejuv_estimable)
        effect_col = "mean_effect" if "mean_effect" in rejuv_estimable.columns else "effect_median"
        rejuv_estimable["effect"] = _numeric(rejuv_estimable, effect_col)
        rejuv_estimable["ci_low_num"] = _numeric(rejuv_estimable, "ci_low")
        rejuv_estimable["ci_high_num"] = _numeric(rejuv_estimable, "ci_high")
        rejuv_estimable = _add_effect_uncertainty_for_plot(rejuv_estimable, effect_col="effect")
        n_tissues = int(len(rejuv_estimable))
        n_supported = int((~rejuv_estimable["ci_crosses_zero_plot"].astype(bool)).sum())
        rows.append(
            {
                "level": 1,
                "claim": "Macaque age and treatment-effect analysis",
                "status": "Observed",
                "detail": f"{n_tissues} estimable tissues; {n_supported} tissue CIs exclude zero. This is phenotype support, not attribution.",
            }
        )

    tier = _first_text(estimability, "tier", "unlinked")
    if plasma.empty or not _bool_series(plasma, "estimable", default=False).any():
        rows.append(
            {
                "level": 1,
                "claim": "Within-plasma treatment association",
                "status": "Not estimable",
                "detail": "No estimable prespecified plasma treatment contrasts.",
            }
        )
    else:
        plasma_work = plasma.loc[_bool_series(plasma, "estimable", default=False)].copy()
        if "is_primary" in plasma_work.columns:
            primary = _bool_series(plasma_work, "is_primary", default=False)
            if primary.any():
                plasma_work = plasma_work.loc[primary].copy()
        plasma_work["protein"] = plasma_work.get("protein", pd.Series("", index=plasma_work.index)).astype(str).str.strip()
        plasma_work["stable"] = _bool_series(plasma_work, "stable_association", default=False)
        n_stable_named = int((plasma_work["stable"] & plasma_work["protein"].map(_is_named_protein)).sum())
        n_fdr = int(_bool_series(plasma_work, "fdr_significant", default=False).sum())
        rows.append(
            {
                "level": 1,
                "claim": "Within-plasma treatment association",
                "status": "Exploratory",
                "detail": (
                    f"Primary GES-vehicle contrast: {n_fdr} FDR-significant and "
                    f"{n_stable_named} bootstrap-stable named proteins. "
                    f"Cross-modal linkage remains {tier}."
                ),
            }
        )

    alignment_estimable = alignment.loc[_bool_series(alignment, "estimable", default=False)].copy() if not alignment.empty else pd.DataFrame()
    if alignment_estimable.empty:
        rows.append(
            {
                "level": 3,
                "claim": "Cross-species exosome alignment",
                "status": "Not estimable",
                "detail": "No estimable exosome-alignment summary.",
            }
        )
    else:
        n_common = int(_numeric_max(alignment_estimable, "n_common_tissues", default=_numeric_max(alignment_estimable, "n_used", default=0)))
        p_values = _numeric(alignment_estimable, "permutation_p_value").dropna()
        best_p = float(p_values.min()) if not p_values.empty else np.nan
        rho_values = _numeric(alignment_estimable, "spearman_rho").dropna()
        best_rho = float(rho_values.max()) if not rho_values.empty else np.nan
        strong_alignment = n_common >= 5 and np.isfinite(best_p) and best_p < 0.05 and np.isfinite(best_rho) and best_rho > 0
        status = "Supported" if strong_alignment else "Exploratory"
        detail = f"{n_common} shared tissues; best permutation p={best_p:.3g}; best Spearman rho={best_rho:.2f}."
        rows.append(
            {
                "level": 3,
                "claim": "Cross-species exosome alignment",
                "status": status,
                "detail": detail,
            }
        )

    mediation_estimable = mediation.loc[_bool_series(mediation, "estimable", default=False)].copy() if not mediation.empty else pd.DataFrame()
    if mediation_estimable.empty:
        reason = _first_text(mediation, "reason_code", "NON_ESTIMABLE_UNSPECIFIED")
        rows.append(
            {
                "level": 4,
                "claim": "Linked mediation under stated assumptions",
                "status": "Not estimable",
                "detail": f"Mediation not estimable; reason_code={reason}.",
            }
        )
    else:
        med_row = mediation_estimable.iloc[0]
        ci_cols = ["ACME_CI", "ADE_CI", "Total_CI", "PropMediated_CI"]
        crossing = []
        for col in ci_cols:
            low, high = _parse_ci(med_row.get(col))
            if _ci_crosses_zero(low, high):
                crossing.append(col.replace("_CI", ""))
        detail = (
            "CIs cross zero for " + ", ".join(crossing) + "; the mediation result remains exploratory."
            if crossing
            else "CIs do not cross zero, but causal assumptions still require independent support."
        )
        rows.append(
            {
                "level": 4,
                "claim": "Linked mediation under stated assumptions",
                "status": "Exploratory",
                "detail": detail,
            }
        )

    _set_style()
    fig, ax = plt.subplots(figsize=(13.2, 5.8))
    ax.axis("off")
    status_colors = {
        "Observed": BLUE,
        "Supported": GREEN,
        "Exploratory": AMBER,
        "Not estimable": GRAY,
        "Not established": RED,
    }
    ax.text(0.02, 0.96, "Design and Estimability Ladder", transform=ax.transAxes, fontsize=15, weight="bold", color=INK)
    ax.text(
        0.02,
        0.9,
        "Levels record design requirements reached; claim status records what the current evidence supports.",
        transform=ax.transAxes,
        fontsize=9.4,
        color=MUTED,
    )
    columns = [("Claim", 0.03, 0.29), ("Level", 0.35, 0.08), ("Status", 0.47, 0.18), ("Interpretation note", 0.68, 0.28)]
    row_h = 0.135
    top = 0.75
    for label, x, width in columns:
        ax.add_patch(plt.Rectangle((x, top), width, row_h * 0.62, transform=ax.transAxes, color=INK, ec="white"))
        ax.text(x + 0.008, top + row_h * 0.31, label, transform=ax.transAxes, color="white", va="center", fontsize=9, weight="bold")
    for i, row in enumerate(rows, start=1):
        y = top - i * row_h
        bg = "white" if i % 2 else PALE
        for _, x, width in columns:
            ax.add_patch(plt.Rectangle((x, y), width, row_h, transform=ax.transAxes, color=bg, ec=GRID, lw=0.7))
        status = str(row["status"])
        color = status_colors.get(status, GRAY)
        ax.text(0.04, y + row_h / 2, _wrapped(row["claim"], 32), transform=ax.transAxes, va="center", fontsize=8.5, color=INK)
        circle = plt.Circle((0.39, y + row_h / 2), 0.033, transform=ax.transAxes, color=color, ec="white", lw=1.0)
        ax.add_patch(circle)
        ax.text(0.39, y + row_h / 2, f"L{row['level']}", transform=ax.transAxes, color="white", ha="center", va="center", fontsize=8.8, weight="bold")
        ax.text(0.48, y + row_h / 2, status, transform=ax.transAxes, va="center", fontsize=8.5, color=INK, weight="bold")
        ax.text(0.69, y + row_h / 2, _wrapped(row["detail"], 45), transform=ax.transAxes, va="center", fontsize=8.0, color=INK)
    ax.text(
        0.02,
        0.04,
        "Observed does not mean statistically confirmed. Causal mediation remains unestablished when estimates are unstable or identifying assumptions lack support.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTED,
    )
    _save(fig, out_path)
    status_counts = pd.Series([row["status"] for row in rows]).value_counts().to_dict()
    message = "; ".join(f"{status}={count}" for status, count in status_counts.items())
    return FigureRecord(out_path.name, out_path, sources, "ok", message)


def _evidence_row(
    label: str,
    filename: str,
    df: pd.DataFrame,
    default_key: str = "",
) -> dict[str, object]:
    if df.empty:
        return {
            "claim": label,
            "source": filename,
            "estimable": False,
            "evidence_level": 0,
            "reason_code": "DATA_FILE_UNAVAILABLE",
            "limitation": "source table unavailable",
            "missing_author_key": default_key or "not_required",
        }
    estimable = bool(_bool_series(df, "estimable", default=False).any())
    level = int(_numeric_max(df, "evidence_level", default=0))
    reason_code = _first_text(df, "reason_code", "OK" if estimable else "NON_ESTIMABLE_UNSPECIFIED")
    author_key = _first_text(df, "missing_author_key", default_key)
    if estimable and (not author_key or str(author_key).strip().lower() in {"nan", "none"}):
        author_key = "not_required"
    return {
        "claim": label,
        "source": filename,
        "estimable": estimable,
        "evidence_level": level,
        "reason_code": reason_code,
        "limitation": _claim_limitation(label, df, estimable=estimable, reason_code=reason_code),
        "missing_author_key": author_key,
    }


def _claim_limitation(
    label: str,
    df: pd.DataFrame,
    *,
    estimable: bool,
    reason_code: str,
) -> str:
    if not estimable:
        return _display_reason_code(reason_code)

    if label == "Macaque rejuvenation":
        estimable_rows = df.loc[_bool_series(df, "estimable", default=False)].copy()
        n_rows = int(len(estimable_rows))
        lows = _numeric(estimable_rows, "ci_low")
        highs = _numeric(estimable_rows, "ci_high")
        finite = lows.notna() & highs.notna()
        if finite.any():
            crosses = (lows <= 0) & (highs >= 0)
            n_cross = int(crosses.loc[finite].sum())
            if n_cross == int(finite.sum()):
                return f"{n_rows} tissues; all CIs cross zero"
            if n_cross:
                return f"{n_rows} tissues; {n_cross} CIs cross zero"
        return f"{n_rows} tissues estimable; tissue-level uncertainty remains"

    if label == "Plasma association":
        n_used = int(_numeric_max(df, "n_used", default=0))
        if str(reason_code) == "LOW_SAMPLE_SIZE_CAUTION" or n_used < 50:
            return f"small plasma cohort (n={n_used})"
        return "association ranking only; not causal"

    if label == "Linked mediation":
        estimable_rows = df.loc[_bool_series(df, "estimable", default=False)].copy()
        if estimable_rows.empty:
            return "not estimable"
        row = estimable_rows.iloc[0]
        unstable = []
        for col in ["ACME_CI", "ADE_CI", "Total_CI", "PropMediated_CI"]:
            low, high = _parse_ci(row.get(col))
            if _ci_crosses_zero(low, high):
                unstable.append(col.replace("_CI", ""))
        n_animals = int(
            pd.to_numeric(
                pd.Series([row.get("n_overlap_animal_ids", row.get("n_used", 0))]),
                errors="coerce",
            ).fillna(0).iloc[0]
        )
        if unstable:
            return f"unstable CIs cross zero; n={n_animals}"
        return f"estimable sensitivity; assumptions still required; n={n_animals}"

    if label == "Exosome alignment":
        estimable_rows = df.loc[_bool_series(df, "estimable", default=False)].copy()
        n_common = int(_numeric_max(estimable_rows, "n_common_tissues", default=_numeric_max(estimable_rows, "n_used", default=0)))
        p_values = _numeric(estimable_rows, "permutation_p_value").dropna()
        best_p = float(p_values.min()) if not p_values.empty else np.nan
        if n_common < 5:
            return f"limited shared tissues (n={n_common})"
        if np.isfinite(best_p) and best_p >= 0.05:
            return f"alignment support weak; best p={best_p:.2g}"
        return "orthogonal support; not direct mediation"

    if label == "Methylation validation":
        return "sample map unresolved; validation blocked"

    if label == "Mammal40 sample map":
        return "author Sentrix/sample sheet required"

    if label == "PBMC single-cell audit":
        return "PBMC-only; not tissue/exosome attribution"

    if str(reason_code) == "OK":
        return "estimable; no blocking metadata gap"
    return _display_reason_code(reason_code)


def plot_public_data_ceiling_matrix(results_dir: Path, out_dir: Path) -> FigureRecord:
    sources = {
        "Macaque rejuvenation": ("rejuvenation_by_tissue.csv", _read_table(results_dir, "rejuvenation_by_tissue.csv")),
        "Plasma association": ("plasma_biomarkers.csv", _read_table(results_dir, "plasma_biomarkers.csv")),
        "Linked mediation": ("mediation_summary.csv", _read_table(results_dir, "mediation_summary.csv")),
        "Exosome alignment": ("exosome_alignment_summary.csv", _read_table(results_dir, "exosome_alignment_summary.csv")),
        "Methylation validation": (
            "methylation_rejuvenation_by_tissue.csv",
            _read_table(results_dir, "methylation_rejuvenation_by_tissue.csv"),
        ),
        "Mammal40 sample map": (
            "omix007582_sample_map_summary.csv",
            _read_table(results_dir, "omix007582_sample_map_summary.csv"),
        ),
        "PBMC single-cell audit": (
            "omix009284_audit_summary.csv",
            _read_table(results_dir, "omix009284_audit_summary.csv"),
        ),
    }
    rows = [
        _evidence_row(label, filename, df)
        for label, (filename, df) in sources.items()
    ]
    matrix = pd.DataFrame(rows)
    out_path = out_dir / "report_public_data_ceiling_matrix.png"
    _set_style()
    fig, ax = plt.subplots(figsize=(12.8, max(6.2, 0.65 * len(matrix) + 2.5)))
    ax.axis("off")
    columns = ["Claim class", "Level", "Status", "Main limitation", "Author-side key"]
    x_positions = [0.02, 0.29, 0.39, 0.51, 0.7]
    widths = [0.25, 0.07, 0.1, 0.17, 0.28]
    row_h = 0.085
    top = 0.78
    ax.text(0.02, 0.96, "Public-Data Evidence Ceiling", transform=ax.transAxes, fontsize=15, weight="bold", color=INK)
    ax.text(
        0.02,
        0.91,
        "Interpretation-facing status of each major claim class.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
    )
    for x, w, col in zip(x_positions, widths, columns):
        ax.add_patch(plt.Rectangle((x, top), w, row_h, transform=ax.transAxes, color=INK, ec="white"))
        ax.text(x + 0.008, top + row_h / 2, col, transform=ax.transAxes, color="white", va="center", fontsize=9, weight="bold")
    level_colors = {0: "#e2e8f0", 1: "#bfdbfe", 2: "#a7f3d0", 3: "#fde68a", 4: "#fecaca"}
    for i, row in enumerate(matrix.itertuples(index=False), start=1):
        y = top - i * row_h
        bg = "white" if i % 2 else PALE
        for x, w in zip(x_positions, widths):
            ax.add_patch(plt.Rectangle((x, y), w, row_h, transform=ax.transAxes, color=bg, ec=GRID, lw=0.7))
        level = int(row.evidence_level)
        ax.add_patch(
            plt.Rectangle((x_positions[1], y), widths[1], row_h, transform=ax.transAxes, color=level_colors.get(level, GRAY), ec=GRID, lw=0.7)
        )
        values = [
            _wrapped(row.claim, 23),
            str(level),
            "estimable" if bool(row.estimable) else "blocked",
            _wrapped(row.limitation, 22),
            _wrapped(_display_author_key(row.missing_author_key), 30),
        ]
        for x, value in zip(x_positions, values):
            ax.text(x + 0.008, y + row_h / 2, value, transform=ax.transAxes, va="center", fontsize=8.0, color=INK)
    ax.text(
        0.02,
        0.04,
        "Causal exosome mediation should remain unavailable unless direct linked-mediation assumptions are met.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTED,
    )
    _save(fig, out_path)
    return FigureRecord(
        out_path.name,
        out_path,
        _source_label(*(filename for filename, _ in sources.values())),
        "ok",
        f"{len(matrix)} claim classes summarized",
    )


def plot_portfolio_aging_rejuvenation(
    results_dir: Path,
    out_dir: Path,
) -> FigureRecord:
    source = _source_label("clock_metrics_primates.csv", "rejuvenation_by_tissue.csv")
    clock = _read_table(results_dir, "clock_metrics_primates.csv")
    rejuvenation = _read_table(results_dir, "rejuvenation_by_tissue.csv")
    out_path = out_dir / "report_portfolio_aging_rejuvenation.png"
    if clock.empty or rejuvenation.empty:
        return _status_figure(
            out_path=out_path,
            title="Aging and Rejuvenation Signal",
            message="Clock metrics or tissue-level rejuvenation results were unavailable.",
            source=source,
        )

    work = rejuvenation.loc[_bool_series(rejuvenation, "estimable", default=False)].copy()
    work = _select_primary_treatment_contrast(work)
    effect_col = "mean_effect" if "mean_effect" in work.columns else "effect_median"
    work["effect"] = _numeric(work, effect_col)
    work["ci_low_num"] = _numeric(work, "ci_low")
    work["ci_high_num"] = _numeric(work, "ci_high")
    work = work.dropna(subset=["effect", "ci_low_num", "ci_high_num"])
    if work.empty:
        return _status_figure(
            out_path=out_path,
            title="Aging and Rejuvenation Signal",
            message="No tissue rows had finite effects and confidence intervals.",
            source=source,
        )

    work["ci_width_plot"] = work["ci_high_num"] - work["ci_low_num"]
    work["priority_score"] = work["effect"].abs() / work["ci_width_plot"].replace(0, np.nan)
    younger = work.loc[work["effect"] < 0].nlargest(4, "priority_score")
    older = work.loc[work["effect"] > 0].nlargest(4, "priority_score")
    selected = pd.concat([younger, older], ignore_index=True).sort_values("effect")
    if selected.empty:
        selected = work.nlargest(min(8, len(work)), "priority_score").sort_values("effect")

    clock_row = clock.iloc[0]
    spearman = float(pd.to_numeric(pd.Series([clock_row.get("spearman_r")]), errors="coerce").iloc[0])
    mae = float(pd.to_numeric(pd.Series([clock_row.get("MAE")]), errors="coerce").iloc[0])
    n_groups = int(
        pd.to_numeric(pd.Series([clock_row.get("cv_n_groups")]), errors="coerce").fillna(0).iloc[0]
    )
    crosses = (work["ci_low_num"] <= 0) & (work["ci_high_num"] >= 0)
    n_crosses = int(crosses.sum())

    _set_style()
    fig = plt.figure(figsize=(13.2, 7.4))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.05, 2.4], hspace=0.3)
    flow_ax = fig.add_subplot(grid[0])
    flow_ax.axis("off")
    fig.suptitle(
        "From Transcriptomic Age Prediction to Tissue-Level Treatment Effects",
        x=0.03,
        ha="left",
        fontsize=16,
        weight="bold",
        color=INK,
    )
    cards = [
        ("Inputs", "Chronological age\n+ RNA abundance"),
        ("Clock", f"Ridge + grouped CV\n{n_groups} animal groups"),
        ("Validation", f"CV predicted age\nSpearman r={spearman:.2f}\nMAE={mae:.2f} years"),
        ("Age deviation", "delta_age = predicted\n- chronological age"),
        ("Tissue contrast", "SRC - vehicle\nanimal bootstrap CI"),
    ]
    card_x = [0.02, 0.215, 0.41, 0.605, 0.80]
    for i, ((title, body), x) in enumerate(zip(cards, card_x)):
        _draw_card(
            flow_ax,
            x=x,
            y=0.14,
            width=0.16,
            height=0.62,
            title=title,
            body=body,
            edge_color=BLUE if i < 3 else TEAL,
            body_offset=0.18,
        )
        if i < len(cards) - 1:
            _draw_arrow(flow_ax, (x + 0.165, 0.45), (card_x[i + 1] - 0.006, 0.45))

    ax = fig.add_subplot(grid[1])
    y = np.arange(len(selected))
    effect = selected["effect"].to_numpy(float)
    lower = effect - selected["ci_low_num"].to_numpy(float)
    upper = selected["ci_high_num"].to_numpy(float) - effect
    point_colors = [TEAL if value < 0 else RUST for value in effect]
    ax.errorbar(
        effect,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor=GRAY,
        elinewidth=1.4,
        capsize=3,
        zorder=1,
    )
    ax.scatter(effect, y, c=point_colors, s=58, edgecolor="white", linewidth=0.8, zorder=2)
    labels = selected["tissue"].astype(str).str.replace("_", " ", regex=False).tolist()
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_xlabel("Treatment effect on delta age (years; negative = younger-like)")
    ax.set_title("Highest-priority nominal tissue shifts (four per direction)", loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.03,
        0.018,
        f"Selection uses |effect| / CI width for display only. {n_crosses}/{len(work)} tissue CIs cross zero; "
        "these are prioritization signals, not confirmed tissue-specific effects.",
        fontsize=8.8,
        color=MUTED,
    )
    _save(fig, out_path)
    return FigureRecord(
        out_path.name,
        out_path,
        source,
        "ok",
        f"clock groups={n_groups}; tissues={len(work)}; CIs crossing zero={n_crosses}",
    )


def plot_portfolio_multimodal_evidence(
    results_dir: Path,
    out_dir: Path,
) -> FigureRecord:
    source = _source_label(
        "rejuvenation_by_tissue.csv",
        "linkage_qc_report.csv",
        "estimability_report.csv",
        "exosome_alignment_summary.csv",
        "cross_species_response_alignment_summary.csv",
        "multimodal_concordance_summary.csv",
    )
    rejuvenation = _read_table(results_dir, "rejuvenation_by_tissue.csv")
    linkage = _read_table(results_dir, "linkage_qc_report.csv")
    estimability = _read_table(results_dir, "estimability_report.csv")
    alignment = _read_table(results_dir, "exosome_alignment_summary.csv")
    response_alignment = _read_table(
        results_dir, "cross_species_response_alignment_summary.csv"
    )
    multimodal = _read_table(results_dir, "multimodal_concordance_summary.csv")
    out_path = out_dir / "report_portfolio_multimodal_evidence.png"

    rejuv_rows = (
        rejuvenation.loc[_bool_series(rejuvenation, "estimable", default=False)].copy()
        if not rejuvenation.empty
        else pd.DataFrame()
    )
    if "is_primary" in rejuv_rows.columns:
        primary_rows = rejuv_rows.loc[
            _bool_series(rejuv_rows, "is_primary", default=False)
        ].copy()
        if not primary_rows.empty:
            rejuv_rows = primary_rows
    rejuv_lows = _numeric(rejuv_rows, "ci_low")
    rejuv_highs = _numeric(rejuv_rows, "ci_high")
    n_nominal_ci = (
        int(((rejuv_lows > 0) | (rejuv_highs < 0)).sum())
        if not rejuv_rows.empty
        else 0
    )
    n_tissues = (
        int(rejuv_rows["tissue"].astype(str).nunique())
        if "tissue" in rejuv_rows.columns
        else int(len(rejuv_rows))
    )

    linkage_row = linkage.iloc[0] if not linkage.empty else pd.Series(dtype=object)
    n_plasma = int(pd.to_numeric(pd.Series([linkage_row.get("n_plasma_total")]), errors="coerce").fillna(0).iloc[0])
    # Bulk alias existence is not confirmed cross-modal identity. Use the
    # canonical gate's high-confidence overlap, never candidate coverage.
    gate_row = estimability.iloc[0] if not estimability.empty else pd.Series(dtype=object)
    n_valid_links = int(
        pd.to_numeric(
            pd.Series([gate_row.get("n_overlap_animal_ids")]),
            errors="coerce",
        )
        .fillna(0)
        .iloc[0]
    )
    linkage_estimable = bool(
        _bool_series(linkage, "estimable", default=False).any()
    )

    alignment_rows = (
        alignment.loc[_bool_series(alignment, "estimable", default=False)].copy()
        if not alignment.empty
        else pd.DataFrame()
    )
    n_common = int(_numeric_max(alignment_rows, "n_common_tissues", default=0))
    p_values = _numeric(alignment_rows, "permutation_p_value").dropna()
    best_p = float(p_values.min()) if not p_values.empty else np.nan
    response_rows = (
        response_alignment.loc[
            _bool_series(response_alignment, "estimable", default=False)
        ].copy()
        if not response_alignment.empty
        else pd.DataFrame()
    )
    if not response_rows.empty:
        response_row = response_rows.iloc[0]
        n_common = int(
            pd.to_numeric(
                pd.Series([response_row.get("n_common_tissues")]), errors="coerce"
            ).fillna(0).iloc[0]
        )
        response_cosine = float(
            pd.to_numeric(
                pd.Series([response_row.get("cosine_similarity")]), errors="coerce"
            ).iloc[0]
        )
        response_p = float(
            pd.to_numeric(
                pd.Series([response_row.get("cosine_permutation_p_value")]),
                errors="coerce",
            ).iloc[0]
        )
    else:
        response_cosine = np.nan
        response_p = np.nan

    methylation_estimable = (
        bool(_bool_series(multimodal, "estimable", default=False).any())
        if not multimodal.empty
        else False
    )
    transcript_available = not rejuv_rows.empty
    plasma_available = linkage_estimable and n_valid_links > 0
    alignment_available = not alignment_rows.empty
    transcript_status = "Observed" if transcript_available else "Not estimable"
    plasma_status = "Exploratory" if plasma_available else "Not estimable"
    alignment_status = "Exploratory" if alignment_available else "Not estimable"
    methylation_status = "Exploratory" if methylation_estimable else "Not estimable"

    _set_style()
    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    ax.axis("off")
    ax.text(
        0.03,
        0.95,
        "Multimodal Pipeline and Evidence Architecture",
        transform=ax.transAxes,
        fontsize=16,
        weight="bold",
        color=INK,
    )
    ax.text(
        0.03,
        0.89,
        "Each modality follows its own analysis path; estimability gates control downstream interpretation.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
    )
    column_labels = (
        (0.03, "Public data"),
        (0.31, "Analysis path"),
        (0.73, "Evidence status"),
    )
    for x, label in column_labels:
        ax.text(
            x,
            0.82,
            label,
            transform=ax.transAxes,
            fontsize=9.2,
            weight="bold",
            color=MUTED,
        )

    rows = (0.64, 0.47, 0.30, 0.13)
    input_cards = (
        (
            "Bulk RNA-seq",
            "OMIX007580\nMacaque tissues",
            AMBER if transcript_available else GRAY,
            "white" if transcript_available else "#f1f5f9",
        ),
        (
            "Plasma proteomics",
            "OMIX007581\nMacaque plasma",
            AMBER if plasma_available else GRAY,
            "white" if plasma_available else "#f1f5f9",
        ),
        (
            "Mammal40 methylation",
            "OMIX007582\nTechnical IDs + beta values",
            GREEN if methylation_estimable else GRAY,
            "white" if methylation_estimable else "#f1f5f9",
        ),
        (
            "Mouse exosome data",
            "OMIX009283\nTissue perturbation",
            AMBER if alignment_available else GRAY,
            "white" if alignment_available else "#f1f5f9",
        ),
    )
    analysis_cards = (
        (
            "Aging and tissue effects",
            "Grouped CV -> clock -> delta_age\n"
            f"-> tissue effects ({n_tissues} tissues; {n_nominal_ci} nominal CIs exclude zero)",
            AMBER if transcript_available else GRAY,
            "white" if transcript_available else "#f1f5f9",
        ),
        (
            "Biomarkers and linkage",
            "Protein ranking -> aging axis -> linkage\n"
            + (
                f"Gate passed: {n_valid_links}/{n_plasma} valid animal links"
                if plasma_available
                else f"Gate blocked: {n_valid_links}/{n_plasma} valid cross-modal links"
            ),
            AMBER if plasma_available else GRAY,
            "white" if plasma_available else "#f1f5f9",
        ),
        (
            "Orthogonal validation",
            "Sample-map gate -> DNAmAge concordance\n"
            + (
                "Biological mapping available"
                if methylation_estimable
                else "Blocked: biological map unavailable"
            ),
            GREEN if methylation_estimable else GRAY,
            "white" if methylation_estimable else "#f1f5f9",
        ),
        (
            "Cross-species alignment",
            "Arm contrasts -> tissue matching -> alignment\n"
            + (
                f"{n_common} tissues; C={response_cosine:.2f}, permutation p={response_p:.3f}"
                if np.isfinite(response_cosine) and np.isfinite(response_p)
                else f"{n_common} tissues; best permutation p={best_p:.3f}"
                if np.isfinite(best_p)
                else f"{n_common} shared tissues"
            ),
            AMBER if alignment_available else GRAY,
            "white" if alignment_available else "#f1f5f9",
        ),
    )

    for y, input_card, analysis_card in zip(rows, input_cards, analysis_cards):
        in_title, in_body, in_color, in_face = input_card
        analysis_title, analysis_body, analysis_color, analysis_face = analysis_card
        _draw_card(
            ax,
            x=0.03,
            y=y,
            width=0.20,
            height=0.14,
            title=in_title,
            body=in_body,
            edge_color=in_color,
            face_color=in_face,
            body_offset=0.092,
        )
        _draw_card(
            ax,
            x=0.31,
            y=y,
            width=0.34,
            height=0.14,
            title=analysis_title,
            body=analysis_body,
            edge_color=analysis_color,
            face_color=analysis_face,
            body_offset=0.092,
        )
        linestyle = "--" if analysis_color == GRAY else "-"
        _draw_arrow(
            ax,
            (0.235, y + 0.07),
            (0.298, y + 0.07),
            color=analysis_color,
            linestyle=linestyle,
        )
        _draw_arrow(
            ax,
            (0.655, y + 0.07),
            (0.718, y + 0.07),
            color=analysis_color,
            linestyle=linestyle,
        )

    _draw_card(
        ax,
        x=0.73,
        y=0.13,
        width=0.24,
        height=0.65,
        title="Current interpretation",
        body=(
            f"{transcript_status}\nTranscriptomic age and tissue effects\n\n"
            f"{plasma_status}\nPlasma-to-tissue association\n\n"
            f"{alignment_status}\nCross-species alignment\n\n"
            f"{methylation_status}\nMethylation concordance\n\n"
            "Not established\nExosome causal attribution"
        ),
        edge_color=INK,
        face_color="white",
        body_color=INK,
        body_offset=0.095,
    )
    ax.text(
        0.03,
        0.035,
        "Solid paths are available in this run; dashed paths are blocked by an estimability gate. "
        "Evidence streams remain separate and are not pooled into one causal effect.",
        transform=ax.transAxes,
        fontsize=8.8,
        color=MUTED,
    )
    _save(fig, out_path)
    methylation_run_status = "estimable" if methylation_estimable else "blocked"
    return FigureRecord(
        out_path.name,
        out_path,
        source,
        "ok",
        f"transcriptomic tissues={n_tissues}; "
        f"plasma gate={'pass' if plasma_available else 'blocked'}; valid links={n_valid_links}; "
        f"methylation={methylation_run_status}; shared tissues={n_common}",
    )


def plot_portfolio_estimability_guardrail(
    results_dir: Path,
    out_dir: Path,
) -> FigureRecord:
    source = _source_label(
        "linkage_qc_report.csv",
        "estimability_report.csv",
        "mediation_summary.csv",
    )
    linkage = _read_table(results_dir, "linkage_qc_report.csv")
    estimability = _read_table(results_dir, "estimability_report.csv")
    mediation = _read_table(results_dir, "mediation_summary.csv")
    out_path = out_dir / "report_portfolio_estimability_guardrail.png"
    if estimability.empty:
        return _status_figure(
            out_path=out_path,
            title="Linked-Mediation Estimability Guardrail",
            message="No estimability report was available.",
            source=source,
        )

    gate = estimability.iloc[0]
    link = linkage.iloc[0] if not linkage.empty else pd.Series(dtype=object)
    can_do = str(gate.get("can_do_mediation", "False")).strip().lower() in {"true", "1", "yes"}
    n_total = int(pd.to_numeric(pd.Series([link.get("n_plasma_total")]), errors="coerce").fillna(0).iloc[0])
    n_linked = int(pd.to_numeric(pd.Series([gate.get("n_overlap_animal_ids")]), errors="coerce").fillna(0).iloc[0])
    n_treated = int(pd.to_numeric(pd.Series([gate.get("n_overlap_treated_animals")]), errors="coerce").fillna(0).iloc[0])
    n_control = int(pd.to_numeric(pd.Series([gate.get("n_overlap_control_animals")]), errors="coerce").fillna(0).iloc[0])
    collisions = int(pd.to_numeric(pd.Series([gate.get("mapping_collision_count")]), errors="coerce").fillna(0).iloc[0])
    min_all = int(pd.to_numeric(pd.Series([gate.get("min_overlap_animals")]), errors="coerce").fillna(0).iloc[0])
    min_treated = int(pd.to_numeric(pd.Series([gate.get("min_treated_overlap")]), errors="coerce").fillna(0).iloc[0])
    min_control = int(pd.to_numeric(pd.Series([gate.get("min_control_overlap")]), errors="coerce").fillna(0).iloc[0])
    reason_code = str(gate.get("reason_code", "NON_ESTIMABLE_UNSPECIFIED"))

    mediation_rows = (
        mediation.loc[_bool_series(mediation, "estimable", default=False)].copy()
        if not mediation.empty
        else pd.DataFrame()
    )
    crossing = 0
    if not mediation_rows.empty:
        med_row = mediation_rows.iloc[0]
        for col in ["ACME_CI", "ADE_CI", "Total_CI", "PropMediated_CI"]:
            low, high = _parse_ci(med_row.get(col))
            crossing += int(_ci_crosses_zero(low, high))

    _set_style()
    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    ax.axis("off")
    ax.text(
        0.03,
        0.95,
        "Animal-Linkage and Mediation Estimability Guardrail",
        transform=ax.transAxes,
        fontsize=16,
        weight="bold",
        color=INK,
    )
    ax.text(
        0.03,
        0.89,
        "The gate controls whether an estimate may be computed; it does not establish causal validity.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
    )

    _draw_card(
        ax,
        x=0.03,
        y=0.58,
        width=0.2,
        height=0.2,
        title="Bulk RNA-seq",
        body="animal_id\ntissue-level age outcome",
        edge_color=BLUE,
    )
    _draw_card(
        ax,
        x=0.03,
        y=0.27,
        width=0.2,
        height=0.2,
        title="Plasma proteomics",
        body=f"{n_total} samples\nanimal_id + confidence",
        edge_color=BLUE,
    )
    _draw_card(
        ax,
        x=0.29,
        y=0.42,
        width=0.22,
        height=0.23,
        title="Linkage audit",
        body=f"{n_linked} linked animals\n{n_treated} treated; {n_control} controls\n{collisions} collisions",
        edge_color=GREEN if collisions == 0 else RED,
    )
    _draw_card(
        ax,
        x=0.56,
        y=0.42,
        width=0.18,
        height=0.23,
        title="Predefined gate",
        body=f"all >= {min_all}\ntreated >= {min_treated}\ncontrols >= {min_control}\nno collisions",
        edge_color=GREEN if can_do else RED,
        face_color="#f0fdf4" if can_do else "#fef2f2",
    )
    _draw_card(
        ax,
        x=0.79,
        y=0.57,
        width=0.18,
        height=0.23,
        title="PASS: estimate",
        body=(
            f"Animal-level mediation\nn={n_linked}\n{crossing}/4 CIs cross zero\nEstimable, not stable"
            if can_do
            else "Available only when\nall gate criteria pass"
        ),
        edge_color=GREEN if can_do else GRAY,
        face_color="#f0fdf4" if can_do else PALE,
    )
    _draw_card(
        ax,
        x=0.79,
        y=0.19,
        width=0.18,
        height=0.23,
        title="FAIL: structured stub",
        body="estimable=False\nreason_code\nmissing_author_key\nno causal estimate",
        edge_color=RED if not can_do else GRAY,
        face_color="#fef2f2" if not can_do else PALE,
    )
    _draw_arrow(ax, (0.235, 0.68), (0.285, 0.56), color=BLUE)
    _draw_arrow(ax, (0.235, 0.37), (0.285, 0.5), color=BLUE)
    _draw_arrow(ax, (0.515, 0.535), (0.555, 0.535), color=GREEN if can_do else RED)
    _draw_arrow(ax, (0.745, 0.56), (0.785, 0.68), color=GREEN if can_do else GRAY)
    _draw_arrow(
        ax,
        (0.745, 0.49),
        (0.785, 0.31),
        color=RED if not can_do else GRAY,
        linestyle="--",
    )
    current_branch = "PASS" if can_do else "FAIL"
    ax.text(
        0.03,
        0.07,
        f"Current full-data path: {current_branch} (reason_code={reason_code}). "
        "The alternative branch is shown to make safe failure behavior explicit.",
        transform=ax.transAxes,
        fontsize=8.8,
        color=MUTED,
    )
    _save(fig, out_path)
    mediation_status = "estimable but unstable" if can_do and crossing else ("estimable" if can_do else "blocked")
    return FigureRecord(
        out_path.name,
        out_path,
        source,
        "ok",
        f"gate={current_branch.lower()}; linked animals={n_linked}; mediation={mediation_status}",
    )


def publish_portfolio_assets(report_dir: Path, assets_dir: Path) -> list[Path]:
    """Copy exactly the three curated portfolio figures into public documentation."""
    report_dir = Path(report_dir)
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    for report_name, asset_name in PORTFOLIO_FIGURE_FILES.items():
        source_path = report_dir / report_name
        if not source_path.exists():
            raise FileNotFoundError(f"Portfolio source figure was not generated: {source_path}")
        target_path = assets_dir / asset_name
        shutil.copy2(source_path, target_path)
        published.append(target_path)
    return published


def generate_report_figures(
    *,
    results_dir: Path = Path("results"),
    out_dir: Path = Path("figures/report"),
    top_n_plasma: int = 15,
    top_n_tissues: int = 8,
    top_n_plasma_categories: int = 30,
    top_n_plasma_axis_loadings: int = 12,
    portfolio_assets_dir: Path | None = None,
) -> pd.DataFrame:
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [
        plot_tissue_rejuvenation_forest(results_dir, out_dir),
        plot_top_tissue_priority(results_dir, out_dir, top_n=top_n_tissues),
        plot_exosome_alignment_summary(results_dir, out_dir),
        plot_exosome_alignment_by_tissue(results_dir, out_dir),
        plot_estimability_status(results_dir, out_dir),
        plot_mediation_uncertainty(results_dir, out_dir),
        plot_evidence_ladder(results_dir, out_dir),
        plot_plasma_biomarkers(results_dir, out_dir, top_n=top_n_plasma),
        plot_plasma_biomarker_categories(results_dir, out_dir, top_n=top_n_plasma_categories),
        plot_oriented_plasma_age_axis(results_dir, out_dir, top_n_loadings=top_n_plasma_axis_loadings),
        plot_sensitivity_robustness(results_dir, out_dir),
        plot_public_data_ceiling_matrix(results_dir, out_dir),
        plot_portfolio_aging_rejuvenation(results_dir, out_dir),
        plot_portfolio_multimodal_evidence(results_dir, out_dir),
        plot_portfolio_estimability_guardrail(results_dir, out_dir),
    ]
    manifest = pd.DataFrame(
        [
            {
                "figure": record.figure,
                "path": str(record.path),
                "source_tables": record.source_tables,
                "status": record.status,
                "message": record.message,
            }
            for record in records
        ]
    )
    manifest_path = out_dir / "report_figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    logger.info("Wrote report figure manifest to %s", manifest_path)
    if portfolio_assets_dir is not None:
        published = publish_portfolio_assets(out_dir, portfolio_assets_dir)
        logger.info("Published %d curated portfolio figures to %s", len(published), portfolio_assets_dir)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate report figures from existing OMIX Exosome Rejuvenation result CSVs."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/report"))
    parser.add_argument("--top-n-plasma", type=int, default=15)
    parser.add_argument("--top-n-tissues", type=int, default=8)
    parser.add_argument("--top-n-plasma-categories", type=int, default=30)
    parser.add_argument("--top-n-plasma-axis-loadings", type=int, default=12)
    parser.add_argument(
        "--portfolio-assets-dir",
        type=Path,
        default=None,
        help="Optionally copy exactly three curated figures into a public documentation directory.",
    )
    args = parser.parse_args(argv)
    manifest = generate_report_figures(
        results_dir=args.results_dir,
        out_dir=args.out_dir,
        top_n_plasma=args.top_n_plasma,
        top_n_tissues=args.top_n_tissues,
        top_n_plasma_categories=args.top_n_plasma_categories,
        top_n_plasma_axis_loadings=args.top_n_plasma_axis_loadings,
        portfolio_assets_dir=args.portfolio_assets_dir,
    )
    logger.info("Generated %d report figures in %s", len(manifest), args.out_dir)


if __name__ == "__main__":
    main()
