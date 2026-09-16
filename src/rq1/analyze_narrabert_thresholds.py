from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ============================================================
# CONFIGURAZIONE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NARRABERT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rq1_narrabert"
    / "dataset_narrabert_emotions.csv"
)

NARRATIVITY_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_narrativity_emotions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rq1_narrabert"
    / "threshold_analysis"
)

THRESHOLDS = [0.30, 0.50, 0.70, 0.80, 0.90]

NARRATIVE_DIMENSIONS = [
    "focalization",
    "emotion",
    "cognition",
    "change_of_state",
    "conflict",
    "concreteness",
    "temporal_grounding",
    "spatial_grounding",
    "sensory",
]

EMOTION_COLUMNS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
]

# Associazioni principali da seguire nei grafici.
KEY_ASSOCIATIONS = [
    ("conflict", "anger"),
    ("emotion", "anger"),
    ("conflict", "disgust"),
    ("change_of_state", "sadness"),
    ("change_of_state", "grief"),
    ("cognition", "anger"),
]


# ============================================================
# UTILITÀ
# ============================================================

def normalize_label(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    out = (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out.loc[numeric == 1] = "FAKE"
    out.loc[numeric == 0] = "REAL"

    return out.replace({
        "TRUE": "REAL",
        "FALSE": "FAKE",
        "1.0": "FAKE",
        "0.0": "REAL",
    })


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    valid = np.isfinite(p)

    if not valid.any():
        return q

    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)

    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[valid] = restored

    return q


def safe_spearman(x: pd.Series, y: pd.Series):
    valid = pd.concat([x, y], axis=1).dropna()

    if len(valid) < 4:
        return len(valid), np.nan, np.nan

    if (
        valid.iloc[:, 0].nunique() < 2
        or valid.iloc[:, 1].nunique() < 2
    ):
        return len(valid), np.nan, np.nan

    rho, p = spearmanr(
        valid.iloc[:, 0],
        valid.iloc[:, 1],
    )

    return len(valid), float(rho), float(p)


# ============================================================
# DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:
    if not NARRABERT_CSV.exists():
        raise FileNotFoundError(
            f"File NarraBERT non trovato:\n{NARRABERT_CSV}"
        )

    if not NARRATIVITY_CSV.exists():
        raise FileNotFoundError(
            f"File StorySeeker non trovato:\n{NARRATIVITY_CSV}"
        )

    narrabert = pd.read_csv(NARRABERT_CSV)

    narrativity = pd.read_csv(
        NARRATIVITY_CSV,
        usecols=[
            "article_id",
            "narrativity_mean",
        ],
    )

    if narrabert["article_id"].duplicated().any():
        raise ValueError(
            "article_id duplicati nel dataset NarraBERT."
        )

    if narrativity["article_id"].duplicated().any():
        raise ValueError(
            "article_id duplicati nel dataset StorySeeker."
        )

    df = narrabert.merge(
        narrativity,
        on="article_id",
        how="inner",
        validate="one_to_one",
    )

    df["label"] = normalize_label(df["label"])

    df = df[
        df["label"].isin(["FAKE", "REAL"])
    ].copy()

    required = {
        "article_id",
        "label",
        "narrativity_mean",
        *EMOTION_COLUMNS,
    }

    for dimension in NARRATIVE_DIMENSIONS:
        required.add(
            f"narrabert_{dimension}_mean"
        )

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Colonne mancanti:\n"
            + "\n".join(missing)
        )

    return df


# ============================================================
# CORRELAZIONI NARRABERT × GOEMOTIONS
# ============================================================

def build_threshold_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for threshold in THRESHOLDS:

        selected = df[
            df["narrativity_mean"] >= threshold
        ].copy()

        scopes = [
            ("ALL", selected),
            (
                "FAKE",
                selected[
                    selected["label"] == "FAKE"
                ],
            ),
            (
                "REAL",
                selected[
                    selected["label"] == "REAL"
                ],
            ),
        ]

        for scope, subset in scopes:

            for dimension in NARRATIVE_DIMENSIONS:

                narrative_column = (
                    f"narrabert_{dimension}_mean"
                )

                for emotion in EMOTION_COLUMNS:

                    n, rho, p = safe_spearman(
                        subset[narrative_column],
                        subset[emotion],
                    )

                    rows.append({
                        "threshold": threshold,
                        "scope": scope,
                        "n": n,
                        "narrative_dimension": dimension,
                        "narrative_column": narrative_column,
                        "emotion": emotion,
                        "spearman_rho": rho,
                        "p_value": p,
                    })

    result = pd.DataFrame(rows)

    result["fdr_q_value"] = np.nan

    # 9 × 27 = 243 test per threshold e per scope.
    for threshold in THRESHOLDS:
        for scope in ["ALL", "FAKE", "REAL"]:

            mask = (
                (result["threshold"] == threshold)
                & (result["scope"] == scope)
            )

            result.loc[
                mask,
                "fdr_q_value",
            ] = benjamini_hochberg(
                result.loc[
                    mask,
                    "p_value",
                ].to_numpy()
            )

    result["significant_fdr_0_05"] = (
        result["fdr_q_value"] < 0.05
    )

    return result


# ============================================================
# ASSOCIAZIONI PRINCIPALI
# ============================================================

def build_key_associations(
    correlations: pd.DataFrame,
) -> pd.DataFrame:

    masks = []

    for dimension, emotion in KEY_ASSOCIATIONS:
        masks.append(
            (
                correlations["narrative_dimension"]
                == dimension
            )
            & (
                correlations["emotion"]
                == emotion
            )
        )

    final_mask = np.logical_or.reduce(masks)

    return (
        correlations[final_mask]
        .copy()
        .sort_values(
            [
                "scope",
                "narrative_dimension",
                "emotion",
                "threshold",
            ]
        )
    )


# ============================================================
# TOP ASSOCIAZIONI PER THRESHOLD
# ============================================================

def build_top_associations(
    correlations: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:

    rows = []

    for threshold in THRESHOLDS:

        for scope in ["FAKE", "REAL"]:

            subset = correlations[
                (
                    correlations["threshold"]
                    == threshold
                )
                & (
                    correlations["scope"]
                    == scope
                )
            ].copy()

            subset["abs_rho"] = (
                subset["spearman_rho"].abs()
            )

            subset = (
                subset
                .sort_values(
                    "abs_rho",
                    ascending=False,
                )
                .head(top_n)
            )

            for rank, (_, row) in enumerate(
                subset.iterrows(),
                start=1,
            ):
                rows.append({
                    "threshold": threshold,
                    "scope": scope,
                    "rank": rank,
                    "narrative_dimension":
                        row["narrative_dimension"],
                    "emotion":
                        row["emotion"],
                    "spearman_rho":
                        row["spearman_rho"],
                    "fdr_q_value":
                        row["fdr_q_value"],
                    "significant_fdr_0_05":
                        row["significant_fdr_0_05"],
                })

    return pd.DataFrame(rows)


# ============================================================
# GRAFICI DI STABILITÀ
# ============================================================

def plot_key_associations(
    key_df: pd.DataFrame,
    scope: str,
    output_name: str,
):
    subset = key_df[
        key_df["scope"] == scope
    ].copy()

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    for dimension, emotion in KEY_ASSOCIATIONS:

        pair = subset[
            (
                subset["narrative_dimension"]
                == dimension
            )
            & (
                subset["emotion"]
                == emotion
            )
        ].sort_values("threshold")

        label = (
            f"{dimension} × {emotion}"
        )

        ax.plot(
            pair["threshold"],
            pair["spearman_rho"],
            marker="o",
            linewidth=2,
            label=label,
        )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Narrativity threshold"
    )

    ax.set_ylabel(
        "Spearman rho"
    )

    ax.set_title(
        f"NarraBERT × GoEmotions stability — {scope}"
    )

    ax.legend(
        fontsize=9,
        loc="best",
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / output_name,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("RQ1 — NARRABERT THRESHOLD ANALYSIS")
    print("=" * 80)

    print(
        f"Thresholds: {THRESHOLDS}"
    )

    print(
        "Narrativity selector: StorySeeker narrativity_mean"
    )

    print(
        "NarraBERT measure: dimension_mean"
    )

    print()

    df = load_dataset()

    print(
        f"Articoli dopo merge: {len(df):,}"
    )

    print(
        df["label"]
        .value_counts()
        .to_string()
    )

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    correlations = (
        build_threshold_correlations(df)
    )

    key_associations = (
        build_key_associations(correlations)
    )

    top_associations = (
        build_top_associations(
            correlations,
            top_n=10,
        )
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    correlations.to_csv(
        OUTPUT_DIR
        / "rq1_narrabert_threshold_correlations.csv",
        index=False,
        encoding="utf-8",
    )

    key_associations.to_csv(
        OUTPUT_DIR
        / "rq1_narrabert_threshold_key_associations.csv",
        index=False,
        encoding="utf-8",
    )

    top_associations.to_csv(
        OUTPUT_DIR
        / "rq1_narrabert_threshold_top_associations.csv",
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # GRAFICI
    # --------------------------------------------------------

    plot_key_associations(
        key_associations,
        "FAKE",
        "rq1_narrabert_threshold_stability_fake.png",
    )

    plot_key_associations(
        key_associations,
        "REAL",
        "rq1_narrabert_threshold_stability_real.png",
    )

    # --------------------------------------------------------
    # OUTPUT CONSOLE
    # --------------------------------------------------------

    print("=" * 80)
    print("ASSOCIAZIONI PRINCIPALI")
    print("=" * 80)

    printable = key_associations[
        key_associations["scope"].isin(
            ["FAKE", "REAL"]
        )
    ][
        [
            "threshold",
            "scope",
            "n",
            "narrative_dimension",
            "emotion",
            "spearman_rho",
            "p_value",
            "fdr_q_value",
        ]
    ]

    print(
        printable
        .round(5)
        .to_string(index=False)
    )

    print()
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(
        f"Cartella: {OUTPUT_DIR}"
    )

    print(
        "- rq1_narrabert_threshold_correlations.csv"
    )
    print(
        "- rq1_narrabert_threshold_key_associations.csv"
    )
    print(
        "- rq1_narrabert_threshold_top_associations.csv"
    )
    print(
        "- rq1_narrabert_threshold_stability_fake.png"
    )
    print(
        "- rq1_narrabert_threshold_stability_real.png"
    )


if __name__ == "__main__":
    main()