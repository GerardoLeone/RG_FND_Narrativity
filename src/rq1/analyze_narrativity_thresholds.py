"""
Modulo: analyze_narrativity_thresholds.py

Scopo:
verificare se, selezionando articoli progressivamente piu' narrativi,
le emozioni diventano piu' intense e se il risultato dipende dallo
sbilanciamento FAKE/REAL del dataset.

Analisi principali:
1. applica 5 threshold cumulative a narrativity_mean;
2. misura la composizione FAKE/REAL a ogni threshold;
3. calcola emotion strength media per ALL, FAKE, REAL e una media
   bilanciata 50/50 tra FAKE e REAL;
4. calcola la media di ciascuna delle 27 emozioni a ogni threshold;
5. calcola Spearman narrativita' x emotion strength dentro ciascun
   sottoinsieme selezionato;
6. calcola Spearman narrativita' x singola emozione;
7. applica Benjamini-Hochberg alle correlazioni multiple;
8. genera CSV e grafici riassuntivi.

Nota metodologica:
le threshold sono cumulative (es. >= 0.70 contiene anche gli articoli
>= 0.80 e >= 0.90). Per questo il trend tra threshold va interpretato
soprattutto in modo descrittivo. Le correlazioni Spearman vengono invece
calcolate sui singoli articoli presenti in ciascun sottoinsieme.
"""

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

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_narrativity_emotions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rq1"
    / "threshold_analysis"
)

# Usiamo la narrativita' media dell'intero articolo.
NARRATIVITY_COLUMN = "narrativity_mean"

# Cinque threshold richieste per l'analisi.
THRESHOLDS = [0.30, 0.50, 0.70, 0.80, 0.90]

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


# ============================================================
# UTILITA'
# ============================================================

def normalize_label(series: pd.Series) -> pd.Series:
    """Uniforma la label in FAKE / REAL."""

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
    """Correzione Benjamini-Hochberg della False Discovery Rate."""

    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    valid_mask = np.isfinite(p)

    if not valid_mask.any():
        return q

    pv = p[valid_mask]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)

    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[valid_mask] = restored

    return q


def safe_spearman(x: pd.Series, y: pd.Series):
    """Calcola Spearman dopo aver eliminato NaN e gestisce casi degeneri."""

    valid = pd.concat([x, y], axis=1).dropna()

    if len(valid) < 4:
        return len(valid), np.nan, np.nan

    if valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return len(valid), np.nan, np.nan

    rho, p = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])

    return len(valid), float(rho), float(p)


# ============================================================
# PREPARAZIONE DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:
    """Carica il dataset e costruisce le misure di emotion strength."""

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"File non trovato:\n{INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required = {
        "label",
        NARRATIVITY_COLUMN,
        *EMOTION_COLUMNS,
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Colonne mancanti:\n" + "\n".join(missing)
        )

    df = df.copy()
    df["label"] = normalize_label(df["label"])
    df = df[df["label"].isin(["FAKE", "REAL"])].copy()

    emotion_values = (
        df[EMOTION_COLUMNS]
        .astype(float)
        .to_numpy()
    )

    # Misura primaria: emozione non-neutrale piu' intensa.
    df["emotion_strength_max"] = np.max(
        emotion_values,
        axis=1,
    )

    # Sensitivity measures, mantenute anche in questa analisi.
    df["emotion_strength_top3_mean"] = np.mean(
        np.sort(emotion_values, axis=1)[:, -3:],
        axis=1,
    )

    df["emotion_strength_rms"] = np.sqrt(
        np.mean(
            np.square(emotion_values),
            axis=1,
        )
    )

    return df


# ============================================================
# SUMMARY DELLE THRESHOLD
# ============================================================

def build_threshold_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per ogni threshold calcola:
    - numerosita' e composizione FAKE/REAL;
    - emotion strength per ALL, FAKE, REAL;
    - media bilanciata 50/50 tra FAKE e REAL.
    """

    strength_cols = [
        "emotion_strength_max",
        "emotion_strength_top3_mean",
        "emotion_strength_rms",
    ]

    rows = []

    for threshold in THRESHOLDS:
        selected = df[
            df[NARRATIVITY_COLUMN] >= threshold
        ].copy()

        fake = selected[selected["label"] == "FAKE"]
        real = selected[selected["label"] == "REAL"]

        n_total = len(selected)
        fake_n = len(fake)
        real_n = len(real)

        fake_share = fake_n / n_total if n_total else np.nan
        real_share = real_n / n_total if n_total else np.nan

        # ALL / FAKE / REAL
        for scope, subset in [
            ("ALL", selected),
            ("FAKE", fake),
            ("REAL", real),
        ]:
            row = {
                "threshold": threshold,
                "scope": scope,
                "n": len(subset),
                "fake_n": fake_n,
                "real_n": real_n,
                "fake_share": fake_share,
                "real_share": real_share,
            }

            for col in strength_cols:
                row[f"{col}_mean"] = subset[col].mean()
                row[f"{col}_median"] = subset[col].median()

            rows.append(row)

        # BALANCED_50_50:
        # nessun resampling. Ogni classe pesa esattamente 50%.
        balanced_row = {
            "threshold": threshold,
            "scope": "BALANCED_50_50",
            "n": np.nan,
            "fake_n": fake_n,
            "real_n": real_n,
            "fake_share": 0.50,
            "real_share": 0.50,
        }

        for col in strength_cols:
            balanced_row[f"{col}_mean"] = np.mean([
                fake[col].mean(),
                real[col].mean(),
            ])

            balanced_row[f"{col}_median"] = np.nan

        rows.append(balanced_row)

    return pd.DataFrame(rows)


# ============================================================
# MEDIA DELLE 27 EMOZIONI
# ============================================================

def build_emotion_means(df: pd.DataFrame) -> pd.DataFrame:
    """Media di ogni emozione alle diverse threshold."""

    rows = []

    for threshold in THRESHOLDS:
        selected = df[
            df[NARRATIVITY_COLUMN] >= threshold
        ].copy()

        fake = selected[selected["label"] == "FAKE"]
        real = selected[selected["label"] == "REAL"]

        scopes = [
            ("ALL", selected),
            ("FAKE", fake),
            ("REAL", real),
        ]

        for scope, subset in scopes:
            for emotion in EMOTION_COLUMNS:
                rows.append({
                    "threshold": threshold,
                    "scope": scope,
                    "emotion": emotion,
                    "n": len(subset),
                    "mean_probability": subset[emotion].mean(),
                    "median_probability": subset[emotion].median(),
                })

        # Versione bilanciata 50/50 per emozione.
        for emotion in EMOTION_COLUMNS:
            rows.append({
                "threshold": threshold,
                "scope": "BALANCED_50_50",
                "emotion": emotion,
                "n": np.nan,
                "mean_probability": np.mean([
                    fake[emotion].mean(),
                    real[emotion].mean(),
                ]),
                "median_probability": np.nan,
            })

    return pd.DataFrame(rows)


# ============================================================
# CORRELAZIONE NARRATIVITA' x EMOTION STRENGTH
# ============================================================

def build_strength_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dentro ogni subset >= threshold calcola Spearman tra:

        narrativity_mean
            x
        emotion_strength_max

    Separatamente per ALL, FAKE e REAL.
    """

    rows = []

    for threshold in THRESHOLDS:
        selected = df[
            df[NARRATIVITY_COLUMN] >= threshold
        ].copy()

        for scope, subset in [
            ("ALL", selected),
            ("FAKE", selected[selected["label"] == "FAKE"]),
            ("REAL", selected[selected["label"] == "REAL"]),
        ]:
            n, rho, p = safe_spearman(
                subset[NARRATIVITY_COLUMN],
                subset["emotion_strength_max"],
            )

            rows.append({
                "threshold": threshold,
                "scope": scope,
                "n": n,
                "spearman_rho": rho,
                "p_value": p,
            })

    result = pd.DataFrame(rows)
    result["fdr_q_value"] = np.nan

    # Corregge i 5 test separatamente in ALL / FAKE / REAL.
    for scope in ["ALL", "FAKE", "REAL"]:
        mask = result["scope"] == scope
        result.loc[mask, "fdr_q_value"] = benjamini_hochberg(
            result.loc[mask, "p_value"].to_numpy()
        )

    result["significant_fdr_0_05"] = (
        result["fdr_q_value"] < 0.05
    )

    return result


# ============================================================
# CORRELAZIONE NARRATIVITA' x SINGOLA EMOZIONE
# ============================================================

def build_emotion_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per ciascuna threshold calcola Spearman tra narrativity_mean
    e ciascuna delle 27 probabilita' GoEmotions.
    """

    rows = []

    for threshold in THRESHOLDS:
        selected = df[
            df[NARRATIVITY_COLUMN] >= threshold
        ].copy()

        for scope, subset in [
            ("ALL", selected),
            ("FAKE", selected[selected["label"] == "FAKE"]),
            ("REAL", selected[selected["label"] == "REAL"]),
        ]:
            for emotion in EMOTION_COLUMNS:
                n, rho, p = safe_spearman(
                    subset[NARRATIVITY_COLUMN],
                    subset[emotion],
                )

                rows.append({
                    "threshold": threshold,
                    "scope": scope,
                    "emotion": emotion,
                    "n": n,
                    "spearman_rho": rho,
                    "p_value": p,
                })

    result = pd.DataFrame(rows)
    result["fdr_q_value"] = np.nan

    # A ogni threshold e per ogni gruppo correggiamo i 27 test.
    for threshold in THRESHOLDS:
        for scope in ["ALL", "FAKE", "REAL"]:
            mask = (
                (result["threshold"] == threshold)
                & (result["scope"] == scope)
            )

            result.loc[mask, "fdr_q_value"] = benjamini_hochberg(
                result.loc[mask, "p_value"].to_numpy()
            )

    result["significant_fdr_0_05"] = (
        result["fdr_q_value"] < 0.05
    )

    return result


# ============================================================
# TOP EMOTIONS
# ============================================================

def build_top_emotions(emotion_means: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Prime emozioni per intensita' media a ogni threshold."""

    rows = []

    for threshold in THRESHOLDS:
        for scope in ["ALL", "FAKE", "REAL", "BALANCED_50_50"]:
            subset = emotion_means[
                (emotion_means["threshold"] == threshold)
                & (emotion_means["scope"] == scope)
            ].sort_values(
                "mean_probability",
                ascending=False,
            ).head(top_n)

            for rank, (_, row) in enumerate(subset.iterrows(), start=1):
                rows.append({
                    "threshold": threshold,
                    "scope": scope,
                    "rank": rank,
                    "emotion": row["emotion"],
                    "mean_probability": row["mean_probability"],
                })

    return pd.DataFrame(rows)


# ============================================================
# GRAFICI
# ============================================================

def plot_class_composition(summary: pd.DataFrame):
    """Percentuale FAKE / REAL mantenuta a ogni threshold."""

    all_rows = (
        summary[summary["scope"] == "ALL"]
        .sort_values("threshold")
    )

    x = np.arange(len(all_rows))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x,
        all_rows["fake_share"] * 100,
        label="FAKE",
    )

    ax.bar(
        x,
        all_rows["real_share"] * 100,
        bottom=all_rows["fake_share"] * 100,
        label="REAL",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([
        f">= {t:.2f}"
        for t in all_rows["threshold"]
    ])

    ax.set_ylim(0, 100)
    ax.set_xlabel("Narrativity threshold")
    ax.set_ylabel("Selected articles (%)")
    ax.set_title("RQ1 threshold analysis — FAKE/REAL composition")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "rq1_threshold_class_composition.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_strength_trend(summary: pd.DataFrame):
    """Emotion strength media al crescere della threshold."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for scope in [
        "ALL",
        "FAKE",
        "REAL",
        "BALANCED_50_50",
    ]:
        subset = (
            summary[summary["scope"] == scope]
            .sort_values("threshold")
        )

        ax.plot(
            subset["threshold"],
            subset["emotion_strength_max_mean"],
            marker="o",
            linewidth=2,
            label=scope,
        )

    ax.set_xlabel("Narrativity threshold")
    ax.set_ylabel("Mean emotion strength (max)")
    ax.set_title("RQ1 threshold analysis — Emotion strength")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "rq1_threshold_emotion_strength.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_correlation_stability(correlations: pd.DataFrame):
    """Spearman narrativity x emotion strength dentro ogni threshold."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for scope in ["ALL", "FAKE", "REAL"]:
        subset = (
            correlations[correlations["scope"] == scope]
            .sort_values("threshold")
        )

        ax.plot(
            subset["threshold"],
            subset["spearman_rho"],
            marker="o",
            linewidth=2,
            label=scope,
        )

    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Narrativity threshold")
    ax.set_ylabel("Spearman rho")
    ax.set_title(
        "RQ1 threshold analysis — Narrativity vs emotion strength"
    )
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "rq1_threshold_spearman_stability.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_emotion_heatmap(
    emotion_means: pd.DataFrame,
    scope: str,
    output_name: str,
    vmax: float,
):
    """Heatmap delle 27 emozioni alle 5 threshold."""

    subset = emotion_means[
        emotion_means["scope"] == scope
    ]

    matrix = subset.pivot(
        index="emotion",
        columns="threshold",
        values="mean_probability",
    )

    matrix = matrix.reindex(EMOTION_COLUMNS)
    matrix = matrix.reindex(columns=THRESHOLDS)

    fig, ax = plt.subplots(
        figsize=(10, 11)
    )

    im = ax.imshow(
        matrix.values,
        aspect="auto",
        vmin=0,
        vmax=vmax,
        cmap="viridis",
    )

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([
        f">= {t:.2f}"
        for t in matrix.columns
    ])

    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    ax.set_xlabel("Narrativity threshold")
    ax.set_ylabel("Emotion")
    ax.set_title(
        f"RQ1 threshold analysis — Mean emotion probability ({scope})"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean GoEmotions probability")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / output_name,
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("RQ1 — NARRATIVITY THRESHOLD ANALYSIS")
    print("=" * 78)

    print(f"Narrativity measure: {NARRATIVITY_COLUMN}")
    print(f"Thresholds: {THRESHOLDS}")
    print()

    df = load_dataset()

    print(f"Articoli totali: {len(df):,}")
    print(df["label"].value_counts().to_string())
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = build_threshold_summary(df)
    emotion_means = build_emotion_means(df)
    strength_correlations = build_strength_correlations(df)
    emotion_correlations = build_emotion_correlations(df)
    top_emotions = build_top_emotions(emotion_means, top_n=5)

    # --------------------------------------------------------
    # SALVATAGGIO CSV
    # --------------------------------------------------------

    summary.to_csv(
        OUTPUT_DIR / "rq1_threshold_summary.csv",
        index=False,
        encoding="utf-8",
    )

    emotion_means.to_csv(
        OUTPUT_DIR / "rq1_threshold_emotion_means.csv",
        index=False,
        encoding="utf-8",
    )

    strength_correlations.to_csv(
        OUTPUT_DIR / "rq1_threshold_strength_correlations.csv",
        index=False,
        encoding="utf-8",
    )

    emotion_correlations.to_csv(
        OUTPUT_DIR / "rq1_threshold_emotion_correlations.csv",
        index=False,
        encoding="utf-8",
    )

    top_emotions.to_csv(
        OUTPUT_DIR / "rq1_threshold_top_emotions.csv",
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # GRAFICI
    # --------------------------------------------------------

    plot_class_composition(summary)
    plot_strength_trend(summary)
    plot_correlation_stability(strength_correlations)

    # Stessa scala colore per FAKE e REAL, cosi' sono confrontabili.
    max_emotion_probability = (
        emotion_means[
            emotion_means["scope"].isin(["FAKE", "REAL"])
        ]["mean_probability"]
        .max()
    )

    plot_emotion_heatmap(
        emotion_means,
        "FAKE",
        "rq1_threshold_emotions_fake.png",
        vmax=max_emotion_probability,
    )

    plot_emotion_heatmap(
        emotion_means,
        "REAL",
        "rq1_threshold_emotions_real.png",
        vmax=max_emotion_probability,
    )

    # --------------------------------------------------------
    # OUTPUT CONSOLE
    # --------------------------------------------------------

    print("=" * 78)
    print("COMPOSIZIONE E EMOTION STRENGTH")
    print("=" * 78)

    printable = summary[
        summary["scope"].isin([
            "ALL",
            "FAKE",
            "REAL",
            "BALANCED_50_50",
        ])
    ][
        [
            "threshold",
            "scope",
            "n",
            "fake_n",
            "real_n",
            "fake_share",
            "emotion_strength_max_mean",
        ]
    ]

    print(
        printable
        .round(5)
        .to_string(index=False)
    )

    print()
    print("=" * 78)
    print("SPEARMAN DENTRO LE THRESHOLD")
    print("=" * 78)

    print(
        strength_correlations[
            [
                "threshold",
                "scope",
                "n",
                "spearman_rho",
                "p_value",
                "fdr_q_value",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print()
    print("=" * 78)
    print("OUTPUT")
    print("=" * 78)
    print(f"Cartella: {OUTPUT_DIR}")
    print("- rq1_threshold_summary.csv")
    print("- rq1_threshold_emotion_means.csv")
    print("- rq1_threshold_strength_correlations.csv")
    print("- rq1_threshold_emotion_correlations.csv")
    print("- rq1_threshold_top_emotions.csv")
    print("- rq1_threshold_class_composition.png")
    print("- rq1_threshold_emotion_strength.png")
    print("- rq1_threshold_spearman_stability.png")
    print("- rq1_threshold_emotions_fake.png")
    print("- rq1_threshold_emotions_real.png")

    print(
        "\nNota: BALANCED_50_50 non e' un nuovo dataset campionato. "
        "E' una media in cui FAKE e REAL pesano ciascuno il 50%, "
        "utile per verificare se il trend ALL e' dovuto alla composizione "
        "variabile delle due classi."
    )


if __name__ == "__main__":
    main()
