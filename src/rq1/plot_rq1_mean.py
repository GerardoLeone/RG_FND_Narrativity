from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAZIONE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rq1"
    / "primary"
    / "rq1_primary_correlations.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rq1"
    / "mean_analysis"
)

NARRATIVITY_MEASURE = "narrativity_mean"


# ============================================================
# GRAFICO PRINCIPALE
# narrativity_mean × emotion_strength_max
# ============================================================

def plot_mean_primary(df: pd.DataFrame):

    subset = df[
        (df["narrativity_measure"] == NARRATIVITY_MEASURE)
        &
        (
            df["emotion_strength_measure"]
            == "emotion_strength_max"
        )
    ].copy()

    order = ["ALL", "FAKE", "REAL"]

    subset["scope"] = pd.Categorical(
        subset["scope"],
        categories=order,
        ordered=True,
    )

    subset = subset.sort_values("scope")

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    bars = ax.barh(
        subset["scope"],
        subset["spearman_rho"],
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    # Valori delle correlazioni sulle barre
    for bar, value in zip(
        bars,
        subset["spearman_rho"]
    ):

        if value < 0:
            x = value - 0.004
            ha = "right"
        else:
            x = value + 0.004
            ha = "left"

        ax.text(
            x,
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha=ha,
        )

    ax.set_title(
        "RQ1 — Mean narrativity vs emotion strength"
    )

    ax.set_xlabel(
        "Spearman correlation"
    )

    ax.set_ylabel(
        "Group"
    )

    # Manteniamo la stessa scala
    # per facilitare il confronto.
    ax.set_xlim(
        -1,
        1
    )

    ax.grid(
        axis="x",
        alpha=0.2
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "rq1_mean_correlations.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Grafico principale salvato:\n"
        f"{output_path}"
    )


# ============================================================
# SENSITIVITY ANALYSIS
# narrativity_mean × 3 definizioni emotion strength
# ============================================================

def plot_mean_sensitivity(df: pd.DataFrame):

    subset = df[
        df["narrativity_measure"]
        == NARRATIVITY_MEASURE
    ].copy()

    scopes = [
        "ALL",
        "FAKE",
        "REAL"
    ]

    measures = [
        "emotion_strength_max",
        "emotion_strength_top3_mean",
        "emotion_strength_rms",
    ]

    labels = [
        "Max",
        "Top-3 mean",
        "RMS",
    ]

    y = np.arange(
        len(scopes)
    )

    height = 0.24

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for offset, measure, label in zip(
        [-height, 0, height],
        measures,
        labels,
    ):

        values = []

        for scope in scopes:

            row = subset[
                (subset["scope"] == scope)
                &
                (
                    subset[
                        "emotion_strength_measure"
                    ]
                    == measure
                )
            ]

            values.append(
                row.iloc[0][
                    "spearman_rho"
                ]
            )

        ax.barh(
            y + offset,
            values,
            height=height,
            label=label,
        )

    ax.axvline(
        0,
        linewidth=1
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        scopes
    )

    ax.set_xlim(
        -1,
        1
    )

    ax.set_xlabel(
        "Spearman correlation"
    )

    ax.set_ylabel(
        "Group"
    )

    ax.set_title(
        "RQ1 — Mean narrativity: sensitivity analysis"
    )

    ax.legend()

    ax.grid(
        axis="x",
        alpha=0.2
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "rq1_mean_sensitivity_analysis.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Sensitivity analysis salvata:\n"
        f"{output_path}"
    )


# ============================================================
# CONFRONTO FIRST CHUNK vs MEAN
# ============================================================

def plot_first_vs_mean(df: pd.DataFrame):

    subset = df[
        (
            df["narrativity_measure"]
            .isin([
                "narrativity_first_chunk",
                "narrativity_mean",
            ])
        )
        &
        (
            df["emotion_strength_measure"]
            == "emotion_strength_max"
        )
    ].copy()

    scopes = [
        "ALL",
        "FAKE",
        "REAL"
    ]

    y = np.arange(
        len(scopes)
    )

    height = 0.36

    first_values = []
    mean_values = []

    for scope in scopes:

        first_row = subset[
            (subset["scope"] == scope)
            &
            (
                subset["narrativity_measure"]
                == "narrativity_first_chunk"
            )
        ]

        mean_row = subset[
            (subset["scope"] == scope)
            &
            (
                subset["narrativity_measure"]
                == "narrativity_mean"
            )
        ]

        first_values.append(
            first_row.iloc[0]["spearman_rho"]
        )

        mean_values.append(
            mean_row.iloc[0]["spearman_rho"]
        )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        y - height / 2,
        first_values,
        height=height,
        label="First chunk",
    )

    ax.barh(
        y + height / 2,
        mean_values,
        height=height,
        label="Mean",
    )

    ax.axvline(
        0,
        linewidth=1
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        scopes
    )

    ax.set_xlim(
        -1,
        1
    )

    ax.set_xlabel(
        "Spearman correlation"
    )

    ax.set_ylabel(
        "Group"
    )

    ax.set_title(
        "RQ1 — First chunk vs mean narrativity"
    )

    ax.legend()

    ax.grid(
        axis="x",
        alpha=0.2
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "rq1_first_chunk_vs_mean.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Confronto first vs mean salvato:\n"
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"File non trovato:\n"
            f"{INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("RQ1 — ANALISI CON NARRATIVITY MEAN")
    print("=" * 72)

    plot_mean_primary(df)

    plot_mean_sensitivity(df)

    plot_first_vs_mean(df)

    print()
    print("Completato.")


if __name__ == "__main__":
    main()