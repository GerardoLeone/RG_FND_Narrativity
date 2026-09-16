from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "rq1"
    / "primary"
    / "rq1_primary_correlations.csv"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "rq1"
    / "primary"
    / "rq1_primary_correlations.png"
)


def main():

    df = pd.read_csv(INPUT_PATH)

    # Teniamo solo l'analisi principale:
    # narrativity_first_chunk × emotion_strength_max
    primary = df[
        (df["narrativity_measure"] == "narrativity_first_chunk")
        & (df["emotion_strength_measure"] == "emotion_strength_max")
    ].copy()

    # Ordine desiderato nel grafico
    order = ["ALL", "FAKE", "REAL"]

    primary["scope"] = pd.Categorical(
        primary["scope"],
        categories=order,
        ordered=True
    )

    primary = primary.sort_values("scope")

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.barh(
        primary["scope"],
        primary["spearman_rho"]
    )

    # Linea dello zero
    ax.axvline(
        0,
        linewidth=1
    )

    # Valori numerici sulle barre
    for bar, value in zip(
        bars,
        primary["spearman_rho"]
    ):
        ax.text(
            value - 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="right"
        )

    ax.set_title(
        "RQ1 — Narrativity vs emotion strength"
    )

    ax.set_xlabel(
        "Spearman correlation"
    )

    ax.set_ylabel(
        "Group"
    )

    ax.set_xlim(
        -0.15,
        0.02
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Grafico salvato in:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()