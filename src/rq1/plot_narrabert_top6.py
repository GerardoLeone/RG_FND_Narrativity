from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAZIONE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rq1_narrabert"
    / "rq1_narrabert_emotion_correlations.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rq1_narrabert"
    / "presentation"
)

TOP_N = 6


# ============================================================
# UTILITÀ
# ============================================================

def pretty_name(value: str) -> str:
    """Rende più leggibili i nomi."""
    return value.replace("_", " ").title()


def get_top_associations(
    df: pd.DataFrame,
    scope: str,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """
    Seleziona le N associazioni con |rho| maggiore
    all'interno del gruppo richiesto.
    """

    subset = df[df["scope"] == scope].copy()

    subset["abs_rho"] = subset["spearman_rho"].abs()

    subset = (
        subset
        .sort_values("abs_rho", ascending=False)
        .head(top_n)
        .copy()
    )

    subset["pair"] = (
        subset["narrative_dimension"].apply(pretty_name)
        + " × "
        + subset["emotion"].apply(pretty_name)
    )

    return subset


# ============================================================
# GRAFICO
# ============================================================

def plot_top_associations(
    subset: pd.DataFrame,
    scope: str,
    output_path: Path,
    x_min: float,
    x_max: float,
):
    """
    Crea un grafico orizzontale leggibile per la presentazione.
    """

    # Dal valore più piccolo al più grande:
    # il maggiore finirà in alto.
    subset = subset.sort_values("spearman_rho")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.barh(
        subset["pair"],
        subset["spearman_rho"],
    )

    ax.axvline(0, linewidth=1)

    # Stessa scala FAKE/REAL per permettere il confronto.
    ax.set_xlim(x_min, x_max)

    ax.set_xlabel("Spearman ρ")
    ax.set_ylabel("")

    ax.set_title(
        f"NarraBERT × GoEmotions — Top {TOP_N} ({scope})"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    # Scrive rho alla fine di ogni barra.
    padding = (x_max - x_min) * 0.015

    for bar, value in zip(
        bars,
        subset["spearman_rho"],
    ):
        if value >= 0:
            x = value + padding
            ha = "left"
        else:
            x = value - padding
            ha = "right"

        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha=ha,
            fontsize=10,
        )

    ax.tick_params(
        axis="y",
        labelsize=10,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"CSV non trovato:\n{INPUT_CSV}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT_CSV)

    required = {
        "scope",
        "narrative_dimension",
        "emotion",
        "spearman_rho",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Colonne mancanti nel CSV: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # TOP 6
    # --------------------------------------------------------

    fake = get_top_associations(
        df,
        "FAKE",
    )

    real = get_top_associations(
        df,
        "REAL",
    )

    # --------------------------------------------------------
    # SCALA COMUNE
    # --------------------------------------------------------

    all_values = pd.concat([
        fake["spearman_rho"],
        real["spearman_rho"],
    ])

    min_value = min(
        0,
        all_values.min(),
    )

    max_value = max(
        0,
        all_values.max(),
    )

    # Un po' di spazio per le etichette numeriche.
    margin = max(
        0.05,
        (max_value - min_value) * 0.15,
    )

    x_min = min_value - margin
    x_max = max_value + margin

    # --------------------------------------------------------
    # GRAFICI
    # --------------------------------------------------------

    plot_top_associations(
        fake,
        "FAKE",
        OUTPUT_DIR / "rq1_narrabert_top6_fake.png",
        x_min,
        x_max,
    )

    plot_top_associations(
        real,
        "REAL",
        OUTPUT_DIR / "rq1_narrabert_top6_real.png",
        x_min,
        x_max,
    )

    # --------------------------------------------------------
    # OUTPUT CONSOLE
    # --------------------------------------------------------

    print("=" * 70)
    print("TOP 6 FAKE")
    print("=" * 70)

    print(
        fake[
            [
                "narrative_dimension",
                "emotion",
                "spearman_rho",
            ]
        ]
        .sort_values(
            "spearman_rho",
            ascending=False,
        )
        .to_string(index=False)
    )

    print()

    print("=" * 70)
    print("TOP 6 REAL")
    print("=" * 70)

    print(
        real[
            [
                "narrative_dimension",
                "emotion",
                "spearman_rho",
            ]
        ]
        .sort_values(
            "spearman_rho",
            ascending=False,
        )
        .to_string(index=False)
    )

    print()
    print("Grafici salvati in:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()