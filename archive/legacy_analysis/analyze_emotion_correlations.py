from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


INPUT_CSV = Path("../../data/processed/dataset_narrativity_emotions.csv")
OUTPUT_DIR = Path("../old results/rq1")

NARRATIVITY_COLUMN = "narrativity_first_chunk"

EMOTION_COLUMNS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise"
]


def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    groups = {
        "ALL": df,
        "FAKE": df[df["label"] == "FAKE"],
        "REAL": df[df["label"] == "REAL"]
    }

    for group_name, group in groups.items():
        for emotion in EMOTION_COLUMNS:
            valid = group[[NARRATIVITY_COLUMN, emotion]].dropna()

            rho, p_value = spearmanr(
                valid[NARRATIVITY_COLUMN],
                valid[emotion]
            )

            rows.append({
                "group": group_name,
                "emotion": emotion,
                "n_articles": len(valid),
                "spearman_rho": rho,
                "p_value": p_value
            })

    return pd.DataFrame(rows)


def save_bar_chart(results: pd.DataFrame, group: str) -> None:
    group_results = (
        results[results["group"] == group]
        .sort_values("spearman_rho")
    )

    plt.figure(figsize=(10, 8))
    plt.barh(
        group_results["emotion"],
        group_results["spearman_rho"]
    )
    plt.axvline(0, linewidth=1)
    plt.xlabel("Spearman correlation with narrativity")
    plt.ylabel("Emotion")
    plt.title(f"Narrativity–emotion correlations: {group}")
    plt.tight_layout()

    output = OUTPUT_DIR / f"rq1_emotion_correlations_{group.lower()}.png"
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def save_fake_real_comparison(results: pd.DataFrame) -> None:
    comparison = (
        results[results["group"].isin(["FAKE", "REAL"])]
        .pivot(
            index="emotion",
            columns="group",
            values="spearman_rho"
        )
        .sort_values("FAKE")
    )

    ax = comparison.plot(
        kind="barh",
        figsize=(10, 9)
    )
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Spearman correlation with narrativity")
    ax.set_ylabel("Emotion")
    ax.set_title("Narrativity–emotion correlations: FAKE vs REAL")

    plt.tight_layout()
    output = OUTPUT_DIR / "rq1_fake_real_emotion_comparison.png"
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"File non trovato: {INPUT_CSV.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    results = calculate_correlations(df)

    results_path = OUTPUT_DIR / "rq1_emotion_correlations.csv"
    results.to_csv(results_path, index=False, encoding="utf-8-sig")

    for group in ["ALL", "FAKE", "REAL"]:
        save_bar_chart(results, group)

    save_fake_real_comparison(results)

    print(f"Risultati salvati in: {results_path.resolve()}")

    for group in ["FAKE", "REAL"]:
        current = results[results["group"] == group]

        print(f"\n{group} - correlazioni positive più forti:")
        print(
            current.nlargest(5, "spearman_rho")[
                ["emotion", "spearman_rho", "p_value"]
            ].round(4).to_string(index=False)
        )

        print(f"\n{group} - correlazioni negative più forti:")
        print(
            current.nsmallest(5, "spearman_rho")[
                ["emotion", "spearman_rho", "p_value"]
            ].round(4).to_string(index=False)
        )


if __name__ == "__main__":
    main()
