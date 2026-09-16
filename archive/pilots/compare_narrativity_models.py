from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

STORYSEEKER_MODEL = "mariaantoniak/storyseeker"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = PROJECT_ROOT / "data" / "interim" / "narrabert_validation_100.csv"
OPTIONAL_ANNOTATIONS = (
    PROJECT_ROOT / "archive" / "validation" / "narrativity_validation_100_annotated.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "narrativity_comparison"
OUTPUT_ARTICLES = OUTPUT_DIR / "storyseeker_vs_narrabert_100.csv"
OUTPUT_CORR = OUTPUT_DIR / "model_correlations.csv"
OUTPUT_GROUPS = OUTPUT_DIR / "fake_real_comparison.csv"
OUTPUT_VALIDATION = OUTPUT_DIR / "manual_validation_metrics.csv"

MAX_LENGTH = 512
BATCH_SIZE = 32

NARRABERT_COLS = [
    "narrabert_focalization",
    "narrabert_emotion",
    "narrabert_cognition",
    "narrabert_change_of_state",
    "narrabert_conflict",
    "narrabert_concreteness",
    "narrabert_temporal_grounding",
    "narrabert_spatial_grounding",
    "narrabert_sensory",
]

EXPLORATORY_COMPOSITES = [
    "narrabert_mean_9d",
    "narrabert_mean_9d_norm",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_title(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta: effect size non-parametrica.
    Positive = valori di x tendenzialmente maggiori di y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    greater = 0
    lower = 0

    for value in x:
        greater += np.sum(value > y)
        lower += np.sum(value < y)

    return (greater - lower) / (len(x) * len(y))


@torch.inference_mode()
def storyseeker_scores(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> np.ndarray:
    outputs = []

    for start in tqdm(
        range(0, len(texts), BATCH_SIZE),
        desc="StorySeeker"
    ):
        batch = texts[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            k: v.to(device)
            for k, v in encoded.items()
        }

        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)

        id2label = model.config.id2label
        narrative_idx = None

        for idx, label in id2label.items():
            if str(label).upper() == "LABEL_1":
                narrative_idx = int(idx)
                break

        if narrative_idx is None:
            # fallback coerente con i test precedenti del progetto:
            # LABEL_1 = narrative
            narrative_idx = 1

        outputs.append(
            probs[:, narrative_idx].detach().cpu().numpy()
        )

    return np.concatenate(outputs)


def compare_fake_real(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []

    for col in columns:
        fake = df.loc[df["label"] == "FAKE", col].dropna().to_numpy()
        real = df.loc[df["label"] == "REAL", col].dropna().to_numpy()

        if len(fake) == 0 or len(real) == 0:
            continue

        u, p = mannwhitneyu(fake, real, alternative="two-sided")

        rows.append({
            "measure": col,
            "fake_n": len(fake),
            "real_n": len(real),
            "fake_mean": np.mean(fake),
            "real_mean": np.mean(real),
            "fake_median": np.median(fake),
            "real_median": np.median(real),
            "mean_difference_fake_minus_real": np.mean(fake) - np.mean(real),
            "mannwhitney_u": u,
            "p_value": p,
            "cliffs_delta_fake_vs_real": cliffs_delta(fake, real),
        })

    return pd.DataFrame(rows)


def compare_with_storyseeker(
    df: pd.DataFrame,
    columns: list[str]
) -> pd.DataFrame:
    rows = []

    for scope, sub in [
        ("ALL", df),
        ("FAKE", df[df["label"] == "FAKE"]),
        ("REAL", df[df["label"] == "REAL"]),
    ]:
        for col in columns:
            valid = sub[["storyseeker_score", col]].dropna()

            if len(valid) < 3:
                continue

            rho, p = spearmanr(
                valid["storyseeker_score"],
                valid[col]
            )

            rows.append({
                "scope": scope,
                "n": len(valid),
                "narrabert_measure": col,
                "spearman_rho_vs_storyseeker": rho,
                "p_value": p,
            })

    return pd.DataFrame(rows)


def manual_validation(
    df: pd.DataFrame,
    annotation_path: Path,
    measures: list[str]
) -> pd.DataFrame | None:
    if not annotation_path.exists():
        return None

    ann = pd.read_csv(annotation_path)

    manual_col = None
    for candidate in [
        "manual_label",
        "manual_annotation",
        "annotation",
        "human_label",
    ]:
        if candidate in ann.columns:
            manual_col = candidate
            break

    if manual_col is None or "title" not in ann.columns:
        warnings.warn(
            f"{annotation_path.name} trovato, ma non riconosco "
            "le colonne title/manual_label. Validazione manuale saltata."
        )
        return None

    ann = ann.copy()
    ann["_title_key"] = normalize_title(ann["title"])
    merged = df.copy()
    merged["_title_key"] = normalize_title(merged["title"])

    merged = merged.merge(
        ann[["_title_key", manual_col]],
        on="_title_key",
        how="inner",
    )

    labels = (
        merged[manual_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    mapping = {
        "NARRATIVE": 1,
        "NARRATIVO": 1,
        "1": 1,
        "NON_NARRATIVE": 0,
        "NON-NARRATIVE": 0,
        "NON NARRATIVE": 0,
        "NON_NARRATIVO": 0,
        "0": 0,
    }

    merged["_manual_binary"] = labels.map(mapping)
    merged = merged[merged["_manual_binary"].notna()].copy()

    if len(merged) < 10 or merged["_manual_binary"].nunique() < 2:
        warnings.warn(
            "Troppi pochi articoli annotati coincidenti per una "
            "validazione ROC-AUC affidabile."
        )
        return None

    y = merged["_manual_binary"].astype(int).to_numpy()
    rows = []

    for measure in measures:
        if measure not in merged.columns:
            continue

        scores = merged[measure].astype(float).to_numpy()
        auc = roc_auc_score(y, scores)

        rows.append({
            "measure": measure,
            "n_manual": len(merged),
            "roc_auc": auc,
            "orientation_free_auc": max(auc, 1.0 - auc),
        })

    return pd.DataFrame(rows).sort_values(
        "orientation_free_auc",
        ascending=False
    )


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"File non trovato: {INPUT_CSV}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA non disponibile. Lo script è configurato per usare la GPU."
        )

    device = torch.device("cuda:0")

    print("=" * 72)
    print("CONFRONTO STORYSEEKER vs NARRABERT")
    print("=" * 72)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Input: {INPUT_CSV}")
    print()

    df = pd.read_csv(INPUT_CSV)

    required = {"label", "title", "text"} | set(NARRABERT_COLS)
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Colonne mancanti nel CSV: {sorted(missing)}"
        )

    print(f"Articoli: {len(df)}")
    print(df["label"].value_counts().to_string())
    print()

    print(f"Caricamento StorySeeker: {STORYSEEKER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(STORYSEEKER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        STORYSEEKER_MODEL
    )
    model.to(device)
    model.eval()

    texts = (
        df["title"].fillna("").astype(str)
        + "\n\n"
        + df["text"].fillna("").astype(str)
    ).tolist()

    df["storyseeker_score"] = storyseeker_scores(
        texts,
        tokenizer,
        model,
        device,
    )

    measures = [
        "storyseeker_score",
        *NARRABERT_COLS,
        *[
            c for c in EXPLORATORY_COMPOSITES
            if c in df.columns
        ],
    ]

    corr_df = compare_with_storyseeker(
        df,
        [
            *NARRABERT_COLS,
            *[
                c for c in EXPLORATORY_COMPOSITES
                if c in df.columns
            ],
        ]
    )

    groups_df = compare_fake_real(df, measures)

    validation_df = manual_validation(
        df,
        OPTIONAL_ANNOTATIONS,
        measures
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_ARTICLES,
        index=False,
        encoding="utf-8"
    )

    corr_df.to_csv(
        OUTPUT_CORR,
        index=False,
        encoding="utf-8"
    )

    groups_df.to_csv(
        OUTPUT_GROUPS,
        index=False,
        encoding="utf-8"
    )

    if validation_df is not None:
        validation_df.to_csv(
            OUTPUT_VALIDATION,
            index=False,
            encoding="utf-8"
        )

    print("\n" + "=" * 72)
    print("1) STORYSEEKER - FAKE vs REAL")
    print("=" * 72)

    story_row = groups_df[
        groups_df["measure"] == "storyseeker_score"
    ].iloc[0]

    print(
        f"FAKE mean={story_row['fake_mean']:.4f} | "
        f"REAL mean={story_row['real_mean']:.4f} | "
        f"delta={story_row['cliffs_delta_fake_vs_real']:.4f} | "
        f"p={story_row['p_value']:.4g}"
    )

    print("\n" + "=" * 72)
    print("2) NARRABERT - differenze FAKE vs REAL")
    print("=" * 72)

    display_cols = [
        "measure",
        "fake_mean",
        "real_mean",
        "mean_difference_fake_minus_real",
        "cliffs_delta_fake_vs_real",
        "p_value",
    ]

    print(
        groups_df[
            groups_df["measure"].isin(NARRABERT_COLS)
        ][display_cols]
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("3) ACCORDO CON STORYSEEKER")
    print("=" * 72)

    print(
        corr_df[corr_df["scope"] == "ALL"][
            [
                "narrabert_measure",
                "spearman_rho_vs_storyseeker",
                "p_value",
            ]
        ]
        .sort_values(
            "spearman_rho_vs_storyseeker",
            ascending=False
        )
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("4) VALIDAZIONE MANUALE")
    print("=" * 72)

    if validation_df is None:
        print(
            "File di annotazione manuale non trovato/compatibile.\n"
            "Il confronto automatico è stato completato comunque.\n"
            "Se rimetti narrativity_validation_100_annotated.csv in:\n"
            f"{OPTIONAL_ANNOTATIONS.parent}\n"
            "e rilanci lo script, verrà calcolata anche la ROC-AUC."
        )
    else:
        print(
            validation_df.round(4).to_string(index=False)
        )

    print("\nOutput:")
    print(f"- {OUTPUT_ARTICLES}")
    print(f"- {OUTPUT_CORR}")
    print(f"- {OUTPUT_GROUPS}")

    if validation_df is not None:
        print(f"- {OUTPUT_VALIDATION}")


if __name__ == "__main__":
    main()