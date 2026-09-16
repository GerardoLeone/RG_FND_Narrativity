from pathlib import Path
import random

import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline


MODEL_NAME = "mariaantoniak/storyseeker"
NARRATIVE_LABEL = "LABEL_1"

# Modifica questi percorsi solo se i CSV si trovano altrove.
FAKE_CSV = Path("../../data/raw/Fake.csv")
TRUE_CSV = Path("../../data/raw/True.csv")
OUTPUT_CSV = Path("../data/processed/narrativity_validation_sample.csv")

SAMPLES_PER_CLASS = 10
MIN_WORDS = 100
RANDOM_SEED = 42

# RoBERTa supporta al massimo 512 token inclusi i token speciali.
MAX_TOKENS = 512
CHUNK_STRIDE = 384


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA non disponibile. Verifica di avere PyTorch con supporto CUDA."
        )

    print(f"GPU rilevata: {torch.cuda.get_device_name(0)}")
    print("Caricamento StorySeeker...")

    classifier = pipeline(
        task="text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=0
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Modello caricato.")
    print("Mappatura label:", classifier.model.config.id2label)
    return classifier, tokenizer


def load_and_clean_dataset(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"File non trovato: {path.resolve()}\n"
            "Modifica FAKE_CSV o TRUE_CSV nello script."
        )

    df = pd.read_csv(path)

    required_columns = {"title", "text", "subject", "date"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Nel file {path} mancano le colonne: {sorted(missing_columns)}"
        )

    df = df.copy()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["label"] = label
    df["word_count"] = df["text"].str.split().str.len()

    rows_before = len(df)

    # Elimina testi vuoti, troppo brevi e duplicati esatti.
    df = df[df["text"] != ""]
    df = df[df["word_count"] >= MIN_WORDS]
    df = df.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)

    print(
        f"{label}: {rows_before} righe iniziali, "
        f"{len(df)} disponibili dopo la pulizia."
    )
    return df


def create_balanced_sample(fake_df: pd.DataFrame,
                           real_df: pd.DataFrame) -> pd.DataFrame:
    if len(fake_df) < SAMPLES_PER_CLASS or len(real_df) < SAMPLES_PER_CLASS:
        raise ValueError("Non ci sono abbastanza articoli per il campione richiesto.")

    fake_sample = fake_df.sample(
        n=SAMPLES_PER_CLASS,
        random_state=RANDOM_SEED
    )
    real_sample = real_df.sample(
        n=SAMPLES_PER_CLASS,
        random_state=RANDOM_SEED
    )

    sample = pd.concat([fake_sample, real_sample], ignore_index=True)

    # Mescola l'ordine per rendere meno immediata la label durante il controllo.
    sample = sample.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    sample.insert(0, "sample_id", range(1, len(sample) + 1))
    return sample


def split_into_token_chunks(tokenizer, title: str, text: str) -> list[str]:
    # Il titolo viene incluso una sola volta, all'inizio dell'articolo.
    full_text = f"{title}. {text}".strip()

    token_ids = tokenizer.encode(
        full_text,
        add_special_tokens=False
    )

    # Riserviamo due posizioni per i token speciali di RoBERTa.
    content_length = MAX_TOKENS - 2
    chunks = []

    for start in range(0, len(token_ids), CHUNK_STRIDE):
        chunk_ids = token_ids[start:start + content_length]

        if not chunk_ids:
            break

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        chunks.append(chunk_text)

        if start + content_length >= len(token_ids):
            break

    return chunks


def get_narrative_probability(results: list[dict]) -> float:
    for result in results:
        if result["label"] == NARRATIVE_LABEL:
            return float(result["score"])

    raise RuntimeError(
        f"Label {NARRATIVE_LABEL} non trovata nell'output: {results}"
    )


def analyze_article(classifier, tokenizer, title: str, text: str) -> dict:
    chunks = split_into_token_chunks(tokenizer, title, text)

    if not chunks:
        raise ValueError("Non è stato possibile creare chunk dal testo.")

    # Elaborazione in batch sulla GPU.
    predictions = classifier(
        chunks,
        truncation=True,
        max_length=MAX_TOKENS,
        top_k=None,
        batch_size=8
    )

    chunk_scores = [
        get_narrative_probability(result)
        for result in predictions
    ]

    return {
        "chunk_count": len(chunk_scores),
        "narrativity_first_chunk": chunk_scores[0],
        "narrativity_mean": sum(chunk_scores) / len(chunk_scores),
        "narrativity_max": max(chunk_scores),
        "chunk_scores": "|".join(f"{score:.4f}" for score in chunk_scores)
    }


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    fake_df = load_and_clean_dataset(FAKE_CSV, "FAKE")
    real_df = load_and_clean_dataset(TRUE_CSV, "REAL")
    sample = create_balanced_sample(fake_df, real_df)

    classifier, tokenizer = load_model()

    analysis_rows = []

    for index, row in sample.iterrows():
        print(
            f"[{index + 1}/{len(sample)}] "
            f"Analisi: {row['title'][:70]}"
        )

        scores = analyze_article(
            classifier=classifier,
            tokenizer=tokenizer,
            title=row["title"],
            text=row["text"]
        )

        analysis_rows.append(scores)

    scores_df = pd.DataFrame(analysis_rows)
    result = pd.concat([sample, scores_df], axis=1)

    # Estratto utile per il controllo manuale nel foglio CSV.
    result["text_preview"] = (
        result["text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 700)
    )

    # Colonne manuali da compilare dopo la lettura.
    result["manual_narrative_label"] = ""
    result["manual_notes"] = ""

    output_columns = [
        "sample_id",
        "label",
        "title",
        "subject",
        "date",
        "word_count",
        "chunk_count",
        "narrativity_first_chunk",
        "narrativity_mean",
        "narrativity_max",
        "chunk_scores",
        "text_preview",
        "manual_narrative_label",
        "manual_notes",
        "text"
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result[output_columns].to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTest completato.")
    print(f"Risultati salvati in: {OUTPUT_CSV.resolve()}")
    print("\nMedia del punteggio di narratività per classe:")
    print(
        result.groupby("label")["narrativity_mean"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(4)
    )


if __name__ == "__main__":
    main()
