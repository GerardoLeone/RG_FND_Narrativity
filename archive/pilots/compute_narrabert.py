from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download
from tqdm import tqdm

MODEL_ID = "teagrjohnson/narrative-likert-roberta"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAKE_CSV = PROJECT_ROOT / "data" / "raw" / "Fake.csv"
REAL_CSV = PROJECT_ROOT / "data" / "raw" / "True.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_CSV = OUTPUT_DIR / "narrabert_validation_100.csv"

N_PER_CLASS = 50
MIN_WORDS = 100
MAX_LENGTH = 256
BATCH_SIZE = 32
SEED = 42

DIMENSIONS = [
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

class NarrativeRoBERTa(nn.Module):
    def __init__(self, model_name: str, n_dims: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_dims)])

    def forward(self, input_ids, attention_mask):
        cls = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        return torch.cat([head(cls) for head in self.heads], dim=1)

def load_narrabert(device: torch.device):
    print(f"Download/caricamento modello: {MODEL_ID}")

    local_dir = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            allow_patterns=[
                "model.pt",
                "tokenizer/*",
                "config.json",
                "*.json",
                "*.txt",
                "merges.txt",
                "vocab.json",
            ],
        )
    )

    tokenizer_dir = local_dir / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(local_dir)

    model = NarrativeRoBERTa("roberta-base", len(DIMENSIONS))

    state_path = local_dir / "model.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"model.pt non trovato nello snapshot: {state_path}")

    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return tokenizer, model

def clean_dataset(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"title", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: colonne mancanti: {sorted(missing)}")

    df = df.copy()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()

    df = df[df["text"] != ""].copy()
    df["word_count"] = df["text"].str.split().str.len().fillna(0).astype(int)
    df = df[df["word_count"] >= MIN_WORDS].copy()
    df = df.drop_duplicates(subset=["title", "text"]).copy()
    df["label"] = label

    return df.reset_index(drop=True)

def build_balanced_sample() -> pd.DataFrame:
    print("Caricamento dataset...")

    fake = clean_dataset(FAKE_CSV, "FAKE")
    real = clean_dataset(REAL_CSV, "REAL")

    print(f"FAKE disponibili dopo cleaning: {len(fake):,}")
    print(f"REAL disponibili dopo cleaning: {len(real):,}")

    fake_sample = fake.sample(n=min(N_PER_CLASS, len(fake)), random_state=SEED)
    real_sample = real.sample(n=min(N_PER_CLASS, len(real)), random_state=SEED)

    sample = pd.concat([fake_sample, real_sample], ignore_index=True)
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample.insert(0, "sample_id", range(1, len(sample) + 1))

    return sample

@torch.inference_mode()
def predict_batch(texts, tokenizer, model, device):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask"}
    }

    outputs = model(**encoded)
    return outputs.detach().cpu()

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile. Questo script richiede la GPU.")

    device = torch.device("cuda:0")

    print("=" * 64)
    print("NarraBERT validation - 100 articoli")
    print("=" * 64)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Modello: {MODEL_ID}")
    print(f"Max length: {MAX_LENGTH}")
    print()

    sample = build_balanced_sample()
    tokenizer, model = load_narrabert(device)

    texts = (
        sample["title"].fillna("").astype(str)
        + "\n\n"
        + sample["text"].fillna("").astype(str)
    ).tolist()

    all_scores = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Inferenza NarraBERT"):
        batch = texts[start:start + BATCH_SIZE]
        scores = predict_batch(batch, tokenizer, model, device)
        all_scores.append(scores)

    scores = torch.cat(all_scores, dim=0).numpy()

    if scores.shape[1] != len(DIMENSIONS):
        raise RuntimeError(
            f"Output inatteso: {scores.shape}. Attese {len(DIMENSIONS)} dimensioni."
        )

    for idx, dim in enumerate(DIMENSIONS):
        sample[f"narrabert_{dim}"] = scores[:, idx]

    dim_cols = [f"narrabert_{d}" for d in DIMENSIONS]
    sample["narrabert_mean_9d"] = sample[dim_cols].mean(axis=1)
    sample["narrabert_mean_9d_norm"] = (
        (sample["narrabert_mean_9d"] - 1.0) / 4.0
    ).clip(0.0, 1.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print()
    print("=" * 64)
    print("RISULTATI")
    print("=" * 64)

    print("\nMedia delle 9 dimensioni per classe:")
    print(
        sample.groupby("label")["narrabert_mean_9d"]
        .agg(["count", "mean", "median", "std"])
        .round(4)
    )

    print("\nMedie per dimensione:")
    print(sample.groupby("label")[dim_cols].mean().T.round(4))

    print(f"\nOutput salvato in:\n{OUTPUT_CSV}")
    print(
        "\nNOTA: narrabert_mean_9d è uno score composito esplorativo. "
        "Non lo useremo ancora come misura definitiva della RQ1."
    )

if __name__ == "__main__":
    main()