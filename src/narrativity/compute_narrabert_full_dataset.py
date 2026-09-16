"""
Modulo: compute_narrabert_full_dataset.py

Scopo:
applicare NarraBERT a tutto il dataset FAKE + REAL.
In quali aspetti è narrativo?
StorySeeker invece si chiede "QUANTO è narrativo un testo?"

NarraBERT non produce un singolo punteggio di narratività.
Restituisce invece 9 dimensioni narrative (caratteristiche narrative):

- focalization
- emotion
- cognition
- change_of_state
- conflict
- concreteness
- temporal_grounding
- spatial_grounding
- sensory

Poiché gli articoli possono essere lunghi, vengono divisi in chunk.
Per ogni chunk si calcolano le 9 dimensioni.
Per ogni dimensione vengono poi calcolati:

- first: valore del primo chunk
- mean: media di tutti i chunk
- max: valore massimo tra i chunk
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download
from tqdm import tqdm


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Repository Hugging Face del modello NarraBERT.
MODEL_ID = "teagrjohnson/narrative-likert-roberta"

# Root del progetto.
#
# Il file si trova in:
# src/narrativity/compute_narrabert_full_dataset.py
#
# parents[2] risale quindi fino a RG_FND/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset originali.
FAKE_CSV = PROJECT_ROOT / "data" / "raw" / "Fake.csv"
REAL_CSV = PROJECT_ROOT / "data" / "raw" / "True.csv"

# Directory e file finale.
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"

OUTPUT_CSV = (
    OUTPUT_DIR
    / "dataset_with_narrativity_narrabert.csv"
)

# Gli articoli con meno di 100 parole vengono eliminati.
MIN_WORDS = 100

# NarraBERT è stato configurato per sequenze da 256 token.
MAX_LENGTH = 256

# Lasciamo spazio ai token speciali di RoBERTa.
CONTENT_TOKENS = 254

# Ogni chunk successivo inizia 192 token dopo il precedente.
# Poiché ogni chunk contiene 254 token, esiste una sovrapposizione.
STRIDE = 192

# Numero di chunk inviati contemporaneamente alla GPU.
BATCH_SIZE = 64

# Ogni 5000 articoli viene salvato un file temporaneo,
# utile in caso di interruzioni.
CHECKPOINT_EVERY = 5000


# Le 9 dimensioni restituite dal modello.
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


# ============================================================
# ARCHITETTURA DEL MODELLO
# ============================================================

class NarrativeRoBERTa(nn.Module):
    """
    Ricostruisce l'architettura usata dal modello NarraBERT.

    Struttura:

        testo
          ↓
        RoBERTa
          ↓
        rappresentazione [CLS]
          ↓
        9 teste lineari
          ↓
        9 punteggi narrativi

    Ogni testa lineare produce il valore di una dimensione narrativa.
    """

    def __init__(
        self,
        model_name: str,
        n_dims: int
    ):
        super().__init__()

        # Backbone RoBERTa.
        #
        # AutoModel restituisce le rappresentazioni interne,
        # senza una testa di classificazione già pronta.
        self.backbone = AutoModel.from_pretrained(
            model_name
        )

        # Dimensione del vettore nascosto di RoBERTa.
        hidden = self.backbone.config.hidden_size

        # Creiamo una testa lineare indipendente
        # per ciascuna delle 9 dimensioni.
        self.heads = nn.ModuleList(
            [
                nn.Linear(hidden, 1)
                for _ in range(n_dims)
            ]
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):
        """
        Forward pass del modello.

        Il primo token della sequenza viene usato
        come rappresentazione globale del testo.
        """

        # last_hidden_state ha forma:
        #
        # batch × sequence_length × hidden_size
        #
        # [:, 0, :] seleziona la rappresentazione
        # del primo token della sequenza.
        cls = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]

        # Ogni testa produce un singolo valore.
        #
        # torch.cat li combina in un vettore:
        #
        # batch × 9
        return torch.cat(
            [
                head(cls)
                for head in self.heads
            ],
            dim=1,
        )


# ============================================================
# CARICAMENTO MODELLO
# ============================================================

def load_model(
    device: torch.device
):
    """
    Scarica/carica NarraBERT e ricostruisce
    l'architettura con i pesi addestrati.

    Returns
    -------
    tokenizer
        Tokenizer RoBERTa.

    model
        Modello NarrativeRoBERTa pronto per inferenza.
    """

    print(
        f"Caricamento modello: "
        f"{MODEL_ID}"
    )

    # Scarica dal repository solo i file necessari.
    #
    # snapshot_download usa la cache Hugging Face,
    # quindi dopo il primo download non serve
    # riscaricare tutto.
    local_dir = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            allow_patterns=[
                "model.pt",
                "tokenizer/*",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "merges.txt",
                "vocab.json",
            ],
        )
    )

    # Alcuni repository salvano il tokenizer
    # in una sottocartella tokenizer/.
    tokenizer_dir = local_dir / "tokenizer"

    if tokenizer_dir.exists():
        tokenizer = (
            AutoTokenizer
            .from_pretrained(tokenizer_dir)
        )

    else:
        tokenizer = (
            AutoTokenizer
            .from_pretrained(local_dir)
        )

    # Ricostruiamo il modello:
    # backbone RoBERTa + 9 teste lineari.
    model = NarrativeRoBERTa(
        "roberta-base",
        len(DIMENSIONS)
    )

    # File contenente i pesi addestrati di NarraBERT.
    state_path = local_dir / "model.pt"

    if not state_path.exists():
        raise FileNotFoundError(
            f"model.pt non trovato: "
            f"{state_path}"
        )

    # Carica i pesi sulla CPU.
    #
    # weights_only=True evita di caricare
    # oggetti Python non necessari.
    state = torch.load(
        state_path,
        map_location="cpu",
        weights_only=True,
    )

    # Inserisce i pesi nel modello appena ricostruito.
    model.load_state_dict(state)

    # Sposta il modello sulla GPU.
    model.to(device)

    # Modalità inferenza.
    model.eval()

    return tokenizer, model


# ============================================================
# PULIZIA DATI
# ============================================================

def clean_dataset(
    path: Path,
    label: str
) -> pd.DataFrame:
    """
    Carica e pulisce un dataset FAKE o REAL.

    Operazioni principali:
    - lettura CSV;
    - controllo colonne;
    - rimozione null;
    - rimozione testi vuoti;
    - filtro articoli < 100 parole;
    - eliminazione duplicati;
    - aggiunta label.
    """

    print(
        f"Caricamento {label}: "
        f"{path.name}"
    )

    df = pd.read_csv(path)

    # Per questa parte bastano titolo e testo.
    required = {
        "title",
        "text"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name}: "
            f"colonne mancanti: "
            f"{sorted(missing)}"
        )

    print(
        f"  iniziali: "
        f"{len(df):,}"
    )

    df = df.copy()

    # Normalizzazione di titolo e testo.
    df["title"] = (
        df["title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Elimina articoli con testo completamente vuoto.
    df = df[
        df["text"] != ""
    ].copy()

    # Conta il numero di parole.
    df["word_count"] = (
        df["text"]
        .str.split()
        .str.len()
        .fillna(0)
        .astype(int)
    )

    # Elimina articoli troppo corti.
    df = df[
        df["word_count"] >= MIN_WORDS
    ].copy()

    # Elimina duplicati esatti basati su titolo + testo.
    df = df.drop_duplicates(
        subset=[
            "title",
            "text"
        ]
    ).copy()

    # Aggiunge la classe FAKE / REAL.
    df["label"] = label

    print(
        f"  dopo cleaning: "
        f"{len(df):,}"
    )

    return df.reset_index(drop=True)


def load_dataset() -> pd.DataFrame:
    """
    Carica entrambi i dataset e li unisce
    in un unico DataFrame.
    """

    fake = clean_dataset(
        FAKE_CSV,
        "FAKE"
    )

    real = clean_dataset(
        REAL_CSV,
        "REAL"
    )

    # Unisce le due classi.
    df = pd.concat(
        [fake, real],
        ignore_index=True
    )

    # Aggiunge un identificatore unico.
    df.insert(
        0,
        "article_id",
        range(1, len(df) + 1)
    )

    return df


# ============================================================
# CHUNKING DEGLI ARTICOLI
# ============================================================

def article_chunks(
    title: str,
    text: str,
    tokenizer,
):
    """
    Divide titolo + testo in chunk da massimo 254 token.

    I chunk sono sovrapposti grazie allo stride di 192 token.
    """

    # NarraBERT riceve titolo + corpo dell'articolo.
    full_text = (
        f"{title}\n\n{text}"
        .strip()
    )

    # Tokenizza l'intero articolo.
    #
    # Non aggiungiamo ancora i token speciali,
    # perché verranno aggiunti in infer_token_batch().
    token_ids = tokenizer.encode(
        full_text,
        add_special_tokens=False,
    )

    # Caso limite: testo vuoto.
    if not token_ids:
        return [[]]

    # Se tutto l'articolo entra nel limite,
    # restituiamo un singolo chunk.
    if len(token_ids) <= CONTENT_TOKENS:
        return [token_ids]

    chunks = []

    # Divide l'articolo in finestre sovrapposte.
    #
    # Esempio:
    #
    # chunk 1: 0 -> 253
    # chunk 2: 192 -> 445
    # chunk 3: 384 -> ...
    for start in range(
        0,
        len(token_ids),
        STRIDE
    ):

        chunk = token_ids[
            start:
            start + CONTENT_TOKENS
        ]

        if not chunk:
            break

        chunks.append(chunk)

        # Se siamo arrivati alla fine dell'articolo,
        # interrompiamo il ciclo.
        if (
            start + CONTENT_TOKENS
            >= len(token_ids)
        ):
            break

    return chunks


# ============================================================
# INFERENZA
# ============================================================

@torch.inference_mode()
def infer_token_batch(
    token_batches: list[list[int]],
    tokenizer,
    model,
    device: torch.device,
) -> np.ndarray:
    """
    Esegue NarraBERT su un batch di chunk.

    Restituisce una matrice:

        numero_chunk × 9 dimensioni
    """

    features = []

    # Ogni lista di token viene trasformata
    # nella forma attesa dal modello.
    for ids in token_batches:

        features.append(
            tokenizer.prepare_for_model(
                ids,

                # Aggiunge token speciali RoBERTa.
                add_special_tokens=True,

                max_length=MAX_LENGTH,

                truncation=True,
            )
        )

    # Padding dinamico:
    # tutti i chunk del batch vengono portati
    # alla stessa lunghezza.
    encoded = tokenizer.pad(
        features,
        padding=True,
        return_tensors="pt",
    )

    # Sposta input_ids e attention_mask sulla GPU.
    input_ids = (
        encoded["input_ids"]
        .to(device)
    )

    attention_mask = (
        encoded["attention_mask"]
        .to(device)
    )

    # Mixed precision:
    # usa float16 sulla GPU per ridurre memoria
    # e velocizzare l'inferenza.
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):

        scores = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # Torna in float32 e sposta i risultati sulla CPU.
    return (
        scores
        .float()
        .cpu()
        .numpy()
    )


# ============================================================
# ACCUMULATORI
# ============================================================

def initialize_accumulators(
    n_articles: int
):
    """
    Prepara le strutture usate per raccogliere
    i risultati dei chunk di ogni articolo.

    Per ogni articolo memorizziamo:

    - first: punteggio del primo chunk
    - sum: somma dei chunk
    - max: massimo
    - count: numero di chunk
    """

    n_dims = len(DIMENSIONS)

    return {

        # Matrice:
        # articoli × 9 dimensioni.
        #
        # Contiene il risultato del primo chunk.
        "first": np.full(
            (n_articles, n_dims),
            np.nan,
            dtype=np.float32,
        ),

        # Somma progressiva dei valori.
        "sum": np.zeros(
            (n_articles, n_dims),
            dtype=np.float64,
        ),

        # Massimo osservato per ogni dimensione.
        "max": np.full(
            (n_articles, n_dims),
            -np.inf,
            dtype=np.float32,
        ),

        # Numero di chunk processati per articolo.
        "count": np.zeros(
            n_articles,
            dtype=np.int32,
        ),
    }


def update_accumulators(
    acc,
    article_indices,
    chunk_positions,
    scores,
):
    """
    Aggiorna gli accumulatori dopo l'inferenza
    di un batch di chunk.
    """

    # Ogni riga di scores corrisponde a un chunk.
    for row_idx, article_idx in enumerate(
        article_indices
    ):

        score = scores[row_idx]

        # Se questo è il primo chunk dell'articolo,
        # salviamo anche la misura FIRST.
        if chunk_positions[row_idx] == 0:
            acc["first"][
                article_idx
            ] = score

        # Accumula la somma. (serve per fare la media)
        acc["sum"][
            article_idx
        ] += score

        # Aggiorna il massimo per ogni dimensione.
        acc["max"][
            article_idx
        ] = np.maximum(
            acc["max"][article_idx],
            score,
        )

        # Incrementa il numero di chunk.
        acc["count"][
            article_idx
        ] += 1


# ============================================================
# COSTRUZIONE OUTPUT
# ============================================================

def build_output(
    df: pd.DataFrame,
    acc,
) -> pd.DataFrame:
    """
    Converte gli accumulatori in colonne finali
    da aggiungere al dataset.
    """

    result = df.copy()

    # Numero di chunk per articolo.
    counts = (
        acc["count"]
        .astype(np.float64)
    )

    # Evita divisione per zero.
    safe_counts = np.where(
        counts == 0,
        np.nan,
        counts
    )

    # Calcolo della media:
    #
    # somma / numero chunk
    means = (
        acc["sum"]
        / safe_counts[:, None]
    )

    # Copia dei massimi.
    max_scores = acc["max"].copy()

    # Eventuali -inf residui diventano NaN.
    max_scores[
        ~np.isfinite(max_scores)
    ] = np.nan

    # Numero totale di chunk per articolo.
    result[
        "narrabert_chunk_count"
    ] = acc["count"]

    # Per ogni dimensione creiamo tre colonne.
    for dim_idx, dim in enumerate(
        DIMENSIONS
    ):

        result[
            f"narrabert_{dim}_first"
        ] = acc["first"][:, dim_idx]

        result[
            f"narrabert_{dim}_mean"
        ] = means[:, dim_idx]

        result[
            f"narrabert_{dim}_max"
        ] = max_scores[:, dim_idx]

    return result


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    df: pd.DataFrame,
    acc,
    processed_articles: int,
):
    """
    Salva un file temporaneo ogni CHECKPOINT_EVERY articoli.

    Serve per non perdere completamente i risultati
    se l'esecuzione viene interrotta.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Costruisce un output solo per gli articoli
    # già processati.
    partial = build_output(
        df.iloc[
            :processed_articles
        ].copy(),

        {
            "first":
                acc["first"][
                    :processed_articles
                ],

            "sum":
                acc["sum"][
                    :processed_articles
                ],

            "max":
                acc["max"][
                    :processed_articles
                ],

            "count":
                acc["count"][
                    :processed_articles
                ],
        },
    )

    checkpoint = (
        OUTPUT_DIR
        / "dataset_with_narrativity_narrabert_checkpoint.csv"
    )

    partial.to_csv(
        checkpoint,
        index=False,
        encoding="utf-8",
    )

    print(
        f"\nCheckpoint: "
        f"{processed_articles:,} articoli "
        f"-> {checkpoint.name}"
    )



# ============================================================
# GRAFICO RISULTATI NARRABERT
# ============================================================

def generate_narrabert_results_plot(result: pd.DataFrame):
    """Genera un confronto FAKE vs REAL delle 9 dimensioni NarraBERT (MEAN)."""
    import matplotlib.pyplot as plt
    import numpy as np

    mean_cols = [f"narrabert_{dim}_mean" for dim in DIMENSIONS]
    means = result.groupby("label")[mean_cols].mean()
    y = np.arange(len(DIMENSIONS))
    height = 0.36

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(y - height / 2, means.loc["FAKE", mean_cols], height=height, label="FAKE")
    ax.barh(y + height / 2, means.loc["REAL", mean_cols], height=height, label="REAL")
    ax.set_yticks(y)
    ax.set_yticklabels(DIMENSIONS)
    ax.set_xlabel("Mean NarraBERT score")
    ax.set_ylabel("Narrative dimension")
    ax.set_title("NarraBERT — Narrative dimensions: FAKE vs REAL")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    output_dir = OUTPUT_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "narrabert_dimensions_fake_real.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico NarraBERT salvato: {output_path}")

# ============================================================
# MAIN
# ============================================================

def main():
    """
    Pipeline completa NarraBERT.

    1. controlla CUDA;
    2. carica e pulisce i dataset;
    3. carica NarraBERT;
    4. divide gli articoli in chunk;
    5. esegue inferenza batch-wise;
    6. aggrega i risultati;
    7. salva il dataset completo.
    """

    # Il progetto richiede GPU.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA non disponibile. "
            "Questo script richiede una GPU NVIDIA."
        )

    device = torch.device("cuda:0")

    print("=" * 72)
    print("NARRABERT - FULL DATASET")
    print("=" * 72)

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: "
        f"{torch.version.cuda}"
    )

    print(
        f"Modello: "
        f"{MODEL_ID}"
    )

    print(
        f"Max length: "
        f"{MAX_LENGTH}"
    )

    print(
        f"Content tokens/chunk: "
        f"{CONTENT_TOKENS}"
    )

    print(
        f"Stride: "
        f"{STRIDE}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print()

    # Carica l'intero dataset.
    df = load_dataset()

    print()

    print(
        f"Totale articoli: "
        f"{len(df):,}"
    )

    print(
        df["label"]
        .value_counts()
        .to_string()
    )

    print()

    # Carica tokenizer e modello.
    tokenizer, model = load_model(
        device
    )

    # Prepara gli accumulatori.
    acc = initialize_accumulators(
        len(df)
    )

    # Questi buffer raccolgono i chunk
    # prima dell'inferenza batch-wise.
    pending_tokens = []
    pending_articles = []
    pending_positions = []

    total_chunks = 0
    last_checkpoint = 0

    def flush():
        """
        Esegue l'inferenza sui chunk attualmente
        presenti nel buffer e aggiorna gli accumulatori.
        """

        nonlocal pending_tokens
        nonlocal pending_articles
        nonlocal pending_positions
        nonlocal total_chunks

        # Se il buffer è vuoto non c'è nulla da fare.
        if not pending_tokens:
            return

        # Inferenza sui chunk del buffer.
        scores = infer_token_batch(
            pending_tokens,
            tokenizer,
            model,
            device,
        )

        # Riporta i risultati ai rispettivi articoli.
        update_accumulators(
            acc,
            pending_articles,
            pending_positions,
            scores,
        )

        total_chunks += len(
            pending_tokens
        )

        # Svuota i buffer.
        pending_tokens = []
        pending_articles = []
        pending_positions = []

    # Barra di avanzamento a livello di articolo.
    progress = tqdm(
        total=len(df),
        desc="Articoli",
        unit="art",
    )

    # Scorre tutto il dataset.
    for article_idx, row in df.iterrows():

        # Divide l'articolo in chunk.
        chunks = article_chunks(
            row["title"],
            row["text"],
            tokenizer,
        )

        # Inserisce ogni chunk nel buffer.
        for chunk_position, token_ids in enumerate(
            chunks
        ):

            pending_tokens.append(
                token_ids
            )

            # Salva l'indice dell'articolo.
            pending_articles.append(
                article_idx
            )

            # Salva la posizione del chunk:
            # 0 = primo chunk.
            pending_positions.append(
                chunk_position
            )

            # Quando il buffer raggiunge BATCH_SIZE
            # viene eseguita l'inferenza.
            if (
                len(pending_tokens)
                >= BATCH_SIZE
            ):
                flush()

        progress.update(1)

        processed = article_idx + 1

        # Checkpoint periodico.
        if (
            processed - last_checkpoint
            >= CHECKPOINT_EVERY
            and processed < len(df)
        ):

            # Prima salviamo eventuali chunk rimasti
            # nel buffer.
            flush()

            save_checkpoint(
                df,
                acc,
                processed
            )

            last_checkpoint = processed

    # Processa eventuali chunk rimasti
    # dopo l'ultimo articolo.
    flush()

    progress.close()

    # Converte gli accumulatori
    # nel dataset finale.
    result = build_output(
        df,
        acc
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Salva il dataset completo.
    result.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
    )

    # Il checkpoint non serve più
    # dopo un'esecuzione completata correttamente.
    checkpoint = (
        OUTPUT_DIR
        / "dataset_with_narrativity_narrabert_checkpoint.csv"
    )

    if checkpoint.exists():
        checkpoint.unlink()

    print()

    print("=" * 72)
    print("COMPLETATO")
    print("=" * 72)

    print(
        f"Articoli: "
        f"{len(result):,}"
    )

    print(
        f"Chunk analizzati: "
        f"{total_chunks:,}"
    )

    print(
        f"Chunk medi per articolo: "
        f"{result['narrabert_chunk_count'].mean():.2f}"
    )

    # Per controllo stampiamo la media
    # delle 9 dimensioni separatamente per classe.
    print(
        "\nMedia delle dimensioni "
        "(aggregazione MEAN):"
    )

    mean_cols = [
        f"narrabert_{dim}_mean"
        for dim in DIMENSIONS
    ]

    summary = (
        result
        .groupby("label")[mean_cols]
        .mean()
        .T
        .round(4)
    )

    print(summary)

    # Generazione immagine delle 9 dimensioni narrative.
    generate_narrabert_results_plot(result)

    print(
        f"\nOutput salvato in:\n"
        f"{OUTPUT_CSV}"
    )

    print(
        "\nIMPORTANTE: il file contiene FIRST, MEAN e MAX "
        "per ogni dimensione. "
        "Non viene creato un singolo score composito "
        "di narratività."
    )


if __name__ == "__main__":
    main()