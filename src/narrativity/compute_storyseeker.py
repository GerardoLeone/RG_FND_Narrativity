"""
Modulo: compute_storyseeker.py

Scopo:
calcolare la narratività degli articoli FAKE e REAL usando
il modello `mariaantoniak/storyseeker`.

Il modello restituisce una probabilità di appartenenza alla classe narrativa.
Nel progetto assumiamo:

    LABEL_1 = narrativa

Poiché gli articoli possono essere più lunghi del limite di RoBERTa,
ogni articolo viene diviso in chunk sovrapposti.

Per ogni articolo vengono salvate diverse misure:
- narrativity_first_chunk
- narrativity_mean
- narrativity_max
- narrativity_min
- narrativity_range
"""

from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Modello Hugging Face utilizzato per classificare il testo
# come narrativo / non narrativo.
MODEL_NAME = "mariaantoniak/storyseeker"

# Dai test preliminari abbiamo verificato che LABEL_1
# corrisponde alla classe narrativa.
NARRATIVE_LABEL = "LABEL_1"

# Dataset di input.
FAKE_CSV = Path("../../data/raw/Fake.csv")
TRUE_CSV = Path("../../data/raw/True.csv")

# Dataset di output contenente i punteggi StorySeeker.
OUTPUT_CSV = Path(
    "../../data/interim/dataset_with_storyseeker.csv"
)

# Scartiamo articoli troppo corti.
MIN_WORDS = 100

# RoBERTa accetta al massimo 512 token.
MAX_TOKENS = 512

# Ogni nuovo chunk parte 384 token dopo il precedente.
# Siccome il chunk contiene circa 510 token,
# otteniamo una sovrapposizione tra chunk consecutivi.
CHUNK_STRIDE = 384

# Numero di chunk processati contemporaneamente sulla GPU.
BATCH_SIZE = 32


# ============================================================
# CARICAMENTO MODELLO
# ============================================================

def load_model():
    """
    Carica StorySeeker e il relativo tokenizer.

    Il progetto richiede esplicitamente CUDA:
    se la GPU non è disponibile lo script si interrompe.

    Returns
    -------
    classifier
        Pipeline Hugging Face per la classificazione del testo.

    tokenizer
        Tokenizer associato al modello StorySeeker.
    """

    # Controlla che CUDA sia effettivamente disponibile.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile.")

    print(
        f"GPU rilevata: "
        f"{torch.cuda.get_device_name(0)}"
    )

    # Pipeline Hugging Face per classificazione testuale.
    #
    # device=0 significa:
    # usa la prima GPU CUDA disponibile.
    classifier = pipeline(
        task="text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=0
    )

    # Il tokenizer viene caricato anche separatamente perché
    # ci serve per dividere manualmente gli articoli in chunk.
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("StorySeeker caricato.")
    print(
        "Label:",
        classifier.model.config.id2label
    )

    return classifier, tokenizer


# ============================================================
# CARICAMENTO E PULIZIA DATASET
# ============================================================

def load_dataset(
    path: Path,
    label: str
) -> pd.DataFrame:
    """
    Carica e pulisce uno dei due dataset.

    Operazioni:
    - verifica esistenza del file;
    - verifica colonne richieste;
    - sostituisce valori nulli;
    - rimuove spazi iniziali/finali;
    - assegna FAKE o REAL;
    - calcola il numero di parole;
    - rimuove testi vuoti;
    - rimuove articoli sotto MIN_WORDS;
    - rimuove duplicati.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"File non trovato: {path.resolve()}"
        )

    df = pd.read_csv(path)

    # Colonne indispensabili per questa pipeline.
    required = {
        "title",
        "text",
        "subject",
        "date"
    }

    if not required.issubset(df.columns):
        missing = required - set(df.columns)

        raise ValueError(
            f"Colonne mancanti in {path}: "
            f"{sorted(missing)}"
        )

    # Lavoriamo su una copia del DataFrame.
    df = df.copy()

    # Normalizzazione delle colonne testuali.
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

    df["subject"] = (
        df["subject"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["date"] = (
        df["date"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Inserisce l'etichetta della classe.
    df["label"] = label

    # Lunghezza dell'articolo espressa in parole.
    df["word_count"] = (
        df["text"]
        .str.split()
        .str.len()
    )

    initial = len(df)

    # Rimuove testi completamente vuoti.
    df = df[df["text"] != ""]

    # Rimuove articoli sotto la soglia minima.
    df = df[
        df["word_count"] >= MIN_WORDS
    ]

    # Rimuove duplicati esatti considerando
    # contemporaneamente titolo e testo.
    df = (
        df
        .drop_duplicates(
            subset=["title", "text"]
        )
        .reset_index(drop=True)
    )

    print(
        f"{label}: {initial} iniziali, "
        f"{len(df)} dopo pulizia."
    )

    return df


# ============================================================
# CHUNKING
# ============================================================

def split_into_chunks(
    tokenizer,
    title: str,
    text: str
) -> list[str]:
    """
    Divide un articolo in chunk compatibili con StorySeeker.

    StorySeeker usa RoBERTa e quindi non può ricevere
    arbitrariamente testi lunghi.

    Il titolo viene concatenato al testo dell'articolo.

    I chunk sono sovrapposti per ridurre la perdita
    di informazioni ai confini.
    """

    # Costruiamo il testo completo da analizzare.
    full_text = f"{title}. {text}".strip()

    # Tokenizziamo l'intero articolo senza troncarlo.
    #
    # add_special_tokens=False perché i token speciali
    # verranno aggiunti successivamente dalla pipeline.
    token_ids = tokenizer.encode(
        full_text,
        add_special_tokens=False,
        truncation=False,
        verbose=False
    )

    # RoBERTa usa due token speciali,
    # quindi lasciamo spazio:
    #
    # 512 - 2 = 510 token di contenuto.
    content_length = MAX_TOKENS - 2

    chunks = []

    # Scorre il testo con stride 384.
    #
    # Esempio:
    #
    # chunk 1: token 0-509
    # chunk 2: token 384-893
    #
    # quindi una parte viene vista in entrambi i chunk. Questo perchè non vogliamo perdere informazione vicino ai confini tra un chunk e l'altro.
    # Se una parte narrativa importante cade proprio a cavallo tra due chunk, senza overlap verrebbe spezzata e il modello vedrebbe metà contesto da una parte e metà dall'altra.
    for start in range(
        0,
        len(token_ids),
        CHUNK_STRIDE
    ):

        chunk_ids = token_ids[
            start:start + content_length
        ]

        if not chunk_ids:
            break

        # Torniamo temporaneamente dal formato token
        # al formato testo, perché la pipeline Hugging Face
        # riceve stringhe.
        chunks.append(
            tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
        )

        # Se questo chunk ha già raggiunto la fine
        # dell'articolo non ne servono altri.
        if (
            start + content_length
            >= len(token_ids)
        ):
            break

    return chunks


# ============================================================
# ESTRAZIONE PROBABILITÀ NARRATIVA
# ============================================================

def narrative_probability(
    prediction: list[dict]
) -> float:
    """
    Estrae la probabilità della classe narrativa.

    La pipeline restituisce qualcosa del tipo:

    [
        {"label": "LABEL_0", "score": ...},
        {"label": "LABEL_1", "score": ...}
    ]

    Noi cerchiamo LABEL_1.
    """

    for item in prediction:

        if item["label"] == NARRATIVE_LABEL:
            return float(item["score"])

    # Se LABEL_1 non compare significa che l'output
    # del modello non è quello atteso.
    raise RuntimeError(
        f"Label narrativa non trovata: "
        f"{prediction}"
    )



# ============================================================
# GRAFICO RISULTATI STORYSEEKER
# ============================================================

def generate_storyseeker_results_plot(result: pd.DataFrame):
    """Genera un confronto FAKE vs REAL delle principali misure StorySeeker."""
    import matplotlib.pyplot as plt
    import numpy as np

    measures = [
        "narrativity_first_chunk",
        "narrativity_mean",
        "narrativity_max",
    ]
    labels = ["First chunk", "Mean", "Max"]

    means = result.groupby("label")[measures].mean()
    y = np.arange(len(measures))
    height = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y - height / 2, means.loc["FAKE", measures], height=height, label="FAKE")
    ax.barh(y + height / 2, means.loc["REAL", measures], height=height, label="REAL")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean StorySeeker narrativity score")
    ax.set_ylabel("Narrativity measure")
    ax.set_title("StorySeeker — Narrativity: FAKE vs REAL")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    output_dir = OUTPUT_CSV.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "storyseeker_fake_real.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico StorySeeker salvato: {output_path.resolve()}")

# ============================================================
# MAIN
# ============================================================

def main():
    """
    Pipeline completa StorySeeker.

    1. carica FAKE e REAL;
    2. pulisce i dataset;
    3. assegna article_id;
    4. divide gli articoli in chunk;
    5. esegue StorySeeker su tutti i chunk;
    6. aggrega i punteggi per articolo;
    7. salva il nuovo dataset.
    """

    # Seed per rendere più riproducibili
    # eventuali operazioni PyTorch.
    torch.manual_seed(42)

    # Carica e pulisce i due dataset.
    fake = load_dataset(
        FAKE_CSV,
        "FAKE"
    )

    real = load_dataset(
        TRUE_CSV,
        "REAL"
    )

    # Unico dataset contenente entrambe le classi.
    dataset = pd.concat(
        [fake, real],
        ignore_index=True
    )

    # Identificatore univoco per ogni articolo.
    dataset.insert(
        0,
        "article_id",
        range(1, len(dataset) + 1)
    )

    # Caricamento del modello.
    classifier, tokenizer = load_model()

    # Conterrà tutti i chunk di tutti gli articoli.
    all_chunks = []

    # Tiene memoria dell'articolo da cui proviene
    # ciascun chunk.
    chunk_article_ids = []

    print("Creazione dei chunk...")

    # Prima creiamo tutti i chunk.
    for index, row in dataset.iterrows():

        chunks = split_into_chunks(
            tokenizer,
            row["title"],
            row["text"]
        )

        # Aggiunge i chunk dell'articolo
        # alla lista globale.
        all_chunks.extend(chunks)

        # Per ogni chunk salviamo l'article_id
        # corrispondente.
        chunk_article_ids.extend(
            [row["article_id"]] * len(chunks)
        )

        # Log ogni 1000 articoli.
        if (index + 1) % 1000 == 0:

            print(
                f"Preparati "
                f"{index + 1}/{len(dataset)} articoli; "
                f"chunk totali: {len(all_chunks)}"
            )

    print(
        f"Inferenza su "
        f"{len(all_chunks)} chunk..."
    )

    # Inferenza vera e propria.
    # Tutti i chunk vengono inviati a StorySeeker
    #
    # top_k=None chiede al modello di restituire
    # tutti i label, non solo quello più probabile.
    predictions = classifier(
        all_chunks,
        truncation=True,
        max_length=MAX_TOKENS,
        top_k=None,
        batch_size=BATCH_SIZE
    )

    # Dizionario:
    #
    # article_id -> lista dei punteggi dei suoi chunk
    scores_by_article = defaultdict(list)

    # Abbina ogni predizione al relativo articolo.
    for article_id, prediction in zip(
        chunk_article_ids,
        predictions
    ):

        scores_by_article[
            article_id
        ].append(
            narrative_probability(prediction)
        )

    rows = []

    # Aggregazione finale a livello di articolo.
    for article_id in dataset["article_id"]:

        scores = scores_by_article[
            article_id
        ]

        rows.append({

            "article_id": article_id,

            # Numero di chunk dell'articolo.
            "chunk_count": len(scores),

            # Punteggio del primo chunk:
            # è la misura primaria usata nella RQ1.
            "narrativity_first_chunk":
                scores[0],

            # Media della narratività di tutti i chunk.
            "narrativity_mean":
                sum(scores) / len(scores),

            # Parte più narrativa dell'articolo.
            "narrativity_max":
                max(scores),

            # Parte meno narrativa dell'articolo.
            "narrativity_min":
                min(scores),

            # Quanto varia la narratività
            # all'interno dell'articolo.
            "narrativity_range":
                max(scores) - min(scores),

            # Conserviamo anche tutti i punteggi
            # per eventuali analisi successive.
            "chunk_scores":
                "|".join(
                    f"{score:.4f}"
                    for score in scores
                )
        })

    # DataFrame contenente una riga per articolo
    # con tutte le misure StorySeeker.
    score_df = pd.DataFrame(rows)

    # Aggiungiamo i risultati al dataset originale.
    result = dataset.merge(
        score_df,
        on="article_id",
        how="left"
    )

    # Crea la directory di output se non esiste.
    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Salva il dataset finale.
    result.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nCompletato: "
        f"{OUTPUT_CSV.resolve()}"
    )

    # Statistiche descrittive separate
    # per FAKE e REAL.
    print("\nStatistiche per classe:")

    print(
        result.groupby("label")[
            [
                "narrativity_first_chunk",
                "narrativity_mean",
                "narrativity_max"
            ]
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std"
            ]
        )
        .round(4)
    )


    # Generazione immagine dei risultati StorySeeker.
    generate_storyseeker_results_plot(result)


# Avvia main() solo quando questo file viene
# eseguito direttamente.
if __name__ == "__main__":
    main()