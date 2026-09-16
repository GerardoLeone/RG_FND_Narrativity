"""
Modulo: analyze_emotion_arcs.py

Scopo:
rispondere alla RQ2 confrontando gli archi di intensità emotiva
degli articoli narrativi FAKE e REAL.
Come cambia la forza emotiva dall'inizio alla fine di un articolo narrativo, e cambia in modo diverso tra FAKE e REAL?

Pipeline:

1. seleziona gli articoli sufficientemente narrativi;
2. divide ogni articolo in 5 sezioni normalizzate;
3. applica GoEmotions a ogni sezione;
4. costruisce una misura di emotion strength per ogni sezione;
5. ricostruisce un arco emotivo composto da 5 punti;
6. confronta gli archi FAKE e REAL;
7. verifica sia:
   - differenze assolute di intensità;
   - differenze nella forma dell'arco.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.stats import mannwhitneyu

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Modello GoEmotions usato per stimare
# le emozioni nelle diverse sezioni dell'articolo.
MODEL_ID = "SamLowe/roberta-base-go_emotions"


# Root del progetto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Dataset che contiene:
# - testo
# - label FAKE/REAL
# - punteggi StorySeeker
# - probabilità emotive aggregate
INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_narrativity_emotions.csv"
)


# Directory in cui verranno salvati
# tutti i risultati della RQ2.
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rq2"
)


# Dataset long:
# una riga per articolo × sezione.
OUTPUT_LONG = (
    OUTPUT_DIR
    / "rq2_emotion_arcs.csv"
)


# Statistiche riassuntive per:
# FAKE/REAL × sezione.
OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "rq2_arc_summary.csv"
)


# Confronti FAKE vs REAL
# effettuati separatamente nelle 5 sezioni.
OUTPUT_SECTION_TESTS = (
    OUTPUT_DIR
    / "rq2_section_differences.csv"
)


# Test globale che confronta
# l'intero arco FAKE con l'intero arco REAL.
OUTPUT_GLOBAL_TEST = (
    OUTPUT_DIR
    / "rq2_global_arc_test.csv"
)


# ============================================================
# SELEZIONE DEGLI ARTICOLI NARRATIVI
# ============================================================

# Misura StorySeeker utilizzata
# per decidere se un articolo è narrativo.
NARRATIVITY_COLUMN = (
    "narrativity_first_chunk"
)


# Consideriamo narrativi solo gli articoli
# con score StorySeeker >= 0.50.
NARRATIVITY_THRESHOLD = 0.50


# ============================================================
# PARAMETRI DELL'ARCO
# ============================================================

# Ogni articolo viene normalizzato
# in 5 sezioni.
N_SECTIONS = 5


# Limite massimo di RoBERTa.
MAX_MODEL_TOKENS = 512


# Lasciamo spazio ai token speciali.
CONTENT_TOKENS = 510


# Se una singola sezione è ancora troppo lunga,
# viene spezzata in chunk sovrapposti.
CHUNK_STRIDE = 384


# Numero di chunk processati
# contemporaneamente sulla GPU.
BATCH_SIZE = 64


# ============================================================
# TEST PERMUTAZIONALE
# ============================================================

# Numero di permutazioni usate
# nel confronto globale degli archi.
N_PERMUTATIONS = 5000


# Seed per rendere il test riproducibile.
RANDOM_SEED = 42


# ============================================================
# EMOZIONI GOEMOTIONS
# ============================================================

# Le 27 emozioni non neutrali.
#
# Neutral viene esclusa perché
# vogliamo misurare intensità emotiva.
EMOTION_LABELS = [
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
# NORMALIZZAZIONE LABEL
# ============================================================

def normalize_label(
    series: pd.Series
) -> pd.Series:
    """
    Uniforma tutte le label nel formato:

        FAKE
        REAL

    Gestisce anche dataset con:
        1 / 0
        TRUE / FALSE
    """

    # Prova a interpretare la label
    # anche come numero.
    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    # Versione testuale normalizzata.
    out = (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # 1 -> FAKE
    out.loc[
        numeric == 1
    ] = "FAKE"

    # 0 -> REAL
    out.loc[
        numeric == 0
    ] = "REAL"

    return out.replace({
        "TRUE": "REAL",
        "FALSE": "FAKE",
        "1.0": "FAKE",
        "0.0": "REAL",
    })


# ============================================================
# DIVISIONE DELL'ARTICOLO IN 5 SEZIONI
# ============================================================

def split_into_five_sections(
    token_ids: list[int]
) -> list[list[int]]:
    """
    Divide l'articolo in 5 sezioni
    di uguale lunghezza normalizzata.

    Non importa quindi se un articolo ha:
        500 token
        1000 token
        2000 token

    Ogni articolo viene sempre rappresentato
    da 5 posizioni:

        1 = inizio
        2
        3 = centro
        4
        5 = fine
    """

    # Caso limite:
    # articolo senza token.
    if not token_ids:
        return [
            []
            for _ in range(N_SECTIONS)
        ]

    # np.linspace costruisce 6 punti:
    #
    # inizio + 4 confini interni + fine
    #
    # che delimitano le 5 sezioni.
    boundaries = np.linspace(
        0,
        len(token_ids),
        N_SECTIONS + 1,
        dtype=int,
    )

    sections = []

    for i in range(N_SECTIONS):

        start = boundaries[i]
        end = boundaries[i + 1]

        section = token_ids[
            start:end
        ]

        # Protezione per testi estremamente corti:
        # garantisce almeno un token per sezione.
        if (
            not section
            and token_ids
        ):

            pos = min(
                start,
                len(token_ids) - 1
            )

            section = token_ids[
                pos:pos + 1
            ]

        sections.append(
            section
        )

    return sections


# ============================================================
# CHUNKING INTERNO ALLE SEZIONI
# ============================================================

def section_chunks(
    section_ids: list[int]
) -> list[list[int]]:
    """
    Se una delle 5 sezioni supera
    il limite di 510 token di contenuto,
    la divide ulteriormente in chunk sovrapposti.

    Questo evita di perdere parti di sezioni molto lunghe.
    """

    # Se la sezione entra nel modello,
    # non serve dividerla.
    if len(section_ids) <= CONTENT_TOKENS:
        return [
            section_ids
        ]

    chunks = []

    for start in range(
        0,
        len(section_ids),
        CHUNK_STRIDE
    ):

        chunk = section_ids[
            start:
            start + CONTENT_TOKENS
        ]

        if not chunk:
            break

        chunks.append(
            chunk
        )

        # Se il chunk ha già raggiunto
        # la fine della sezione, ci fermiamo.
        if (
            start + CONTENT_TOKENS
            >= len(section_ids)
        ):
            break

    return chunks


# ============================================================
# COSTRUZIONE DEGLI ELEMENTI DA ANALIZZARE
# ============================================================

def build_work_items(
    df: pd.DataFrame,
    tokenizer
):
    """
    Prepara tutti i chunk da mandare a GoEmotions.

    Restituisce tre liste parallele:

    token_chunks
        contenuto dei chunk;

    article_positions
        articolo di provenienza;

    section_positions
        sezione di provenienza, da 1 a 5.

    Questo permette dopo l'inferenza di ricostruire
    correttamente l'arco di ciascun articolo.
    """

    token_chunks = []
    article_positions = []
    section_positions = []

    # Scorre tutti gli articoli selezionati.
    for article_pos, text in enumerate(
        tqdm(
            df["text"]
            .fillna("")
            .astype(str),
            desc="Preparazione articoli"
        )
    ):

        # Tokenizza l'intero articolo
        # senza troncarlo.
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        # Divide l'articolo in 5 sezioni.
        sections = split_into_five_sections(
            token_ids
        )

        # Analizza ciascuna sezione.
        for section_idx, section_ids in enumerate(
            sections,
            start=1
        ):

            # Se necessario divide ulteriormente
            # la sezione in chunk.
            chunks = section_chunks(
                section_ids
            )

            for chunk in chunks:

                token_chunks.append(
                    chunk
                )

                # Tiene traccia dell'articolo.
                article_positions.append(
                    article_pos
                )

                # Tiene traccia della sezione.
                section_positions.append(
                    section_idx
                )

    return (
        token_chunks,
        article_positions,
        section_positions
    )


# ============================================================
# INFERENZA GOEMOTIONS
# ============================================================

@torch.inference_mode()
def infer_chunks(
    token_chunks: list[list[int]],
    tokenizer,
    model,
    device: torch.device,
) -> np.ndarray:
    """
    Applica GoEmotions a tutti i chunk preparati.

    Restituisce una matrice:

        numero_chunk × numero_etichette

    con le probabilità emotive.
    """

    all_probs = []

    # Inferenza batch-wise.
    for start in tqdm(
        range(
            0,
            len(token_chunks),
            BATCH_SIZE
        ),
        desc="GoEmotions",
    ):

        batch_tokens = token_chunks[
            start:
            start + BATCH_SIZE
        ]

        features = []

        # Prepara ogni chunk nel formato
        # richiesto da RoBERTa.
        for ids in batch_tokens:

            features.append(
                tokenizer.prepare_for_model(
                    ids,
                    add_special_tokens=True,
                    max_length=MAX_MODEL_TOKENS,
                    truncation=True,
                )
            )

        # Padding dinamico del batch.
        encoded = tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        # Sposta tutto sulla GPU.
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        # Mixed precision per velocizzare l'inferenza.
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):

            logits = (
                model(**encoded)
                .logits
            )

        # GoEmotions è multilabel:
        # sigmoid trasforma ogni logit
        # in probabilità indipendente.
        probs = (
            torch.sigmoid(logits)
            .float()
            .cpu()
            .numpy()
        )

        all_probs.append(
            probs
        )

    # Unisce tutti i batch.
    return np.vstack(
        all_probs
    )


# ============================================================
# INDICI DELLE EMOZIONI NON NEUTRALI
# ============================================================

def get_non_neutral_indices(
    model
) -> list[int]:
    """
    Recupera gli indici delle 27 emozioni
    non neutrali dall'ordine interno del modello.

    È importante non assumere manualmente
    che l'ordine delle label sia sempre lo stesso.
    """

    # Mapping:
    #
    # indice -> nome label
    id2label = {
        int(k): str(v)
        for k, v
        in model.config.id2label.items()
    }

    # Mapping inverso:
    #
    # nome label -> indice
    label_to_idx = {
        label.lower(): idx
        for idx, label
        in id2label.items()
    }

    # Controlla che tutte le emozioni
    # attese siano presenti.
    missing = [
        label
        for label in EMOTION_LABELS
        if label not in label_to_idx
    ]

    if missing:
        raise ValueError(
            "Etichette GoEmotions "
            "mancanti nel modello: "
            + ", ".join(missing)
        )

    # Restituisce gli indici
    # nell'ordine EMOTION_LABELS.
    return [
        label_to_idx[label]
        for label in EMOTION_LABELS
    ]


# ============================================================
# AGGREGAZIONE A LIVELLO DI SEZIONE
# ============================================================

def aggregate_sections(
    selected: pd.DataFrame,
    probs: np.ndarray,
    article_positions: list[int],
    section_positions: list[int],
    non_neutral_indices: list[int],
) -> pd.DataFrame:
    """
    Riporta le predizioni dal livello di chunk
    al livello di:

        articolo × sezione

    Se una sezione aveva più chunk,
    le probabilità vengono mediate.

    Per ogni sezione vengono poi costruite
    tre misure di emotion strength:
    - max
    - top3 mean
    - RMS
    """

    # Dizionario:
    #
    # (article_pos, section_idx)
    #     ->
    # lista dei vettori GoEmotions
    accum = {}

    for i, (
        article_pos,
        section_idx
    ) in enumerate(
        zip(
            article_positions,
            section_positions
        )
    ):

        key = (
            article_pos,
            section_idx
        )

        accum.setdefault(
            key,
            []
        ).append(

            # Mantiene solo le 27 emozioni
            # non neutrali.
            probs[
                i,
                non_neutral_indices
            ]
        )

    rows = []

    # Ricostruiamo ogni articolo.
    for article_pos in range(
        len(selected)
    ):

        source_row = selected.iloc[
            article_pos
        ]

        # Ogni articolo deve avere
        # esattamente 5 sezioni.
        for section_idx in range(
            1,
            N_SECTIONS + 1
        ):

            vectors = accum[
                (
                    article_pos,
                    section_idx
                )
            ]

            # Se una sezione aveva più chunk,
            # calcoliamo il profilo emotivo medio.
            profile = np.mean(
                np.vstack(vectors),
                axis=0
            )

            # ------------------------------------------------
            # EMOTION STRENGTH MAX
            # ------------------------------------------------

            # Emozione singola più intensa.
            max_strength = float(
                np.max(profile)
            )

            # ------------------------------------------------
            # TOP 3 MEAN
            # ------------------------------------------------

            # Media delle tre emozioni
            # con probabilità maggiore.
            top3_strength = float(
                np.mean(
                    np.sort(profile)[-3:]
                )
            )

            # ------------------------------------------------
            # RMS
            # ------------------------------------------------

            # Misura che tiene conto
            # dell'intero profilo emotivo.
            rms_strength = float(
                np.sqrt(
                    np.mean(
                        np.square(profile)
                    )
                )
            )

            rows.append({

                # Identificatore articolo.
                "article_id":
                    (
                        source_row["article_id"]
                        if "article_id"
                        in selected.columns

                        else
                        article_pos + 1
                    ),

                # FAKE / REAL.
                "label":
                    source_row["label"],

                # Score StorySeeker
                # usato per selezionare
                # l'articolo come narrativo.
                "narrativity_score":
                    source_row[
                        NARRATIVITY_COLUMN
                    ],

                # Posizione normalizzata 1-5.
                "section":
                    section_idx,

                # Tre misure di intensità.
                "emotion_strength_max":
                    max_strength,

                "emotion_strength_top3_mean":
                    top3_strength,

                "emotion_strength_rms":
                    rms_strength,
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# CLIFF'S DELTA
# ============================================================

def cliffs_delta(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Calcola Cliff's delta.

    È una misura non parametrica
    della dimensione dell'effetto.

    Interpretazione del segno:

        > 0
        valori di x tendono a essere maggiori

        < 0
        valori di y tendono a essere maggiori

    Qui:
        x = FAKE
        y = REAL
    """

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    # Mann-Whitney U permette
    # di ricavare Cliff's delta.
    u, _ = mannwhitneyu(
        x,
        y,
        alternative="two-sided"
    )

    return float(
        (
            2.0 * u
        )
        /
        (
            len(x)
            * len(y)
        )
        - 1.0
    )


# ============================================================
# BENJAMINI-HOCHBERG
# ============================================================

def benjamini_hochberg(
    p_values: np.ndarray
) -> np.ndarray:
    """
    Corregge i p-value multipli
    controllando la False Discovery Rate.
    """

    p = np.asarray(
        p_values,
        dtype=float
    )

    # Ordine crescente.
    order = np.argsort(
        p
    )

    ranked = p[
        order
    ]

    m = len(
        ranked
    )

    # Correzione BH.
    adjusted = (
        ranked
        * m
        / np.arange(
            1,
            m + 1
        )
    )

    # Impone monotonicità.
    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0,
        1
    )

    # Ripristina ordine originale.
    out = np.empty_like(
        adjusted
    )

    out[
        order
    ] = adjusted

    return out


# ============================================================
# CONFRONTO FAKE / REAL PER SEZIONE
# ============================================================

def section_tests(
    long_df: pd.DataFrame,
    measure: str
) -> pd.DataFrame:
    """
    Confronta FAKE e REAL
    separatamente nelle 5 sezioni.

    Per ciascuna sezione calcola:
    - media FAKE;
    - media REAL;
    - differenza;
    - Mann-Whitney U;
    - Cliff's delta;
    - p-value corretto FDR.
    """

    rows = []

    for section in range(
        1,
        N_SECTIONS + 1
    ):

        # Valori FAKE della sezione.
        fake = long_df.loc[
            (
                long_df["label"]
                == "FAKE"
            )
            &
            (
                long_df["section"]
                == section
            ),
            measure,
        ].to_numpy()

        # Valori REAL della sezione.
        real = long_df.loc[
            (
                long_df["label"]
                == "REAL"
            )
            &
            (
                long_df["section"]
                == section
            ),
            measure,
        ].to_numpy()

        # Mann-Whitney:
        # test non parametrico
        # tra i due gruppi indipendenti.
        u, p = mannwhitneyu(
            fake,
            real,
            alternative="two-sided",
        )

        rows.append({

            "measure":
                measure,

            "section":
                section,

            "fake_n":
                len(fake),

            "real_n":
                len(real),

            "fake_mean":
                fake.mean(),

            "real_mean":
                real.mean(),

            # Positivo:
            # FAKE più intenso.
            "difference_fake_minus_real":
                (
                    fake.mean()
                    - real.mean()
                ),

            # Dimensione dell'effetto.
            "cliffs_delta_fake_vs_real":
                cliffs_delta(
                    fake,
                    real
                ),

            "p_value":
                p,
        })

    result = pd.DataFrame(
        rows
    )

    # Correzione dei 5 test.
    result[
        "fdr_q_value"
    ] = benjamini_hochberg(
        result[
            "p_value"
        ].to_numpy()
    )

    return result


# ============================================================
# TEST GLOBALE DELL'ARCO
# ============================================================

def permutation_arc_test(
    long_df: pd.DataFrame,
    measure: str,
    centered: bool,
    rng: np.random.Generator,
):
    """
    Confronta l'intero arco medio FAKE
    con l'intero arco medio REAL
    mediante permutation test.

    Sono possibili due versioni:

    centered = False
        confronta gli archi assoluti.

    centered = True
        sottrae prima la media
        di ogni articolo.

        In questo modo si confronta
        soprattutto la FORMA dell'arco,
        non il livello emotivo medio.
    """

    # Trasforma il dataset long:

    # article_id | section | strength

    # in:

    # article_id | sec1 | sec2 | sec3 | sec4 | sec5
    wide = long_df.pivot(
        index="article_id",
        columns="section",
        values=measure,
    )

    # Recupera FAKE/REAL per ogni articolo.
    labels = (
        long_df[
            [
                "article_id",
                "label"
            ]
        ]
        .drop_duplicates(
            "article_id"
        )
        .set_index(
            "article_id"
        )
        .loc[
            wide.index,
            "label"
        ]
        .to_numpy()
    )

    # Matrice:
    #
    # articoli × 5 sezioni.
    values = wide.to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # CENTERING
    # --------------------------------------------------------

    if centered:

        # Sottrae la media di ciascun articolo.
        #
        # Esempio:
        #
        # [0.20, 0.25, 0.30, 0.35, 0.40]
        #
        # media = 0.30
        #
        # centered:
        # [-.10, -.05, 0, .05, .10]
        #
        # Rimane la forma, non il livello assoluto.
        values = (
            values
            - values.mean(
                axis=1,
                keepdims=True
            )
        )

    # Maschere dei due gruppi.
    fake_mask = (
        labels == "FAKE"
    )

    real_mask = (
        labels == "REAL"
    )

    # Arco medio FAKE.
    fake_arc = (
        values[fake_mask]
        .mean(axis=0)
    )

    # Arco medio REAL.
    real_arc = (
        values[real_mask]
        .mean(axis=0)
    )

    # Distanza quadratica totale
    # tra i due archi.
    observed = float(
        np.sum(
            (
                fake_arc
                - real_arc
            ) ** 2
        )
    )

    # Conterrà le distanze ottenute
    # sotto l'ipotesi nulla.
    permuted_stats = np.empty(
        N_PERMUTATIONS
    )

    # ========================================================
    # PERMUTAZIONI
    # ========================================================

    for i in range(
        N_PERMUTATIONS
    ):

        # Mescola casualmente FAKE/REAL
        # mantenendo invariati gli archi.
        permuted_labels = rng.permutation(
            labels
        )

        fmask = (
            permuted_labels
            == "FAKE"
        )

        rmask = (
            permuted_labels
            == "REAL"
        )

        # Arco medio dei gruppi casuali.
        f_arc = (
            values[fmask]
            .mean(axis=0)
        )

        r_arc = (
            values[rmask]
            .mean(axis=0)
        )

        # Distanza sotto l'ipotesi nulla.
        permuted_stats[i] = np.sum(
            (
                f_arc
                - r_arc
            ) ** 2
        )

    # p-value:
    #
    # quante permutazioni producono
    # una distanza almeno grande
    # quanto quella osservata realmente?
    p = (
        np.sum(
            permuted_stats
            >= observed
        )
        + 1
    ) / (
        N_PERMUTATIONS
        + 1
    )

    return (
        observed,
        p,
        fake_arc,
        real_arc
    )


# ============================================================
# GRAFICO DELL'ARCO ASSOLUTO
# ============================================================

def make_arc_plot(
    summary: pd.DataFrame,
    measure: str,
    output_path: Path,
):
    """
    Disegna gli archi medi FAKE e REAL
    usando l'emotion strength assoluta.
    """

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    # Disegna una linea per ciascun gruppo.
    for label in [
        "FAKE",
        "REAL"
    ]:

        sub = summary[
            (
                summary["label"]
                == label
            )
            &
            (
                summary["measure"]
                == measure
            )
        ]

        x = sub[
            "section"
        ].to_numpy()

        mean = sub[
            "mean"
        ].to_numpy()

        se = sub[
            "sem"
        ].to_numpy()

        # Linea dell'arco.
        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2,
            label=label,
        )

        # Intervallo circa 95%:
        # media ± 1.96 * errore standard.
        ax.fill_between(
            x,
            mean - 1.96 * se,
            mean + 1.96 * se,
            alpha=0.15,
        )

    ax.set_xticks(
        range(
            1,
            N_SECTIONS + 1
        )
    )

    ax.set_xlabel(
        "Normalized article section"
    )

    ax.set_ylabel(
        "Emotion strength"
    )

    ax.set_title(
        "RQ2 — Emotion-strength arcs"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# GRAFICO DELL'ARCO CENTRATO
# ============================================================

def make_centered_arc_plot(
    long_df: pd.DataFrame,
    measure: str,
    output_path: Path,
):
    """
    Disegna gli archi dopo aver sottratto
    da ogni articolo la propria intensità media.

    Serve a visualizzare la FORMA
    dell'arco indipendentemente dal livello assoluto.
    """

    temp = long_df[
        [
            "article_id",
            "label",
            "section",
            measure
        ]
    ].copy()

    # Centering individuale.
    temp[
        "centered"
    ] = (
        temp[measure]
        -
        temp
        .groupby(
            "article_id"
        )[measure]
        .transform(
            "mean"
        )
    )

    # Media FAKE/REAL per sezione.
    summary = (
        temp
        .groupby(
            [
                "label",
                "section"
            ]
        )[
            "centered"
        ]
        .agg(
            [
                "mean",
                "sem"
            ]
        )
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    for label in [
        "FAKE",
        "REAL"
    ]:

        sub = summary[
            summary["label"]
            == label
        ]

        x = sub[
            "section"
        ].to_numpy()

        mean = sub[
            "mean"
        ].to_numpy()

        se = sub[
            "sem"
        ].to_numpy()

        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2,
            label=label,
        )

        ax.fill_between(
            x,
            mean - 1.96 * se,
            mean + 1.96 * se,
            alpha=0.15,
        )

    # Linea dello zero:
    # indica la media individuale.
    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_xticks(
        range(
            1,
            N_SECTIONS + 1
        )
    )

    ax.set_xlabel(
        "Normalized article section"
    )

    ax.set_ylabel(
        "Centered emotion strength"
    )

    ax.set_title(
        "RQ2 — Arc shape after article-level centering"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Pipeline completa della RQ2.

    1. controlla CUDA;
    2. carica il dataset;
    3. seleziona gli articoli narrativi;
    4. carica GoEmotions;
    5. divide ogni articolo in 5 sezioni;
    6. calcola emotion strength;
    7. costruisce gli archi;
    8. confronta FAKE e REAL per sezione;
    9. confronta globalmente gli archi;
    10. genera i grafici.
    """

    # GPU obbligatoria.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA non disponibile. "
            "Lo script richiede una GPU NVIDIA."
        )

    # Controllo del dataset.
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"File non trovato: "
            f"{INPUT_CSV}"
        )

    device = torch.device(
        "cuda:0"
    )

    # Generatore casuale riproducibile
    # per il permutation test.
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    print("=" * 76)
    print(
        "RQ2 — EMOTION-STRENGTH ARCS"
    )
    print("=" * 76)

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Modello emozioni: "
        f"{MODEL_ID}"
    )

    print(
        f"Sezioni normalizzate: "
        f"{N_SECTIONS}"
    )

    print(
        f"Soglia narratività: "
        f"{NARRATIVITY_THRESHOLD}"
    )

    print()

    # Carica dataset.
    df = pd.read_csv(
        INPUT_CSV
    )

    # Colonne minime necessarie.
    required = {
        "text",
        "label",
        NARRATIVITY_COLUMN,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Colonne mancanti: "
            f"{sorted(missing)}"
        )

    # Uniforma FAKE/REAL.
    df["label"] = normalize_label(
        df["label"]
    )

    # ========================================================
    # SELEZIONE ARTICOLI NARRATIVI
    # ========================================================

    # Mantiene solo gli articoli
    # con StorySeeker >= 0.50.
    selected = df[
        df[NARRATIVITY_COLUMN]
        >= NARRATIVITY_THRESHOLD
    ].copy()

    # Mantiene soltanto le due classi valide.
    selected = (
        selected[
            selected["label"]
            .isin(
                [
                    "FAKE",
                    "REAL"
                ]
            )
        ]
        .reset_index(
            drop=True
        )
    )

    # Se manca article_id,
    # ne viene creato uno.
    if (
        "article_id"
        not in selected.columns
    ):

        selected.insert(
            0,
            "article_id",
            np.arange(
                1,
                len(selected) + 1
            ),
        )

    print(
        f"Articoli totali: "
        f"{len(df):,}"
    )

    print(
        f"Articoli narrativi selezionati: "
        f"{len(selected):,}"
    )

    print(
        selected[
            "label"
        ]
        .value_counts()
        .to_string()
    )

    print()

    # ========================================================
    # CARICAMENTO GOEMOTIONS
    # ========================================================

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_ID
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_ID
        )
    )

    # Sposta sulla GPU.
    model.to(
        device
    )

    model.eval()

    # Recupera gli indici delle 27 emozioni.
    non_neutral_indices = (
        get_non_neutral_indices(
            model
        )
    )

    # ========================================================
    # PREPARAZIONE CHUNK
    # ========================================================

    (
        token_chunks,
        article_positions,
        section_positions
    ) = build_work_items(
        selected,
        tokenizer
    )

    print(
        f"\nChunk GoEmotions "
        f"da analizzare: "
        f"{len(token_chunks):,}"
    )

    # ========================================================
    # INFERENZA
    # ========================================================

    probs = infer_chunks(
        token_chunks,
        tokenizer,
        model,
        device,
    )

    # Riporta le predizioni
    # al livello articolo × sezione.
    long_df = aggregate_sections(
        selected,
        probs,
        article_positions,
        section_positions,
        non_neutral_indices,
    )

    # Crea cartella output.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Salva dataset long.
    long_df.to_csv(
        OUTPUT_LONG,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # MISURE DI EMOTION STRENGTH
    # ========================================================

    # Il risultato della RQ1 cambia se definisco la forza emotiva in un modo diverso?
    measures = [
        "emotion_strength_max", # prendiamo l'emozione più forte
        "emotion_strength_top3_mean", # la media delle 3 emozioni più forti
        "emotion_strength_rms", # considera tutte le 27 emozioni
    ]

    # ========================================================
    # STATISTICHE DESCRITTIVE
    # ========================================================

    summaries = []

    for measure in measures:

        tmp = (
            long_df
            .groupby(
                [
                    "label",
                    "section"
                ]
            )[measure]
            .agg(
                [
                    "count",
                    "mean",
                    "median",
                    "std",
                    "sem"
                ]
            )
            .reset_index()
        )

        # Aggiunge il nome della misura
        # usata per questa tabella.
        tmp.insert(
            0,
            "measure",
            measure
        )

        summaries.append(
            tmp
        )

    # Unisce i risultati
    # delle tre misure.
    summary = pd.concat(
        summaries,
        ignore_index=True,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # TEST PER SEZIONE (prende tutti i fake e tutti i real e confronta le distribuzioni)
    # ========================================================

    tests = pd.concat(

        [
            section_tests(
                long_df,
                measure
            )
            for measure in measures
        ],

        ignore_index=True,
    )

    tests.to_csv(
        OUTPUT_SECTION_TESTS,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # TEST GLOBALE DELL'ARCO (non confronta più una sezione alla volta, ma confronta la forma completa delle due curve)
    # ========================================================

    global_rows = []

    # Ripetiamo il test per ciascuna
    # definizione di emotion strength.
    for measure in measures:

        # centered=False:
        # confronta livello + forma.
        #
        # centered=True:
        # confronta principalmente la forma.
        for centered in [
            False,
            True
        ]:

            (
                statistic,
                p,
                fake_arc,
                real_arc
            ) = permutation_arc_test(

                long_df,
                measure,

                centered=centered,

                rng=rng,
            )

            global_rows.append({

                "measure":
                    measure,

                "comparison":
                    (
                        "shape_centered"
                        if centered
                        else "absolute_arc"
                    ),

                # Distanza totale tra
                # i due archi.
                "statistic_squared_distance":
                    statistic,

                # Significatività permutation test.
                "permutation_p_value":
                    p,

                "n_permutations":
                    N_PERMUTATIONS,

                # Valori medi delle 5 sezioni FAKE.
                "fake_section_1":
                    fake_arc[0],

                "fake_section_2":
                    fake_arc[1],

                "fake_section_3":
                    fake_arc[2],

                "fake_section_4":
                    fake_arc[3],

                "fake_section_5":
                    fake_arc[4],

                # Valori medi delle 5 sezioni REAL.
                "real_section_1":
                    real_arc[0],

                "real_section_2":
                    real_arc[1],

                "real_section_3":
                    real_arc[2],

                "real_section_4":
                    real_arc[3],

                "real_section_5":
                    real_arc[4],
            })

    global_tests = pd.DataFrame(
        global_rows
    )

    global_tests.to_csv(
        OUTPUT_GLOBAL_TEST,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # GRAFICI
    # ========================================================

    # Grafico principale:
    # valori assoluti.
    make_arc_plot(
        summary,
        "emotion_strength_max",
        OUTPUT_DIR
        / "rq2_emotion_arc_primary.png",
    )

    # Grafico centrato:
    # evidenzia la forma.
    make_centered_arc_plot(
        long_df,
        "emotion_strength_max",
        OUTPUT_DIR
        / "rq2_emotion_arc_centered.png",
    )

    # ========================================================
    # OUTPUT CONSOLE
    # ========================================================

    print()

    print("=" * 76)
    print(
        "ARCO PRINCIPALE — "
        "emotion_strength_max"
    )
    print("=" * 76)

    # Seleziona solo la misura primaria.
    primary_summary = summary[
        summary["measure"]
        == "emotion_strength_max"
    ]

    print(
        primary_summary[
            [
                "label",
                "section",
                "count",
                "mean",
                "median",
                "std"
            ]
        ]
        .round(5)
        .to_string(index=False)
    )

    print()

    print("=" * 76)
    print(
        "FAKE vs REAL PER SEZIONE"
    )
    print("=" * 76)

    # Test relativi alla misura primaria.
    primary_tests = tests[
        tests["measure"]
        == "emotion_strength_max"
    ]

    print(
        primary_tests[
            [
                "section",
                "fake_mean",
                "real_mean",
                "difference_fake_minus_real",
                "cliffs_delta_fake_vs_real",
                "fdr_q_value",
            ]
        ]
        .round(5)
        .to_string(index=False)
    )

    print()

    print("=" * 76)
    print(
        "TEST GLOBALE DELL'ARCO"
    )
    print("=" * 76)

    print(
        global_tests[
            global_tests["measure"]
            == "emotion_strength_max"
        ][
            [
                "comparison",
                "statistic_squared_distance",
                "permutation_p_value",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print()

    print("=" * 76)
    print("OUTPUT")
    print("=" * 76)

    print(
        f"- {OUTPUT_LONG}"
    )

    print(
        f"- {OUTPUT_SUMMARY}"
    )

    print(
        f"- {OUTPUT_SECTION_TESTS}"
    )

    print(
        f"- {OUTPUT_GLOBAL_TEST}"
    )

    print(
        f"- "
        f"{OUTPUT_DIR / 'rq2_emotion_arc_primary.png'}"
    )

    print(
        f"- "
        f"{OUTPUT_DIR / 'rq2_emotion_arc_centered.png'}"
    )

    print(
        "\nPRIMARY: emotion_strength_max.\n"

        "top3_mean e RMS sono "
        "sensitivity analyses.\n"

        "Il test 'absolute_arc' confronta "
        "l'intero profilo FAKE/REAL.\n"

        "Il test 'shape_centered' sottrae "
        "la media di ogni articolo "
        "e verifica se cambia la FORMA "
        "dell'arco indipendentemente "
        "dal livello emotivo complessivo."
    )


if __name__ == "__main__":
    main()