"""
Modulo: merge_narrativity_emotions.py

Scopo:
unire il dataset con i punteggi di narratività StorySeeker
al dataset con le probabilità GoEmotions.

Dopo il merge:
- calcola una misura sintetica di intensità emotiva;
- identifica l'emozione dominante;
- calcola le correlazioni tra narratività e emotion strength;
- salva il dataset finale usato nella RQ1.
"""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Dataset contenente i punteggi StorySeeker.
NARRATIVITY_CSV = Path(
    "../../data/interim/dataset_with_storyseeker.csv"
)

# Dataset contenente le probabilità GoEmotions.
EMOTIONS_CSV = Path(
    "../../data/interim/dataset_with_emotions.csv"
)

# Dataset finale ottenuto unendo narratività ed emozioni.
OUTPUT_MERGED = Path(
    "../../data/processed/dataset_narrativity_emotions.csv"
)

# File con i risultati delle correlazioni RQ1.
OUTPUT_RESULTS = Path(
    "../../archive/old results/rq1/rq1_correlations.csv"
)


# Le 27 emozioni GoEmotions non neutrali.
#
# "neutral" viene esclusa perché la RQ1 vuole misurare
# la forza emotiva, non la neutralità del testo.
EMOTION_COLUMNS = [
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
    "surprise"
]


# Misure StorySeeker disponibili per ciascun articolo.
NARRATIVITY_COLUMNS = [
    "narrativity_first_chunk",
    "narrativity_mean",
    "narrativity_max"
]


# ============================================================
# NORMALIZZAZIONE DELLE CHIAVI
# ============================================================

def normalize_key(
    series: pd.Series
) -> pd.Series:
    """
    Normalizza una colonna usata come chiave di merge.

    Operazioni:
    - sostituisce i valori nulli con stringhe vuote;
    - converte tutto in stringa;
    - rimuove spazi iniziali e finali.

    Questo serve a ridurre problemi di merge dovuti
    a differenze banali di formattazione.
    """

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


# ============================================================
# CARICAMENTO DEL DATASET NARRATIVITÀ
# ============================================================

def load_narrativity() -> pd.DataFrame:
    """
    Carica il dataset StorySeeker e normalizza
    le colonne usate per il merge.
    """

    df = pd.read_csv(
        NARRATIVITY_CSV
    )

    # Queste colonne devono combaciare con il dataset
    # delle emozioni per poter identificare lo stesso articolo.
    for column in [
        "title",
        "subject",
        "date",
        "label"
    ]:
        df[column] = normalize_key(
            df[column]
        )

    return df


# ============================================================
# CARICAMENTO DEL DATASET EMOZIONI
# ============================================================

def load_emotions() -> pd.DataFrame:
    """
    Carica il dataset GoEmotions e uniforma le label.

    Nel dataset delle emozioni:
        1 = FAKE
        0 = REAL

    Nel dataset StorySeeker invece:
        "FAKE"
        "REAL"

    Le due rappresentazioni devono quindi essere uniformate
    prima del merge.
    """

    df = pd.read_csv(
        EMOTIONS_CSV
    )

    # Se la colonna label è numerica:
    #
    # 1 -> FAKE
    # 0 -> REAL
    if pd.api.types.is_numeric_dtype(
        df["label"]
    ):

        df["label"] = (
            df["label"]
            .map({
                1: "FAKE",
                0: "REAL"
            })
        )

    else:
        # Se la label è già testuale,
        # la normalizziamo e la trasformiamo in maiuscolo.
        df["label"] = (
            normalize_key(
                df["label"]
            )
            .str.upper()
        )

    # Normalizza le altre chiavi del merge.
    for column in [
        "title",
        "subject",
        "date"
    ]:

        df[column] = normalize_key(
            df[column]
        )

    # Controlla che tutte le 27 colonne emotive
    # siano presenti nel dataset.
    missing = [
        column
        for column in EMOTION_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colonne emotive mancanti: "
            f"{missing}"
        )

    return df


# ============================================================
# GESTIONE DEI DUPLICATI
# ============================================================

def aggregate_emotion_duplicates(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Riduce eventuali righe duplicate nel dataset emozioni.

    Alcuni articoli possono comparire più volte.
    Per ottenere una singola riga per articolo,
    calcoliamo la media delle probabilità emotive.
    """

    # Le quattro colonne identificano l'articolo
    # nel merge.
    keys = [
        "title",
        "label",
        "subject",
        "date"
    ]

    # Raggruppa tutte le righe con le stesse chiavi
    # e calcola la media delle probabilità.
    return (
        df
        .groupby(
            keys,
            as_index=False
        )[
            EMOTION_COLUMNS
            + ["neutral"]
        ]
        .mean()
    )


# ============================================================
# COSTRUZIONE DELLA FORZA EMOTIVA
# ============================================================

def add_emotion_strength(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Costruisce due nuove variabili:

    emotion_strength
        probabilità più alta tra le 27 emozioni non neutrali.

    dominant_emotion
        nome dell'emozione con probabilità più alta.

    Esempio:
        anger     = 0.10
        sadness   = 0.65
        fear      = 0.22

    allora:
        emotion_strength = 0.65
        dominant_emotion = "sadness"
    """

    df = df.copy()

    # Intensità emotiva primaria:
    # prendiamo la probabilità massima
    # tra tutte le emozioni non neutrali.
    df["emotion_strength"] = (
        df[EMOTION_COLUMNS]
        .max(axis=1)
    )

    # Individua quale emozione ha prodotto
    # il valore massimo.
    df["dominant_emotion"] = (
        df[EMOTION_COLUMNS]
        .idxmax(axis=1)
    )

    return df


# ============================================================
# CORRELAZIONI RQ1
# ============================================================

def calculate_correlations(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcola la correlazione di Spearman tra:

        narratività
            e
        emotion_strength

    separatamente per:
    - tutti gli articoli;
    - FAKE;
    - REAL.

    Vengono testate tutte e tre le misure StorySeeker:
    - first_chunk
    - mean
    - max
    """

    rows = []

    # Analizziamo prima tutto il dataset,
    # poi separatamente FAKE e REAL.
    for group_name, group in [

        ("ALL", df),

        (
            "FAKE",
            df[
                df["label"] == "FAKE"
            ]
        ),

        (
            "REAL",
            df[
                df["label"] == "REAL"
            ]
        )
    ]:

        # Ripete il test per ciascuna misura
        # di narratività.
        for narrativity_column in (
            NARRATIVITY_COLUMNS
        ):

            # Elimina solo le righe con valori mancanti
            # nelle due variabili analizzate.
            valid = group[
                [
                    narrativity_column,
                    "emotion_strength"
                ]
            ].dropna()

            # Spearman misura una relazione monotona
            # senza assumere linearità o normalità.
            #
            # rho:
            #   forza e direzione dell'associazione
            #
            # p_value:
            #   significatività statistica
            rho, p_value = spearmanr(
                valid[narrativity_column],
                valid["emotion_strength"]
            )

            rows.append({

                # Gruppo analizzato.
                "group": group_name,

                # Misura StorySeeker utilizzata.
                "narrativity_measure":
                    narrativity_column,

                # Numero di articoli validi.
                "n_articles":
                    len(valid),

                # Coefficiente di correlazione Spearman.
                "spearman_rho":
                    rho,

                # p-value associato.
                "p_value":
                    p_value
            })

    return pd.DataFrame(
        rows
    )



# ============================================================
# GRAFICO RISULTATI MERGE / RQ1 PRELIMINARE
# ============================================================

def generate_merge_correlation_plot(results: pd.DataFrame):
    """Genera un bar chart delle correlazioni StorySeeker × emotion strength."""
    import matplotlib.pyplot as plt
    import numpy as np

    measures = NARRATIVITY_COLUMNS
    labels = ["First chunk", "Mean", "Max"]
    y = np.arange(len(measures))
    height = 0.24

    fig, ax = plt.subplots(figsize=(11, 6))
    for offset, group in zip([-height, 0, height], ["ALL", "FAKE", "REAL"]):
        subset = results[results["group"] == group].set_index("narrativity_measure")
        values = [subset.loc[m, "spearman_rho"] for m in measures]
        ax.barh(y + offset, values, height=height, label=group)

    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman correlation with emotion strength")
    ax.set_ylabel("StorySeeker measure")
    ax.set_title("RQ1 preliminary — Narrativity vs emotion strength")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    output_dir = OUTPUT_MERGED.parent / "rq1"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rq1_storyseeker_emotion_strength_correlations.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico correlazioni salvato in: {output_path.resolve()}")

# ============================================================
# MAIN
# ============================================================

def main():
    """
    Pipeline completa di merge per la RQ1.

    1. carica StorySeeker;
    2. carica GoEmotions;
    3. aggrega eventuali duplicati;
    4. unisce i due dataset;
    5. costruisce emotion_strength;
    6. calcola le correlazioni;
    7. salva dataset e risultati.
    """

    # Caricamento del dataset StorySeeker.
    narrativity = load_narrativity()

    # Caricamento del dataset GoEmotions
    # e aggregazione degli eventuali duplicati.
    emotions = (
        aggregate_emotion_duplicates(
            load_emotions()
        )
    )

    # Chiavi utilizzate per riconoscere
    # lo stesso articolo nei due dataset.
    keys = [
        "title",
        "label",
        "subject",
        "date"
    ]

    # Left join:
    #
    # manteniamo tutti gli articoli StorySeeker
    # e proviamo ad associare le emozioni.
    merged = narrativity.merge(

        emotions,

        on=keys,

        how="left",

        # Ogni articolo StorySeeker deve trovare
        # al massimo una riga nel dataset emozioni.
        validate="many_to_one",

        # Crea una colonna temporanea "_merge"
        # che permette di capire quali righe
        # sono state unite correttamente.
        indicator=True
    )

    # Conta quanti articoli non hanno trovato
    # una corrispondenza nel dataset emozioni.
    unmatched = (
        merged["_merge"] != "both"
    ).sum()

    print(
        f"Articoli narratività: "
        f"{len(narrativity)}"
    )

    print(
        f"Articoli uniti correttamente: "
        f"{len(merged) - unmatched}"
    )

    print(
        f"Articoli senza emozioni: "
        f"{unmatched}"
    )

    if unmatched > 0:
        print(
            "ATTENZIONE: alcuni articoli "
            "non sono stati uniti."
        )

    # Manteniamo soltanto gli articoli
    # presenti in entrambi i dataset.
    merged = (
        merged[
            merged["_merge"] == "both"
        ]
        .drop(
            columns="_merge"
        )
    )

    # Costruisce emotion_strength
    # e dominant_emotion.
    merged = add_emotion_strength(
        merged
    )

    # Calcola le correlazioni RQ1.
    results = calculate_correlations(
        merged
    )

    # Crea la directory di output se necessaria.
    OUTPUT_MERGED.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Salva il dataset completo:
    #
    # StorySeeker + GoEmotions + emotion_strength.
    merged.to_csv(
        OUTPUT_MERGED,
        index=False,
        encoding="utf-8-sig"
    )

    # Salva la tabella delle correlazioni.
    results.to_csv(
        OUTPUT_RESULTS,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nDataset unito salvato in: "
        f"{OUTPUT_MERGED.resolve()}"
    )

    print(
        f"Risultati RQ1 salvati in: "
        f"{OUTPUT_RESULTS.resolve()}"
    )

    print(
        "\nCorrelazioni:"
    )

    print(
        results
        .round(5)
        .to_string(index=False)
    )


    # Generazione immagine delle correlazioni preliminari.
    generate_merge_correlation_plot(results)


if __name__ == "__main__":
    main()