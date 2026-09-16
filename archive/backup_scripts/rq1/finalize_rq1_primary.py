"""
Modulo: finalize_rq1_primary.py

Scopo:
finalizzare la RQ1 principale verificando la relazione tra:

    narratività StorySeeker (per ogni articolo prendiamo lo score del primo chunk)
            e
    forza emotiva GoEmotions (per ogni articolo abbiamo 27 probabilità)

L'analisi principale usa:
    narrativity_first_chunk
    emotion_strength_max

Inoltre vengono calcolate due sensitivity analysis:
    emotion_strength_top3_mean
    emotion_strength_rms

Questo serve a verificare che il risultato non dipenda
da una sola definizione di intensità emotiva.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Root del progetto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset già unito:
# StorySeeker + GoEmotions.
INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_narrativity_emotions.csv"
)

# Cartella dedicata alla RQ1 principale.
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rq1"
    / "primary"
)

# File con le correlazioni finali.
OUTPUT_RESULTS = (
    OUTPUT_DIR
    / "rq1_primary_correlations.csv"
)

# Dataset arricchito con le tre misure
# di emotion strength.
OUTPUT_DATASET = (
    OUTPUT_DIR
    / "rq1_primary_dataset.csv"
)


# Misura di narratività scelta come principale.
#
# StorySeeker viene applicato a chunk dell'articolo.
# Qui usiamo il primo chunk come misura primaria.
PRIMARY_NARRATIVITY = "narrativity_first_chunk"


# Manteniamo anche mean e max
# per verifiche alternative.
NARRATIVITY_COLUMNS = [
    "narrativity_first_chunk",
    "narrativity_mean",
    "narrativity_max",
]


# ============================================================
# ETICHETTE GOEMOTIONS
# ============================================================

# Le 27 emozioni non neutrali.
#
# La neutralità viene esclusa perché
# vogliamo misurare la forza emotiva.
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
    "surprise",
]


# ============================================================
# NORMALIZZAZIONE LABEL
# ============================================================

def normalize_label(
    series: pd.Series
) -> pd.Series:
    """
    Uniforma la colonna label nel formato:

        FAKE
        REAL

    Gestisce sia valori numerici:
        1 -> FAKE
        0 -> REAL

    sia valori testuali:
        TRUE  -> REAL
        FALSE -> FAKE
    """

    # Prova a convertire la colonna in numerico.
    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    # Crea una versione testuale normalizzata.
    out = (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Mappatura numerica.
    out.loc[
        numeric == 1
    ] = "FAKE"

    out.loc[
        numeric == 0
    ] = "REAL"

    # Mappatura testuale alternativa.
    return out.replace({
        "TRUE": "REAL",
        "FALSE": "FAKE",
        "1.0": "FAKE",
        "0.0": "REAL",
    })


# ============================================================
# VALIDAZIONE COLONNE
# ============================================================

def validate_columns(
    df: pd.DataFrame
):
    """
    Controlla che il dataset contenga tutte
    le colonne necessarie per l'analisi.
    """

    required = {
        "label",
        *NARRATIVITY_COLUMNS,
        *EMOTION_COLUMNS
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Colonne mancanti nel dataset:\n"
            + "\n".join(missing)
        )


# ============================================================
# COSTRUZIONE DELLE MISURE DI EMOTION STRENGTH
# ============================================================

def build_emotion_strengths(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Costruisce tre misure di forza emotiva.

    1. emotion_strength_max
       Massima probabilità tra le 27 emozioni.

    Queste altre due non sono l'analisi principale: sono sensitivity analysis, controlli per vedere se il risultato cambia usando un modo diverso di definire la forza emotiva
    2. emotion_strength_top3_mean
       Media delle tre emozioni più forti.

    3. emotion_strength_rms
       Root Mean Square delle 27 probabilità.

    La prima è la misura principale.
    Le altre due servono come sensitivity analysis.
    """

    # Converte le 27 colonne emotive
    # in una matrice NumPy:
    #
    # articoli × emozioni.
    values = (
        df[EMOTION_COLUMNS]
        .astype(float)
        .to_numpy()
    )

    # --------------------------------------------------------
    # MISURA 1: MAX
    # --------------------------------------------------------

    # Per ogni articolo prende la probabilità
    # della singola emozione più forte.
    #
    # Esempio:
    #
    # anger   0.20
    # sadness 0.65
    # fear    0.10
    #
    # max = 0.65
    df["emotion_strength_max"] = (
        np.max(
            values,
            axis=1
        )
    )

    # --------------------------------------------------------
    # MISURA 2: TOP 3 MEAN
    # --------------------------------------------------------

    # Ordina le probabilità per ogni articolo.
    #
    # [:, -3:] prende le tre più alte.
    top3 = (
        np.sort(
            values,
            axis=1
        )[:, -3:]
    )

    # Calcola la media delle tre emozioni
    # più forti.
    df[
        "emotion_strength_top3_mean"
    ] = np.mean(
        top3,
        axis=1
    )

    # --------------------------------------------------------
    # MISURA 3: RMS
    # --------------------------------------------------------

    # Root Mean Square:
    #
    # sqrt(mean(x^2))
    #
    # Tiene conto di tutto il profilo emotivo
    # dando più peso ai valori alti.
    df[
        "emotion_strength_rms"
    ] = np.sqrt(
        np.mean(
            np.square(values),
            axis=1
        )
    )

    return df


# ============================================================
# CALCOLO DELLE CORRELAZIONI
# ============================================================

def correlate(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcola tutte le correlazioni Spearman tra:

    3 misure di narratività
    ×
    3 misure di emotion strength

    separatamente per:
    - ALL
    - FAKE
    - REAL

    In totale:
        3 gruppi × 3 narratività × 3 strength
        = 27 correlazioni.

    Domanda: Quando aumenta la narratività, tende ad aumentare anche la forza emotiva?
    """

    rows = []

    # Le tre misure di forza emotiva.
    strength_columns = [
        "emotion_strength_max", # le altre due colonne servono a controllare che il risultato non dipenda dal fatto che abbiamo scelto il massimo!
        "emotion_strength_top3_mean",
        "emotion_strength_rms",
    ]

    # Analisi per gruppo.
    for scope, subset in [

        (
            "ALL",
            df
        ),

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
        ),
    ]:

        # Per ogni misura StorySeeker.
        for narr_col in (
            NARRATIVITY_COLUMNS
        ):

            # Per ogni definizione di emotion strength.
            for strength_col in (
                strength_columns
            ):

                # Mantiene solo le righe
                # senza valori mancanti.
                valid = subset[
                    [
                        narr_col,
                        strength_col
                    ]
                ].dropna()

                # Spearman misura associazioni monotone.
                # rho = +0.80 forte relazione positiva
                # rho = 0 nessuna relazione
                # rho = -0.80 quando una aumenta, l'altra tende a diminuire
                rho, p = spearmanr(
                    valid[narr_col],
                    valid[strength_col],
                )

                rows.append({

                    # Gruppo analizzato.
                    "scope":
                        scope,

                    # Numero di articoli.
                    "n":
                        len(valid),

                    # Misura StorySeeker.
                    "narrativity_measure":
                        narr_col,

                    # Definizione di emotion strength.
                    "emotion_strength_measure":
                        strength_col,

                    # Coefficiente Spearman.
                    "spearman_rho":
                        rho,

                    # Significatività statistica.
                    "p_value":
                        p,

                    # Identifica la combinazione
                    # scelta come analisi principale.
                    "primary_analysis": (
                        narr_col
                        == PRIMARY_NARRATIVITY

                        and

                        strength_col
                        == "emotion_strength_max"
                    ),
                })

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Pipeline finale della RQ1 principale.

    1. carica dataset StorySeeker + GoEmotions;
    2. controlla le colonne;
    3. normalizza FAKE/REAL;
    4. ricalcola le misure di emotion strength;
    5. calcola le correlazioni;
    6. salva risultati;
    7. stampa primary analysis e sensitivity analysis.
    """

    print("=" * 72)
    print(
        "RQ1 PRIMARY — "
        "NARRATIVITY vs EMOTION STRENGTH"
    )
    print("=" * 72)

    # Controllo file di input.
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"File non trovato: "
            f"{INPUT_CSV}"
        )

    # Carica il dataset.
    df = pd.read_csv(
        INPUT_CSV
    )

    # Verifica che tutte le colonne necessarie
    # siano effettivamente presenti.
    validate_columns(
        df
    )

    # Uniforma le label.
    df["label"] = normalize_label(
        df["label"]
    )

    print(
        f"Articoli: "
        f"{len(df):,}"
    )

    print(
        f"Emozioni non-neutral usate: "
        f"{len(EMOTION_COLUMNS)}"
    )

    print(
        ", ".join(
            EMOTION_COLUMNS
        )
    )

    print()

    # --------------------------------------------------------
    # RIMOZIONE DI EVENTUALI VECCHIE COLONNE
    # --------------------------------------------------------

    # Questa parte è importante.
    #
    # Se il dataset contiene già colonne con nomi
    # emotion_strength_* le eliminiamo prima di ricalcolarle.
    #
    # Serve a evitare contaminazioni o risultati
    # provenienti da versioni precedenti.
    for col in [
        "emotion_strength_max",
        "emotion_strength_top3_mean",
        "emotion_strength_rms",
    ]:

        if col in df.columns:
            df.drop(
                columns=col,
                inplace=True
            )

    # Costruisce le tre nuove misure.
    df = build_emotion_strengths(
        df
    )

    # Calcola tutte le correlazioni.
    results = correlate(
        df
    )

    # Crea la directory di output.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Salva il dataset finale.
    df.to_csv(
        OUTPUT_DATASET,
        index=False,
        encoding="utf-8",
    )

    # Salva la tabella completa
    # delle correlazioni.
    results.to_csv(
        OUTPUT_RESULTS,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # ANALISI PRINCIPALE
    # ========================================================

    print("=" * 72)
    print(
        "ANALISI PRINCIPALE"
    )
    print("=" * 72)

    # Filtra solo le righe marcate
    # come primary_analysis.
    #
    # Quindi:
    #
    # narrativity_first_chunk
    # ×
    # emotion_strength_max
    primary = results[
        results["primary_analysis"]
    ][
        [
            "scope",
            "n",
            "spearman_rho",
            "p_value"
        ]
    ]

    print(
        primary
        .round(6)
        .to_string(index=False)
    )

    print()

    # ========================================================
    # SENSITIVITY ANALYSIS
    # ========================================================

    print("=" * 72)
    print(
        "SENSITIVITY ANALYSIS"
    )
    print("=" * 72)

    # Mantiene fissa la narratività primaria:
    #
    # narrativity_first_chunk
    #
    # e confronta le tre definizioni
    # di emotion strength.
    sensitivity = results[

        results[
            "narrativity_measure"
        ] == PRIMARY_NARRATIVITY

    ][
        [
            "scope",
            "emotion_strength_measure",
            "spearman_rho",
            "p_value",
        ]
    ]

    print(
        sensitivity
        .round(6)
        .to_string(index=False)
    )

    print()

    # ========================================================
    # OUTPUT
    # ========================================================

    print("=" * 72)
    print("OUTPUT")
    print("=" * 72)

    print(
        f"- {OUTPUT_RESULTS}"
    )

    print(
        f"- {OUTPUT_DATASET}"
    )

    print(
        "\nPRIMARY = "
        "narrativity_first_chunk "
        "× emotion_strength_max\n"

        "Le misure top3_mean e RMS "
        "sono sensitivity analysis.\n"

        "Le 27 colonne GoEmotions "
        "sono definite esplicitamente; "
        "neutral e colonne derivate "
        "sono escluse."
    )


if __name__ == "__main__":
    main()