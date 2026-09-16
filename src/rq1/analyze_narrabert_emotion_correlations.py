"""
Modulo: analyze_narrabert_emotion_correlations.py

Scopo:
analizzare la relazione tra le 9 dimensioni narrative di NarraBERT
e le 27 emozioni GoEmotions.

L'analisi viene svolta separatamente per:
- ALL
- FAKE
- REAL

Per ogni coppia:
    dimensione narrativa (9) × emozione (27)

viene calcolata una correlazione di Spearman.

Inoltre:
- i p-value vengono corretti con Benjamini-Hochberg;
- confrontiamo prima FAKE, poi REAL e facciamo rho_FAKE - rho_REAL (quindi positivo relazione più forte nei fake, negativo nei real)
- vengono generate heatmap riassuntive.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Root del progetto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset con le 9 dimensioni NarraBERT.
NARRATIVITY_CSV = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dataset_with_narrativity_narrabert.csv"
)

# Dataset con le probabilità GoEmotions.
EMOTIONS_CSV = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dataset_with_emotions.csv"
)

# Directory dove vengono salvati i risultati
# dell'estensione NarraBERT della RQ1.
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rq1_narrabert"
)

# Dataset finale unito NarraBERT + GoEmotions.
MERGED_CSV = (
    OUTPUT_DIR
    / "dataset_narrabert_emotions.csv"
)

# Correlazioni dimensione narrativa × emozione.
CORRELATIONS_CSV = (
    OUTPUT_DIR
    / "rq1_narrabert_emotion_correlations.csv"
)

# Differenze tra correlazioni FAKE e REAL.
DIFFERENCES_CSV = (
    OUTPUT_DIR
    / "rq1_fake_real_correlation_differences.csv"
)


# ============================================================
# SCELTA DELLA MISURA NARRATIVA
# ============================================================

# Per questa analisi usiamo la media dei chunk dell'articolo.
#
# Quindi, ad esempio:
#
# narrabert_emotion_mean
# narrabert_conflict_mean
# ...
#
# La media rappresenta la dimensione narrativa
# lungo l'intero articolo.
NARRATIVE_SUFFIX = "_mean"


# Le 9 dimensioni prodotte da NarraBERT.
NARRATIVE_DIMENSIONS = [
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
# COLONNE CHE NON SONO EMOZIONI
# ============================================================

# Questo insieme serve a evitare che colonne tecniche
# vengano erroneamente interpretate come emozioni.
NON_EMOTION_COLUMNS = {
    "article_id",
    "title",
    "text",
    "subject",
    "date",
    "label",
    "word_count",
    "neutral",
}


# Chiavi concettuali del merge.
MERGE_KEYS = [
    "title",
    "label",
    "subject",
    "date"
]


# ============================================================
# NORMALIZZAZIONE DELLE CHIAVI TESTUALI
# ============================================================

def normalize_text_key(
    series: pd.Series
) -> pd.Series:
    """
    Normalizza una colonna testuale usata per il merge.

    Operazioni:
    - sostituisce NaN con stringa vuota;
    - converte in stringa;
    - elimina spazi iniziali/finali;
    - converte in minuscolo;
    - comprime spazi multipli.

    Serve per rendere più robusto il confronto
    tra chiavi provenienti da due CSV diversi.
    """

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
    )


# ============================================================
# NORMALIZZAZIONE DELLA LABEL
# ============================================================

def normalize_label(
    series: pd.Series
) -> pd.Series:
    """
    Uniforma la label nel formato:

        FAKE
        REAL

    Gestisce:
    - 1 / 0
    - "FAKE" / "REAL"
    - "TRUE" / "FALSE"
    """

    s = series.copy()

    # Prova a leggere la label come numerica.
    numeric = pd.to_numeric(
        s,
        errors="coerce"
    )

    # Indica quali valori sono effettivamente numerici.
    numeric_mask = numeric.notna()

    # Versione testuale normalizzata.
    out = (
        s
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # 1 -> FAKE
    out.loc[
        numeric_mask
        & (numeric == 1)
    ] = "FAKE"

    # 0 -> REAL
    out.loc[
        numeric_mask
        & (numeric == 0)
    ] = "REAL"

    # Gestione varianti testuali.
    replacements = {
        "TRUE": "REAL",
        "FALSE": "FAKE",
        "1.0": "FAKE",
        "0.0": "REAL",
    }

    return out.replace(
        replacements
    )


# ============================================================
# COSTRUZIONE DELLE CHIAVI DI MERGE
# ============================================================

def prepare_merge_keys(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Crea versioni normalizzate delle colonne
    usate per il merge.

    Le colonne originali non vengono modificate
    definitivamente: vengono create colonne tecniche
    prefissate con "_".
    """

    df = df.copy()

    # La label è necessaria.
    if "label" not in df.columns:
        raise ValueError(
            "Colonna 'label' mancante."
        )

    # Uniforma FAKE / REAL.
    df["label"] = normalize_label(
        df["label"]
    )

    # Per titolo, subject e date
    # creiamo colonne normalizzate.
    for col in [
        "title",
        "subject",
        "date"
    ]:

        if col not in df.columns:
            raise ValueError(
                f"Colonna '{col}' mancante."
            )

        df[
            f"_{col}_key"
        ] = normalize_text_key(
            df[col]
        )

    # Anche la label viene trasformata
    # in chiave normalizzata.
    df[
        "_label_key"
    ] = normalize_text_key(
        df["label"]
    )

    return df


# ============================================================
# RILEVAMENTO DELLE COLONNE EMOTIVE
# ============================================================

def detect_emotion_columns(
    df: pd.DataFrame
) -> list[str]:
    """
    Cerca automaticamente le colonne GoEmotions.

    Una colonna viene considerata candidata se:
    - non è una colonna tecnica;
    - è quasi interamente numerica;
    - contiene valori nell'intervallo [0, 1].

    'neutral' viene esclusa.
    """

    candidates = []

    for col in df.columns:

        # Esclude colonne note non emotive.
        if col in NON_EMOTION_COLUMNS:
            continue

        # Esclude chiavi tecniche create per il merge.
        if col.startswith("_"):
            continue

        # Prova a convertire la colonna in numerico.
        numeric = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        # Richiediamo che almeno il 95% dei valori
        # sia numerico.
        if (
            numeric.notna().mean()
            < 0.95
        ):
            continue

        valid = numeric.dropna()

        if len(valid) == 0:
            continue

        # Le probabilità GoEmotions
        # devono stare nell'intervallo [0,1].
        if (
            valid.min() >= 0
            and valid.max() <= 1
        ):
            candidates.append(
                col
            )

    # Neutral viene esclusa
    # anche se fosse stata rilevata.
    candidates = [
        c
        for c in candidates
        if c.lower() != "neutral"
    ]

    return candidates


# ============================================================
# CORREZIONE MULTIPLA BENJAMINI-HOCHBERG
# ============================================================

def benjamini_hochberg(
    p_values: np.ndarray
) -> np.ndarray:
    """
    Corregge una serie di p-value
    controllando la False Discovery Rate.

    Perché serve:
    in questa analisi eseguiamo molte correlazioni.

    9 dimensioni narrative × molte emozioni.

    Senza correzione aumenterebbe la probabilità
    di trovare significatività per puro caso.
    """

    p = np.asarray(
        p_values,
        dtype=float
    )

    # Inizialmente tutti NaN.
    q = np.full_like(
        p,
        np.nan,
        dtype=float
    )

    # Consideriamo solo p-value validi.
    valid_mask = np.isfinite(
        p
    )

    if not valid_mask.any():
        return q

    pv = p[
        valid_mask
    ]

    # Ordine crescente dei p-value.
    order = np.argsort(
        pv
    )

    ranked = pv[
        order
    ]

    # Numero di test.
    m = len(
        ranked
    )

    # Formula base BH:
    #
    # p_i * m / rango
    adjusted = (
        ranked
        * m
        / np.arange(
            1,
            m + 1
        )
    )

    # Impone monotonicità ai q-value.
    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    # I q-value devono stare tra 0 e 1.
    adjusted = np.clip(
        adjusted,
        0,
        1
    )

    # Ripristina l'ordine originale.
    restored = np.empty_like(
        adjusted
    )

    restored[
        order
    ] = adjusted

    q[
        valid_mask
    ] = restored

    return q


# ============================================================
# CONFRONTO TRA DUE CORRELAZIONI
# ============================================================

def fisher_difference_test(
    rho_fake: float,
    n_fake: int,
    rho_real: float,
    n_real: int,
) -> tuple[float, float]:
    """
    Confronta approssimativamente due correlazioni indipendenti:

        rho_FAKE
        rho_REAL

    usando la trasformazione Fisher z.

    Restituisce:
    - statistica z;
    - p-value della differenza.

    Nel nostro caso viene usato in modo esplorativo
    anche sulle correlazioni Spearman.
    """

    # Controlli di validità.
    if (
        not np.isfinite(rho_fake)
        or not np.isfinite(rho_real)
        or n_fake <= 3
        or n_real <= 3
    ):
        return (
            np.nan,
            np.nan
        )

    # arctanh diverge esattamente per rho = ±1.
    # Quindi limitiamo leggermente l'intervallo.
    rf = np.clip(
        rho_fake,
        -0.999999,
        0.999999
    )

    rr = np.clip(
        rho_real,
        -0.999999,
        0.999999
    )

    # Trasformazione Fisher.
    zf = np.arctanh(
        rf
    )

    zr = np.arctanh(
        rr
    )

    # Errore standard della differenza.
    se = math.sqrt(
        1.0 / (n_fake - 3)
        +
        1.0 / (n_real - 3)
    )

    # Statistica z.
    z = (
        zf - zr
    ) / se

    # Test bilaterale.
    p = (
        2.0
        * norm.sf(
            abs(z)
        )
    )

    return (
        z,
        p
    )


# ============================================================
# CORRELAZIONI NARRABERT × GOEMOTIONS
# ============================================================

def compute_correlations(
    df: pd.DataFrame,
    narrative_cols: list[str],
    emotion_cols: list[str],
) -> pd.DataFrame:
    """
    Calcola Spearman per tutte le coppie:

        dimensione NarraBERT × emozione GoEmotions

    separatamente per:
    - ALL
    - FAKE
    - REAL
    """

    rows = []

    # Tre gruppi di analisi.
    for scope, subset in [

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
        ),
    ]:

        # Per ogni dimensione narrativa.
        for narrative_col in (
            narrative_cols
        ):

            # Per ogni emozione.
            for emotion_col in (
                emotion_cols
            ):

                # Prende solo le due colonne
                # che servono alla correlazione.
                valid = subset[
                    [
                        narrative_col,
                        emotion_col
                    ]
                ].dropna()

                # Se ci sono troppi pochi dati
                # non calcoliamo la correlazione.
                if len(valid) < 4:
                    rho = np.nan
                    p = np.nan

                else:
                    rho, p = spearmanr(
                        valid[narrative_col],
                        valid[emotion_col],
                    )

                rows.append({

                    "scope":
                        scope,

                    "n":
                        len(valid),

                    # Nome leggibile della dimensione.
                    #
                    # Da:
                    # narrabert_conflict_mean
                    #
                    # a:
                    # conflict
                    "narrative_dimension": (
                        narrative_col
                        .replace(
                            "narrabert_",
                            ""
                        )
                        .replace(
                            NARRATIVE_SUFFIX,
                            ""
                        )
                    ),

                    # Nome completo della colonna.
                    "narrative_column":
                        narrative_col,

                    # Emozione GoEmotions.
                    "emotion":
                        emotion_col,

                    # Correlazione.
                    "spearman_rho":
                        rho,

                    # p-value grezzo.
                    "p_value":
                        p,
                })

    result = pd.DataFrame(
        rows
    )

    # Colonna destinata ai q-value.
    result[
        "fdr_q_value"
    ] = np.nan

    # La correzione FDR viene effettuata
    # separatamente per ALL, FAKE e REAL.
    for scope in (
        result["scope"].unique()
    ):

        mask = (
            result["scope"]
            == scope
        )

        result.loc[
            mask,
            "fdr_q_value"
        ] = benjamini_hochberg(
            result.loc[
                mask,
                "p_value"
            ].to_numpy()
        )

    # Flag booleano:
    # True se significativo dopo FDR.
    result[
        "significant_fdr_0_05"
    ] = (
        result[
            "fdr_q_value"
        ] < 0.05
    )

    return result


# ============================================================
# DIFFERENZE FAKE VS REAL
# ============================================================

def compute_fake_real_differences(
    corr: pd.DataFrame,
) -> pd.DataFrame:
    """
    Confronta direttamente le correlazioni FAKE e REAL.

    Esempio:

        FAKE:
        conflict ↔ anger = 0.47

        REAL:
        conflict ↔ anger = 0.49

    Calcoliamo:

        delta_rho = rho_FAKE - rho_REAL

    e testiamo se la differenza è statisticamente significativa.
    """

    # Separa le correlazioni dei due gruppi.
    fake = corr[
        corr["scope"] == "FAKE"
    ].copy()

    real = corr[
        corr["scope"] == "REAL"
    ].copy()

    # Chiavi per riconoscere
    # la stessa coppia nei due gruppi.
    keys = [
        "narrative_dimension",
        "narrative_column",
        "emotion",
    ]

    # Merge FAKE / REAL.
    merged = fake.merge(
        real,
        on=keys,
        suffixes=(
            "_fake",
            "_real"
        ),
    )

    rows = []

    for _, row in merged.iterrows():

        # Test della differenza.
        z, p = fisher_difference_test(

            row[
                "spearman_rho_fake"
            ],

            int(
                row["n_fake"]
            ),

            row[
                "spearman_rho_real"
            ],

            int(
                row["n_real"]
            ),
        )

        rows.append({

            "narrative_dimension":
                row[
                    "narrative_dimension"
                ],

            "emotion":
                row["emotion"],

            "n_fake":
                int(
                    row["n_fake"]
                ),

            "n_real":
                int(
                    row["n_real"]
                ),

            # Correlazione nei fake.
            "rho_fake":
                row[
                    "spearman_rho_fake"
                ],

            # Correlazione nei real.
            "rho_real":
                row[
                    "spearman_rho_real"
                ],

            # Differenza diretta.
            #
            # positivo:
            # relazione più forte nei FAKE.
            #
            # negativo:
            # relazione più forte nei REAL.
            "delta_rho_fake_minus_real": (
                row[
                    "spearman_rho_fake"
                ]
                -
                row[
                    "spearman_rho_real"
                ]
            ),

            "fisher_z":
                z,

            "p_value_difference":
                p,
        })

    result = pd.DataFrame(
        rows
    )

    # Correzione FDR anche sui test
    # di differenza FAKE/REAL.
    result[
        "fdr_q_value_difference"
    ] = benjamini_hochberg(
        result[
            "p_value_difference"
        ].to_numpy()
    )

    result[
        "significant_difference_fdr_0_05"
    ] = (
        result[
            "fdr_q_value_difference"
        ] < 0.05
    )

    return result


# ============================================================
# HEATMAP DELLE CORRELAZIONI
# ============================================================

def make_heatmap(
    corr: pd.DataFrame,
    scope: str,
    output_path: Path,
):
    """
    Genera una heatmap:

        righe   = emozioni
        colonne = dimensioni NarraBERT
        colore  = Spearman rho
    """

    # Seleziona il gruppo richiesto.
    subset = corr[
        corr["scope"] == scope
    ]

    # Crea matrice emozione × dimensione.
    matrix = subset.pivot(
        index="emotion",
        columns="narrative_dimension",
        values="spearman_rho",
    )

    # Mantiene sempre lo stesso ordine
    # delle 9 dimensioni.
    matrix = matrix[
        [
            d
            for d in NARRATIVE_DIMENSIONS
            if d in matrix.columns
        ]
    ]

    # Dimensione della figura.
    fig, ax = plt.subplots(
        figsize=(
            15,
            max(
                9,
                len(matrix) * 0.38
            )
        )
    )

    # Visualizzazione della matrice.
    #
    # vmin/vmax fissi consentono di confrontare
    # direttamente le heatmap ALL/FAKE/REAL.
    im = ax.imshow(
        matrix.values,
        aspect="auto",
        vmin=-0.6,
        vmax=0.6,
        cmap="RdBu_r",
    )

    # Etichette asse X.
    ax.set_xticks(
        range(
            len(matrix.columns)
        )
    )

    ax.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right",
    )

    # Etichette asse Y.
    ax.set_yticks(
        range(
            len(matrix.index)
        )
    )

    ax.set_yticklabels(
        matrix.index
    )

    ax.set_title(
        f"RQ1 — NarraBERT vs GoEmotions ({scope})\n"
        "Spearman correlation"
    )

    ax.set_xlabel(
        "NarraBERT dimension"
    )

    ax.set_ylabel(
        "Emotion"
    )

    # Barra dei colori.
    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        "Spearman rho"
    )

    fig.tight_layout()

    # Salva il grafico.
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    # Libera memoria.
    plt.close(
        fig
    )


# ============================================================
# HEATMAP DELLE DIFFERENZE
# ============================================================

def make_difference_heatmap(
    differences: pd.DataFrame,
    output_path: Path,
):
    """
    Genera la heatmap:

        delta rho = rho_FAKE - rho_REAL

    positivo -> relazione più forte nei fake
    negativo -> relazione più forte nei real
    """

    matrix = differences.pivot(
        index="emotion",
        columns="narrative_dimension",
        values="delta_rho_fake_minus_real",
    )

    # Ordine fisso delle dimensioni.
    matrix = matrix[
        [
            d
            for d in NARRATIVE_DIMENSIONS
            if d in matrix.columns
        ]
    ]

    fig, ax = plt.subplots(
        figsize=(
            15,
            max(
                9,
                len(matrix) * 0.38
            )
        )
    )

    im = ax.imshow(
        matrix.values,
        aspect="auto",
        vmin=-0.6,
        vmax=0.6,
        cmap="RdBu_r",
    )

    ax.set_xticks(
        range(
            len(matrix.columns)
        )
    )

    ax.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        range(
            len(matrix.index)
        )
    )

    ax.set_yticklabels(
        matrix.index
    )

    ax.set_title(
        "RQ1 — Difference between FAKE and REAL correlations\n"
        "delta rho = rho_FAKE - rho_REAL"
    )

    ax.set_xlabel(
        "NarraBERT dimension"
    )

    ax.set_ylabel(
        "Emotion"
    )

    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        "Delta Spearman rho"
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
# BAR CHART RIASSUNTIVI NARRABERT × GOEMOTIONS
# ============================================================

def make_top_associations_barplot(
    corr: pd.DataFrame,
    scope: str,
    output_path: Path,
    top_n: int = 15,
):
    """Mostra le associazioni NarraBERT × emozione più forti per un gruppo."""
    subset = corr[corr["scope"] == scope].copy()
    subset["abs_rho"] = subset["spearman_rho"].abs()
    subset = subset.sort_values("abs_rho", ascending=False).head(top_n).copy()
    subset["pair"] = subset["narrative_dimension"] + " × " + subset["emotion"]
    subset = subset.sort_values("spearman_rho")

    fig, ax = plt.subplots(figsize=(12, max(7, top_n * 0.45)))
    ax.barh(subset["pair"], subset["spearman_rho"])
    ax.axvline(0, linewidth=1)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman correlation")
    ax.set_ylabel("NarraBERT dimension × emotion")
    ax.set_title(f"RQ1 NarraBERT — Top associations ({scope})")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fake_real_comparison_barplot(
    differences: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
):
    """Confronta rho FAKE e rho REAL per le coppie con differenza maggiore."""
    import numpy as np

    subset = differences.copy()
    subset["abs_delta"] = subset["delta_rho_fake_minus_real"].abs()
    subset = subset.sort_values("abs_delta", ascending=False).head(top_n).copy()
    subset["pair"] = subset["narrative_dimension"] + " × " + subset["emotion"]
    subset = subset.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(subset))
    height = 0.36
    fig, ax = plt.subplots(figsize=(13, max(8, top_n * 0.5)))
    ax.barh(y - height / 2, subset["rho_fake"], height=height, label="FAKE")
    ax.barh(y + height / 2, subset["rho_real"], height=height, label="REAL")
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(subset["pair"])
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman correlation")
    ax.set_ylabel("NarraBERT dimension × emotion")
    ax.set_title("RQ1 NarraBERT — FAKE vs REAL: largest correlation differences")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================

def main():
    """
    Pipeline completa dell'estensione NarraBERT della RQ1.

    1. carica dataset NarraBERT;
    2. carica dataset GoEmotions;
    3. normalizza le chiavi;
    4. unisce i dataset;
    5. calcola tutte le correlazioni;
    6. applica FDR;
    7. confronta FAKE e REAL;
    8. genera heatmap;
    9. salva risultati.
    """

    print("=" * 76)
    print(
        "RQ1 — NARRABERT x GOEMOTIONS"
    )
    print("=" * 76)

    # Controllo dei file.
    if not NARRATIVITY_CSV.exists():
        raise FileNotFoundError(
            f"File non trovato:\n"
            f"{NARRATIVITY_CSV}"
        )

    if not EMOTIONS_CSV.exists():
        raise FileNotFoundError(
            f"File non trovato:\n"
            f"{EMOTIONS_CSV}"
        )

    print(
        f"NarraBERT: "
        f"{NARRATIVITY_CSV}"
    )

    print(
        f"Emotions:  "
        f"{EMOTIONS_CSV}"
    )

    print()

    # Caricamento dei due dataset.
    narr = pd.read_csv(
        NARRATIVITY_CSV
    )

    emo = pd.read_csv(
        EMOTIONS_CSV
    )

    print(
        f"NarraBERT rows: "
        f"{len(narr):,}"
    )

    print(
        f"Emotions rows:  "
        f"{len(emo):,}"
    )

    # Crea le chiavi normalizzate.
    narr = prepare_merge_keys(
        narr
    )

    emo = prepare_merge_keys(
        emo
    )

    # Costruisce i nomi delle colonne NarraBERT
    # che vogliamo analizzare.
    #
    # Esempio:
    #
    # narrabert_conflict_mean
    narrative_cols = [
        f"narrabert_{dim}{NARRATIVE_SUFFIX}"
        for dim in NARRATIVE_DIMENSIONS
    ]

    # Verifica che tutte le dimensioni
    # siano effettivamente presenti.
    missing_narrative = [
        c
        for c in narrative_cols
        if c not in narr.columns
    ]

    if missing_narrative:
        raise ValueError(
            "Colonne NarraBERT mancanti:\n"
            + "\n".join(
                missing_narrative
            )
        )

    # Individua le colonne GoEmotions.
    emotion_cols = detect_emotion_columns(
        emo
    )

    if not emotion_cols:
        raise ValueError(
            "Non sono riuscito a rilevare "
            "le colonne GoEmotions."
        )

    print(
        f"Emozioni rilevate: "
        f"{len(emotion_cols)}"
    )

    print(
        ", ".join(
            emotion_cols
        )
    )

    print()

    # Chiavi tecniche create
    # da prepare_merge_keys().
    merge_key_cols = [
        "_title_key",
        "_label_key",
        "_subject_key",
        "_date_key",
    ]

    # Se nel dataset GoEmotions
    # lo stesso articolo compare più volte,
    # media le probabilità emotive.
    emo_reduced = (
        emo[
            merge_key_cols
            + emotion_cols
        ]
        .groupby(
            merge_key_cols,
            as_index=False,
        )[emotion_cols]
        .mean()
    )

    # Merge NarraBERT + GoEmotions.
    merged = narr.merge(
        emo_reduced,
        on=merge_key_cols,
        how="left",

        # Ogni articolo NarraBERT deve trovare
        # al massimo una riga emotiva.
        validate="many_to_one",
    )

    # Conta gli articoli che hanno
    # almeno un valore emotivo valido.
    matched = (
        merged[
            emotion_cols
        ]
        .notna()
        .any(axis=1)
        .sum()
    )

    missing = (
        len(merged)
        - matched
    )

    print(
        f"Merge completato: "
        f"{matched:,}/{len(merged):,}"
    )

    print(
        f"Articoli senza emozioni: "
        f"{missing:,}"
    )

    # Se il merge non ha funzionato per niente,
    # interrompe l'esecuzione.
    if matched == 0:
        raise RuntimeError(
            "Il merge non ha trovato "
            "alcuna corrispondenza."
        )

    # Mantiene solo gli articoli
    # con valori emotivi disponibili.
    merged = merged[
        merged[
            emotion_cols
        ]
        .notna()
        .any(axis=1)
    ].copy()

    # Crea cartella output.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Colonne tecniche usate solo per il merge.
    technical_keys = [
        "_title_key",
        "_label_key",
        "_subject_key",
        "_date_key",
    ]

    # Le rimuoviamo dal dataset finale.
    merged.drop(
        columns=[
            c
            for c in technical_keys
            if c in merged.columns
        ],
        inplace=True,
    )

    # Salva dataset NarraBERT + GoEmotions.
    merged.to_csv(
        MERGED_CSV,
        index=False,
        encoding="utf-8",
    )

    print(
        "\nCalcolo correlazioni Spearman..."
    )

    # Calcola tutte le correlazioni.
    corr = compute_correlations(
        merged,
        narrative_cols,
        emotion_cols,
    )

    # Salva tabella delle correlazioni.
    corr.to_csv(
        CORRELATIONS_CSV,
        index=False,
        encoding="utf-8",
    )

    # Calcola differenze FAKE / REAL.
    differences = (
        compute_fake_real_differences(
            corr
        )
    )

    # Salva differenze.
    differences.to_csv(
        DIFFERENCES_CSV,
        index=False,
        encoding="utf-8",
    )

    print(
        "Generazione heatmap..."
    )

    # Heatmap su tutti gli articoli.
    make_heatmap(
        corr,
        "ALL",
        OUTPUT_DIR
        / "rq1_narrabert_all.png",
    )

    # Heatmap FAKE.
    make_heatmap(
        corr,
        "FAKE",
        OUTPUT_DIR
        / "rq1_narrabert_fake.png",
    )

    # Heatmap REAL.
    make_heatmap(
        corr,
        "REAL",
        OUTPUT_DIR
        / "rq1_narrabert_real.png",
    )

    # Heatmap differenze FAKE/REAL.
    make_difference_heatmap(
        differences,
        OUTPUT_DIR
        / "rq1_fake_real_difference.png",
    )


    # Bar chart aggiuntivi per una lettura più immediata dei risultati.
    make_top_associations_barplot(
        corr,
        "FAKE",
        OUTPUT_DIR / "rq1_narrabert_top15_fake.png",
    )
    make_top_associations_barplot(
        corr,
        "REAL",
        OUTPUT_DIR / "rq1_narrabert_top15_real.png",
    )
    make_fake_real_comparison_barplot(
        differences,
        OUTPUT_DIR / "rq1_narrabert_fake_real_top_differences.png",
    )

    # ========================================================
    # RISULTATI RIASSUNTIVI
    # ========================================================

    print()

    print("=" * 76)
    print(
        "RISULTATI PRINCIPALI"
    )
    print("=" * 76)

    # Stampa le associazioni più forti
    # separatamente per FAKE e REAL.
    for scope in [
        "FAKE",
        "REAL"
    ]:

        subset = corr[
            corr["scope"] == scope
        ].copy()

        # Valore assoluto della correlazione:
        # serve solo per ordinare
        # dalla relazione più forte.
        subset[
            "abs_rho"
        ] = (
            subset[
                "spearman_rho"
            ]
            .abs()
        )

        # Prende le prime 15.
        strongest = (
            subset
            .sort_values(
                "abs_rho",
                ascending=False,
            )
            .head(15)
        )

        print(
            f"\nTOP 15 associazioni — "
            f"{scope}"
        )

        print(
            strongest[
                [
                    "narrative_dimension",
                    "emotion",
                    "spearman_rho",
                    "fdr_q_value",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )

    # ========================================================
    # DIFFERENZE PIÙ FORTI
    # ========================================================

    print(
        "\nTOP 15 differenze "
        "FAKE vs REAL"
    )

    strongest_diff = (
        differences.copy()
    )

    # Valore assoluto della differenza.
    strongest_diff[
        "abs_delta"
    ] = (
        strongest_diff[
            "delta_rho_fake_minus_real"
        ]
        .abs()
    )

    # Ordina e prende le prime 15.
    strongest_diff = (
        strongest_diff
        .sort_values(
            "abs_delta",
            ascending=False,
        )
        .head(15)
    )

    print(
        strongest_diff[
            [
                "narrative_dimension",
                "emotion",
                "rho_fake",
                "rho_real",
                "delta_rho_fake_minus_real",
                "fdr_q_value_difference",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()

    print("=" * 76)
    print("OUTPUT")
    print("=" * 76)

    print(
        f"- {MERGED_CSV}"
    )

    print(
        f"- {CORRELATIONS_CSV}"
    )

    print(
        f"- {DIFFERENCES_CSV}"
    )

    print(
        f"- "
        f"{OUTPUT_DIR / 'rq1_narrabert_all.png'}"
    )

    print(
        f"- "
        f"{OUTPUT_DIR / 'rq1_narrabert_fake.png'}"
    )

    print(
        f"- "
        f"{OUTPUT_DIR / 'rq1_narrabert_real.png'}"
    )

    print(
        f"- "
        f"{OUTPUT_DIR / 'rq1_fake_real_difference.png'}"
    )

    print(
        "\nNota metodologica: "
        "questa analisi usa le colonne '*_mean', "
        "cioè la media della dimensione NarraBERT "
        "sui chunk dell'intero articolo. "
        "I p-value vengono corretti con "
        "Benjamini-Hochberg (FDR)."
    )


if __name__ == "__main__":
    main()