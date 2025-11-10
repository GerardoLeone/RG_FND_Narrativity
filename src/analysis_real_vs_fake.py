# src/analysis_real_vs_fake.py
# task 1 - studiare le emozioni espresse da real e fake news

import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from src import config

OUTPUT_DIR = os.path.join(config.PROCESSED_DATA_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_dataset():
    # Path del CSV già arricchito con le colonne delle emozioni
    path = os.path.join(config.PROCESSED_DATA_DIR, "dataset_with_emotions.csv")
    return pd.read_csv(path)

def plot_emotion_heatmap(df_mean_emotions, fname="real_vs_fake_heatmap.png"):
    plt.figure(figsize=(14, 6))
    # Heatmap delle medie per emozione e classe.
    sns.heatmap(df_mean_emotions.T, annot=False, cmap="coolwarm", fmt=".3f", linewidths=0.5)
    plt.title("Media delle emozioni - REAL vs FAKE")
    plt.ylabel("Emozioni")
    plt.xlabel("Classe")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(out, dpi=200)
    plt.close()
    return out

def plot_emotion_difference(df_mean_emotions, fname="real_vs_fake_diff.png"):
    # Vettore differenza: media(FAKE) - media(REAL) per ciascuna emozione
    diff = df_mean_emotions.loc["FAKE"] - df_mean_emotions.loc["REAL"]

    # Ordina le emozioni dalla differenza più grande (positiva) alla più piccola (negativa)
    diff = diff.sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    # Barplot semplice delle differenze; niente palette esplicita per evitare warning futuri di seaborn
    sns.barplot(x=diff.index, y=diff.values)

    # Ruota le etichette delle emozioni per leggibilità
    plt.xticks(rotation=45, ha="right")

    plt.title("Differenza media emozioni: FAKE - REAL")     # Titolo
    plt.ylabel("Differenza media (FAKE - REAL)")            # Etichetta asse Y
    plt.tight_layout()                                      # Sistema il layout

    out = os.path.join(OUTPUT_DIR, fname)  # Percorso output
    plt.savefig(out, dpi=200)
    plt.close()
    return out

def analyze_emotions_real_vs_fake():
    df = load_dataset()  # Carica il DataFrame con le emozioni già calcolate a monte

    # Seleziona solo le colonne necessarie:
    # - config.GO_EMOTION_LABELS: lista delle colonne di punteggio/probabilità per ciascuna emozione
    # - config.LABEL_COLUMN: colonna con l’etichetta della classe (codificata 0/1 REAL/FAKE)
    cols = config.GO_EMOTION_LABELS + [config.LABEL_COLUMN]
    df = df[cols]  # Sotto-DataFrame con solo emozioni + etichetta classe

    # Raggruppa per classe (REAL vs FAKE) e calcola la media delle emozioni per ogni gruppo
    # Quindi restituisce una tabella con due righe 0 e 1, colonne con tutte le emozioni e le medie per ogni colonna
    df_mean = df.groupby(config.LABEL_COLUMN).mean()

    # Imposta indici testuali per le righe:
    df_mean.index = ["REAL", "FAKE"]

    # Crea e salva la heatmap delle medie
    heatmap_path = plot_emotion_heatmap(df_mean)

    # Crea e salva il barplot delle differenze (FAKE - REAL)
    diff_path = plot_emotion_difference(df_mean)

    # Messaggi di log in console con i percorsi dei file generati
    print(f"🖼️ Heatmap REAL vs FAKE salvata: {heatmap_path}")
    print(f"🖼️ Differenze FAKE-REAL salvate: {diff_path}")

    # Restituisce i percorsi per uso successivo (es. report o test)
    return {"heatmap": heatmap_path, "diff": diff_path}

if __name__ == "__main__":
    analyze_emotions_real_vs_fake()
