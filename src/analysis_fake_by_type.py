# src/analysis_fake_by_type.py
# task 2 - studiare le emozioni espresse dalle singole tipologie di fake news

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
    path = os.path.join(config.PROCESSED_DATA_DIR, "dataset_with_emotions.csv")
    return pd.read_csv(path)

def pick_type_column(df):
    # Identifica la colonna type e la restituisce
    for cand in ["subject", "category", "type"]:
        if cand in df.columns:
            return cand
    raise ValueError(
        "Non trovo una colonna di tipologia (subject/category/type). "
        "Aggiungine una o rinomina quella esistente."
    )

def summarize_emotions_by_type(df_fake, type_col):
    # Seleziona solo colonne delle emozioni + colonna tipologia
    cols = config.GO_EMOTION_LABELS + [type_col]
    df_fake = df_fake[cols].copy()

    # Calcola la media delle emozioni per ogni tipologia di fake news
    # Restituisce una tabella le cui righe sono formate dalle tiplogie di fake news e le cui colonne sono tutte le medie delle emozioni.
    mean_by_type = df_fake.groupby(type_col).mean().sort_index()
    return mean_by_type

def plot_heatmap(mean_by_type, fname="emotions_by_fake_type_heatmap.png"):
    plt.figure(figsize=(min(18, 2 + 0.5*len(mean_by_type)), 10))
    # Heatmap: righe = emozioni, colonne = tipologie
    sns.heatmap(
        mean_by_type.T, annot=False, cmap="coolwarm", fmt=".3f",
        cbar_kws={"label": "Media probabilità emozione"}
    )
    plt.title("Media delle emozioni per tipologia di fake news")
    plt.ylabel("Emozioni")
    plt.xlabel("Tipologia")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(out, dpi=200)
    plt.close()
    return out

def plot_top_emotions_per_type(mean_by_type, top_k=5, fname="top_emotions_per_type.png"):
    # Per ogni tipologia di fake news, seleziona le top-k emozioni con media più alta
    records = []
    for t, row in mean_by_type.iterrows():
        top = row.sort_values(ascending=False).head(top_k)
        for emo, val in top.items():
            records.append({"type": t, "emotion": emo, "mean": val})

    df_top = pd.DataFrame(records)

    # Grafico a barre orizzontali: y = tipologia, colore = emozione
    plt.figure(figsize=(12, max(6, 0.7*df_top["type"].nunique())))
    sns.barplot(
        data=df_top, x="mean", y="type", hue="emotion", orient="h"
    )
    plt.xlabel("Media probabilità emozione")
    plt.ylabel("Tipologia")
    plt.title(f"Top {top_k} emozioni per tipologia di fake news")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(out, dpi=200)
    plt.close()
    return out

def main():
    print("📥 Carico dataset con emozioni…")
    df = load_dataset()

    # Filtra solo le fake news (etichetta = 1)
    print("🔎 Filtro solo FAKE…")
    df_fake = df[df[config.LABEL_COLUMN] == 1].copy()

    # Individua colonna che descrive la tipologia di fake news
    type_col = pick_type_column(df_fake)
    print(f"🏷️  Colonna tipologia rilevata: '{type_col}'")

    # Calcola la media delle emozioni per tipologia
    print("📊 Calcolo medie emozioni per tipologia…")
    mean_by_type = summarize_emotions_by_type(df_fake, type_col)

    # Salva i valori medi in un CSV per analisi testuale
    out_csv = os.path.join(config.PROCESSED_DATA_DIR, "emotions_by_fake_type.csv")
    mean_by_type.to_csv(out_csv)
    print(f"💾 Salvato riepilogo: {out_csv}")

    # Genera e salva grafici
    heatmap_path = plot_heatmap(mean_by_type)
    top_path = plot_top_emotions_per_type(mean_by_type, top_k=5)

    # Messaggi di log con i percorsi
    print(f"🖼️ Heatmap salvata: {heatmap_path}")
    print(f"🖼️ Top emozioni per tipologia salvato: {top_path}")
    print("✅ Analisi per tipologie completata.")
    return {"heatmap": heatmap_path, "top": top_path}

if __name__ == "__main__":
    main()
