# main.py

import os
import argparse
import pandas as pd
from src import config
from src.preprocess import preprocess_dataframe
from src.emotion_inference import EmotionDetector
from src.train_fnd_model import train_fnd_model
from src.load_data import load_and_merge_dataset

from src.analysis_real_vs_fake import analyze_emotions_real_vs_fake
from src.analysis_fake_by_type import main as analyze_fake_by_type

PROCESSED_FILENAME = "dataset_with_emotions.csv"

def ensure_dataset_with_emotions(force: bool=False):
    """
    Se esiste il CSV processato e force=False, lo ricarica.
    Altrimenti esegue merge+preprocess+inferenza emozioni e lo rigenera.
    """
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    processed_path = os.path.join(config.PROCESSED_DATA_DIR, PROCESSED_FILENAME)

    if os.path.exists(processed_path) and not force:
        print(f"📄 Trovato dataset con emozioni: {processed_path}")
        df = pd.read_csv(processed_path)
        print(f"ℹ️  Righe nel CSV processato: {len(df)}")
        return df

    print("🔹 Rigenero il dataset con emozioni (force o assente)…")
    print("📥 Caricamento True.csv + Fake.csv…")
    df = load_and_merge_dataset()
    print(f"✔ Dataset unificato: {df.shape[0]} righe")

    # Log diagnostico utile
    print("🔎 Verifica righe per label prima della pulizia:")
    print(df["label"].value_counts(dropna=False).rename({0:"REAL",1:"FAKE"}))

    print("🧹 Pulizia testo…")
    df = preprocess_dataframe(df, text_col=config.TEXT_COLUMN)

    print("🤖 Inference emozioni…")
    detector = EmotionDetector()
    df_with_emotions = detector.predict_emotions_df(df, text_col=config.TEXT_COLUMN)

    df_with_emotions.to_csv(processed_path, index=False)
    print(f"💾 Salvato: {processed_path} ({len(df_with_emotions)} righe)")

    return df_with_emotions

def sanity_check_raw_paths():
    """Stampa i path effettivi dei file raw per evitare di usare copie sbagliate."""
    from src import config
    import os
    true_path = os.path.join(config.RAW_DATA_DIR, "True.csv")
    fake_path = os.path.join(config.RAW_DATA_DIR, "Fake.csv")
    print(f"📂 RAW True: {os.path.abspath(true_path)}  | esiste={os.path.exists(true_path)}")
    print(f"📂 RAW Fake: {os.path.abspath(fake_path)}  | esiste={os.path.exists(fake_path)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Rigenera dataset_with_emotions.csv rifacendo l'inferenza.")
    args = parser.parse_args()

    sanity_check_raw_paths()

    # Assicura il dataset con emozioni (crea o carica)
    df_with_emotions = ensure_dataset_with_emotions(force=args.force)

    # —— TASK 1: REAL vs FAKE ——
    print("📊 [Task 1] Analisi emozioni REAL vs FAKE…")
    analyze_emotions_real_vs_fake()
    print("✅ [Task 1] Completato.")

    # —— TASK 2: Tipologie di FAKE ——
    print("📊 [Task 2] Analisi emozioni per tipologia di fake news…")
    analyze_fake_by_type()
    print("✅ [Task 2] Completato.")

    # —— TASK 3: Training FND ——
    print("🎯 [Task 3] Training classificatore FND (solo emozioni)…")
    train_fnd_model(df_with_emotions, model_type="logreg")
    print("✅ [Task 3] Completato.")

    print("🏁 Tutti i task completati.")

if __name__ == "__main__":
    main()
