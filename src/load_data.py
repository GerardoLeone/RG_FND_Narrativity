# src/load_data.py

import pandas as pd
import os
from src import config

def load_and_merge_dataset():
    """
    Carica True.csv e Fake.csv, assegna le etichette binarie, concatena e mescola.
    """
    true_path = os.path.join(config.RAW_DATA_DIR, "True.csv")
    fake_path = os.path.join(config.RAW_DATA_DIR, "Fake.csv")

    df_true = pd.read_csv(true_path)
    df_true["label"] = 0

    df_fake = pd.read_csv(fake_path)
    df_fake["label"] = 1

    df = pd.concat([df_true, df_fake], ignore_index=True)
    df = df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    # sample fa lo shuffle delle righe

    # Combina title e text in un'unica colonna
    df["text"] = df["title"].fillna("") + ". " + df["text"].fillna("")

    return df
