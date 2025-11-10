# src/preprocess.py

import re
import pandas as pd

def clean_text(text: str) -> str:
    """
    Pulisce un testo rimuovendo URL, menzioni, caratteri speciali e extra spazi.
    """
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URL
    text = re.sub(r"\@\w+|\#", "", text)                 # @menzioni e #
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)           # caratteri speciali
    text = re.sub(r"\s+", " ", text).strip()             # spazi multipli
    return text

def preprocess_dataframe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Applica clean_text a una colonna di un DataFrame.
    """
    df[text_col] = df[text_col].apply(clean_text)
    return df
