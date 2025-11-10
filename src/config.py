# src/config.py

import os

# Directory base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cartelle dati
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Parametri generali
RANDOM_SEED = 42
MAX_TOKENS = 512

# Colonne principali
TEXT_COLUMN = "text"      # useremo title + text
LABEL_COLUMN = "label"    # 0 = real, 1 = fake

# Etichette emozioni
GO_EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]
