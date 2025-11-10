# src/emotion_inference.py
# Modulo per fare inferenza delle emozioni sui testi con il modello GoEmotions (SamLowe/roberta-base-go_emotions)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from src.config import GO_EMOTION_LABELS, MAX_TOKENS

class EmotionDetector:
    def __init__(self, model_name="SamLowe/roberta-base-go_emotions", device=None):
        # 🔹 Inizializzazione del detector: carica tokenizer e modello
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)                  # converte testo → token numerici
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name) # carica il modello pre-addestrato per classificazione multilabel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")    # sceglie GPU se disponibile, altrimenti CPU
        self.model.to(self.device)                                                  # sposta il modello sul device scelto
        self.model.eval()                                                           # modalità "evaluation": niente training, solo inferenza
        print(f"🖥️  Utilizzo dispositivo: {self.device}")

    def predict_emotions(self, texts, batch_size=16):
        """
        Esegue inferenza batch-wise su una lista di testi.
        Restituisce una lista di dizionari {emozione: probabilità}.
        """
        results = []
        total = len(texts)
        n_batches = (total + batch_size - 1) // batch_size   # calcolo numero batch (ceil division)

        # Log iniziale
        print(f"Totale testi: {total} | batch_size: {batch_size} | n_batch: {n_batches}")

        # tqdm: barra di avanzamento
        with tqdm(total=total, desc="Inferenza emozioni", unit="text") as pbar:
            for i in range(0, total, batch_size):
                batch_texts = texts[i:i + batch_size]     # prende un blocco di testi

                # Tokenizzazione: padding e truncation per avere batch uniforme
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,                         # aggiunge [PAD] per lunghezze diverse
                    truncation=True,                      # tronca se troppo lungo
                    max_length=MAX_TOKENS,                # lunghezza massima (config)
                    return_tensors="pt"                   # output come tensori PyTorch
                ).to(self.device)                         # sposta i tensori su CPU/GPU

                # Inferenza senza gradiente (più veloce, meno memoria)
                with torch.no_grad():
                    outputs = self.model(**inputs)        # passa il batch nel modello
                    probs = F.sigmoid(outputs.logits).cpu().numpy()
                    # sigmoid → converte i logit in probabilità (multi-label: ogni emozione può attivarsi indipendentemente)

                # Converte ogni vettore di probabilità in un dict {emozione: valore}
                for prob_vector in probs:
                    results.append(dict(zip(GO_EMOTION_LABELS, prob_vector)))

                # Aggiorna la barra di avanzamento
                pbar.update(len(batch_texts))

        return results

    def predict_emotions_df(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """
        Riceve un DataFrame con una colonna di testo, aggiunge colonne emozionali (una per emozione).
        """
        texts = df[text_col].tolist()                      # estrae la colonna dei testi in lista
        emotion_dicts = self.predict_emotions(texts)       # predice emozioni su tutti i testi
        emotion_df = pd.DataFrame(emotion_dicts)           # converte in DataFrame con colonne = emozioni
        # concatena il dataset originale con le colonne emozioni → unico DataFrame arricchito
        return pd.concat([df.reset_index(drop=True), emotion_df], axis=1)
