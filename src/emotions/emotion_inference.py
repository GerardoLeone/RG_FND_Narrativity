"""
Modulo: emotion_inference.py

Scopo:
capire quali emozioni sono presenti in ogni articolo. Usa il modello già addestrato
SamLowe/roberta-base-go_emotions.

Il modello è multilabel:
ogni emozione viene valutata indipendentemente dalle altre.
Per questo viene applicata una sigmoid ai logit del modello.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import pandas as pd

# Importiamo dal file config.py:
# - l'elenco delle etichette GoEmotions
# - il numero massimo di token da fornire al modello
from src.config import GO_EMOTION_LABELS, MAX_TOKENS


class EmotionDetector:
    """
    Classe che gestisce:
    - caricamento del tokenizer (serve a trasformare il testo in numeri leggibili dal modello);
    - caricamento del modello GoEmotions;
    - scelta CPU/GPU;
    - inferenza delle emozioni;
    - aggiunta delle probabilità emotive a un DataFrame.
    """

    def __init__(
        self,
        model_name="SamLowe/roberta-base-go_emotions",
        device=None
    ):
        """
        Inizializza il modello.

        Se non viene specificato manualmente un device,
        usa CUDA quando disponibile, altrimenti CPU.
        """

        # Il tokenizer trasforma il testo in token numerici
        # compatibili con RoBERTa.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Carica il modello già addestrato per la classificazione
        # multilabel delle emozioni GoEmotions.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )

        # Preferiamo la GPU, se disponibile.
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Sposta il modello sul device scelto.
        self.model.to(self.device)

        # Modalità evaluation:
        # il modello viene usato solo per inferenza, non per training.
        self.model.eval()

        print(f"Utilizzo dispositivo: {self.device}")

    def predict_emotions(self, texts, batch_size=16):
        """
        Calcola le probabilità emotive per una lista di testi.

        Parameters
        ----------
        texts : list[str]
            Lista dei testi da analizzare.

        batch_size : int
            Numero di testi elaborati contemporaneamente.

        Returns
        -------
        list[dict]
            Per ogni testo restituisce un dizionario del tipo:

            {
                "anger": 0.21,
                "sadness": 0.13,
                ...
            }
        """

        results = []

        # Numero totale di testi da processare.
        total = len(texts)

        # Calcola quanti batch saranno necessari.
        # È una divisione arrotondata verso l'alto.
        n_batches = (total + batch_size - 1) // batch_size

        print(
            f"Totale testi: {total} | "
            f"batch_size: {batch_size} | "
            f"n_batch: {n_batches}"
        )

        # Barra di avanzamento.
        with tqdm(
            total=total,
            desc="Inferenza emozioni",
            unit="text"
        ) as pbar:

            # Analizziamo i testi a batch invece di elaborarli
            # tutti contemporaneamente.
            for i in range(0, total, batch_size):

                batch_texts = texts[i:i + batch_size]

                # Tokenizzazione del batch.
                inputs = self.tokenizer(
                    batch_texts,

                    # Aggiunge padding ai testi più corti,
                    # così tutti i tensori del batch hanno
                    # la stessa lunghezza.
                    padding=True,

                    # Se un testo supera MAX_TOKENS viene troncato.
                    truncation=True,

                    max_length=MAX_TOKENS,

                    # Restituisce tensori PyTorch.
                    return_tensors="pt"

                ).to(self.device)

                with torch.no_grad():

                    # Passaggio del batch nel modello.
                    outputs = self.model(**inputs)

                    # outputs.logits contiene un valore grezzo
                    # per ogni emozione.

                    # Applichiamo sigmoid perché GoEmotions è multilabel:
                    # ogni emozione viene valutata indipendentemente tra 0 e 1.
                    #
                    # Una stessa frase può quindi avere,
                    # per esempio:
                    # anger = 0.60
                    # sadness = 0.45
                    # fear = 0.30
                    probs = torch.sigmoid(
                        outputs.logits
                    ).cpu().numpy()

                # Trasformiamo ogni vettore di probabilità
                # in un dizionario:
                #
                # emozione -> probabilità
                for prob_vector in probs:

                    results.append(
                        dict(
                            zip(
                                GO_EMOTION_LABELS,
                                prob_vector
                            )
                        )
                    )

                # Aggiorna la barra di avanzamento.
                pbar.update(len(batch_texts))

        return results

    def predict_emotions_df(
        self,
        df: pd.DataFrame,
        text_col: str
    ) -> pd.DataFrame:
        """
        Arricchisce ogni articolo con una colonna per ciascuna emozione, questo dataset viene poi usato nelle analisi successive.
        title | text | label + "| anger | joy | fear | sadness | ..."
        Applica GoEmotions direttamente a un DataFrame.

        Il risultato contiene il dataset originale più una
        nuova colonna per ogni emozione.
        """

        # Estrae la colonna testuale come lista.
        texts = df[text_col].tolist()

        # Calcola le emozioni per tutti i testi.
        emotion_dicts = self.predict_emotions(texts)

        # Converte la lista di dizionari in DataFrame.
        #
        # Ogni riga = un articolo.
        # Ogni colonna = un'emozione.
        emotion_df = pd.DataFrame(emotion_dicts)

        # Unisce il dataset originale con le nuove colonne emotive.
        #
        # reset_index evita problemi se df aveva indici non consecutivi.
        return pd.concat(
            [
                df.reset_index(drop=True),
                emotion_df
            ],
            axis=1
        )