# src/train_fnd_model.py
# task 3 - addestrare un modello per provare a fare FND usando le emozioni

import pandas as pd
from sklearn.model_selection import train_test_split        # split train/test con opzione stratify
from sklearn.metrics import classification_report, confusion_matrix  # metriche di valutazione
from sklearn.ensemble import RandomForestClassifier         # modello non lineare, robusto, non richiede scaling
from sklearn.linear_model import LogisticRegression         # modello lineare probabilistico
from sklearn.preprocessing import StandardScaler            # standardizzazione (mean=0, std=1), utile per logreg
from src.config import PROCESSED_DATA_DIR, GO_EMOTION_LABELS, LABEL_COLUMN, RANDOM_SEED
import os

def load_dataset(filename="dataset_with_emotions.csv"):
    # Costruisce il path al CSV processato e lo carica in un DataFrame
    path = os.path.join(PROCESSED_DATA_DIR, filename)
    return pd.read_csv(path)

def train_fnd_model(df, model_type="logreg"):
    # X = features (solo le colonne delle emozioni), y = target (label REAL/FAKE codificata 0/1)
    X = df[GO_EMOTION_LABELS]
    y = df[LABEL_COLUMN]

    # Suddivisione in train/test. Stratify=y preserva la proporzione classi (utile se il dataset è sbilanciato).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Standardizzazione: si adatta SOLO sul train (fit) e si applica a train e test (transform).
    # Serve per modelli sensibili alla scala (es. Logistic Regression). Non è necessario per Random Forest.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Scelta e addestramento del modello
    if model_type == "logreg":
        # Logistic Regression: modello lineare che stima P(FAKE|emozioni).
        # Qui il modello impara i pesi per ogni emozione, cercando la combinazione che meglio separa REAL da FAKE.
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        model.fit(X_train_scaled, y_train)  # allena sul train scalato
    elif model_type == "rf":
        # Random Forest: insieme di alberi decisionali; gestisce non-linearità e interazioni tra emozioni.
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)         # niente scaling qui
    else:
        raise ValueError(f"Modello '{model_type}' non supportato.")

    # Predizione sul test (rispetta lo stesso pre-processing usato in training)
    y_pred = model.predict(X_test_scaled if model_type == "logreg" else X_test)

    # Report dettagliato: precision, recall, f1 per classe e macro/weighted average
    print("\n📊 Report Classificazione:")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

    # Matrice di confusione: righe = vero, colonne = predetto
    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model

if __name__ == "__main__":
    df = load_dataset()
    print(f"✔ Dataset caricato: {df.shape[0]} righe")
    train_fnd_model(df, model_type="logreg")  # default: usa la logistic
