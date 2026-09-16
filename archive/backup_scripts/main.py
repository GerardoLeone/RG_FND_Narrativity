import os
import argparse
import pandas as pd

from src import config
from src.preprocess import preprocess_dataframe
from src.emotions.emotion_inference import EmotionDetector
from src.load_data import load_and_merge_dataset


PROCESSED_FILENAME = "dataset_with_emotions.csv"


def ensure_dataset_with_emotions(force: bool = False):

    os.makedirs(
        config.PROCESSED_DATA_DIR,
        exist_ok=True
    )

    processed_path = os.path.join(
        config.PROCESSED_DATA_DIR,
        PROCESSED_FILENAME
    )

    if os.path.exists(processed_path) and not force:

        print(
            f"📄 Trovato dataset con emozioni: "
            f"{processed_path}"
        )

        df = pd.read_csv(
            processed_path
        )

        print(
            f"ℹ️ Righe nel CSV processato: "
            f"{len(df)}"
        )

        return df

    print(
        "🔹 Rigenero il dataset con emozioni..."
    )

    print(
        "📥 Caricamento True.csv + Fake.csv..."
    )

    df = load_and_merge_dataset()

    print(
        f"✔ Dataset unificato: "
        f"{df.shape[0]} righe"
    )

    print(
        "🔎 Verifica righe per label "
        "prima della pulizia:"
    )

    print(
        df["label"]
        .value_counts(dropna=False)
        .rename({
            0: "REAL",
            1: "FAKE"
        })
    )

    print(
        "🧹 Pulizia testo..."
    )

    df = preprocess_dataframe(
        df,
        text_col=config.TEXT_COLUMN
    )

    print(
        "🤖 Inferenza emozioni..."
    )

    detector = EmotionDetector()

    df_with_emotions = (
        detector.predict_emotions_df(
            df,
            text_col=config.TEXT_COLUMN
        )
    )

    df_with_emotions.to_csv(
        processed_path,
        index=False
    )

    print(
        f"💾 Salvato: "
        f"{processed_path} "
        f"({len(df_with_emotions)} righe)"
    )

    return df_with_emotions


def sanity_check_raw_paths():

    true_path = os.path.join(
        config.RAW_DATA_DIR,
        "True.csv"
    )

    fake_path = os.path.join(
        config.RAW_DATA_DIR,
        "Fake.csv"
    )

    print(
        f"📂 RAW True: "
        f"{os.path.abspath(true_path)} "
        f"| esiste={os.path.exists(true_path)}"
    )

    print(
        f"📂 RAW Fake: "
        f"{os.path.abspath(fake_path)} "
        f"| esiste={os.path.exists(fake_path)}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rigenera dataset_with_emotions.csv "
            "rifacendo l'inferenza."
        )
    )

    args = parser.parse_args()

    sanity_check_raw_paths()

    ensure_dataset_with_emotions(
        force=args.force
    )


if __name__ == "__main__":
    main()