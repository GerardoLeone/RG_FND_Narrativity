import torch
from transformers import pipeline


MODEL_NAME = "mariaantoniak/storyseeker"
NARRATIVE_LABEL = "LABEL_1"  # da verificare con l'output del modello


def load_narrativity_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU CUDA non disponibile. Verifica di avere installato PyTorch con supporto CUDA."
        )

    print(f"GPU rilevata: {torch.cuda.get_device_name(0)}")
    print("Caricamento del modello StorySeeker...")

    classifier = pipeline(
        task="text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=0
    )

    print("Modello caricato correttamente.")
    print("Mappatura delle label:", classifier.model.config.id2label)

    return classifier


def analyze_text(classifier, text: str) -> list[dict]:
    if not text or not text.strip():
        raise ValueError("Il testo non può essere vuoto.")

    results = classifier(
        text,
        truncation=True,
        max_length=512,
        top_k=None
    )

    # Alcune versioni di transformers restituiscono una lista annidata.
    if results and isinstance(results[0], list):
        results = results[0]

    return results


def get_narrativity_score(results: list[dict]) -> float:
    for result in results:
        if result["label"] == NARRATIVE_LABEL:
            return float(result["score"])

    raise RuntimeError(
        f"Label narrativa {NARRATIVE_LABEL} non trovata. "
        "Controlla la mappatura delle label stampata dal modello."
    )


def main():
    classifier = load_narrativity_model()

    test_texts = [
        (
            "TESTO NARRATIVO",
            """
            When Sarah arrived at the hospital, she immediately noticed that
            something was wrong. The corridors were empty and the lights were
            flickering. She called her brother, but nobody answered. A few
            minutes later, she found a letter explaining what had happened.
            """
        ),
        (
            "TESTO INFORMATIVO",
            """
            The Department of Health published its monthly report on hospital
            admissions. According to the report, admissions increased by
            twelve percent compared with the previous month. The data were
            collected from public hospitals across the country.
            """
        )
    ]

    for name, text in test_texts:
        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        results = analyze_text(classifier, text)

        for result in results:
            print(f'{result["label"]}: {result["score"]:.4f}')

        narrativity_score = get_narrativity_score(results)
        print(f"Punteggio di narratività: {narrativity_score:.4f}")


if __name__ == "__main__":
    main()