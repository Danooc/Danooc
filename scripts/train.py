#!/usr/bin/env python3
"""
Train the intent classifier from labeled examples.

Usage:
    python scripts/train.py
    python scripts/train.py --data data/training_data.json --output data/intent_model.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from danooc.nlp.classifier import IntentClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train intent classifier")
    parser.add_argument("--data", default="data/training_data.json", help="Path to training data JSON")
    parser.add_argument("--output", default="data/intent_model.json", help="Path to save model")
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        examples = json.load(f)

    print(f"Training data: {len(examples)} examples")

    clf = IntentClassifier()
    metrics = clf.train(examples)

    print(f"Intents:        {metrics['num_intents']}")
    print(f"Vocabulary:     {metrics['vocab_size']} words")
    print(f"Train accuracy: {metrics['train_accuracy']}%")

    clf.save(args.output)
    print(f"Model saved to: {args.output}")

    print("\n--- Test predictions ---")
    test_phrases = [
        "quiero comprar algo",
        "mi internet no funciona",
        "necesito mi factura",
        "quiero cancelar todo",
        "a qué hora abren",
        "pasarme con alguien",
    ]
    for phrase in test_phrases:
        intent, conf = clf.predict(phrase)
        print(f"  '{phrase}' → {intent} ({conf:.2%})")


if __name__ == "__main__":
    main()
