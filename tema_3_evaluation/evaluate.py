import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.service import model_function


def accuracy(predictions, labels):
    correct = 0
    for p, l in zip(predictions, labels):
        if p == l:
            correct += 1
    return correct / len(labels)


def run_evaluation():
    test_cases = [
        ("Îmi place produsul", "pozitiv"),
        ("Nu îmi place produsul", "negativ"),
        ("Este ok", "neutru"),
        ("Nu recomand", "negativ"),
    ]

    predictions = []
    labels = []

    for text, label in test_cases:
        result = model_function(text)
        predictions.append(result)
        labels.append(label)

    acc = accuracy(predictions, labels)

    print("Accuracy:", acc)


if __name__ == "__main__":
    run_evaluation()
