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


def precision(predictions, labels, target):
    true_positive = 0
    predicted_positive = 0

    for p, l in zip(predictions, labels):
        if p == target:
            predicted_positive += 1
            if p == l:
                true_positive += 1

    if predicted_positive == 0:
        return 0

    return true_positive / predicted_positive


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
    prec_neg = precision(predictions, labels, "negativ")

    print("Accuracy:", acc)
    print("Precision (negativ):", prec_neg)

    print("\nDetalii:")
    for (text, _), pred, label in zip(test_cases, predictions, labels):
        print(f"Input: {text} | Predicted: {pred} | Expected: {label}")


if __name__ == "__main__":
    run_evaluation()
