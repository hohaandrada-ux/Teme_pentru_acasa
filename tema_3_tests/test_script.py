import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.service import model_function


def test_positive():
    assert model_function("Îmi place produsul") == "pozitiv"


def test_negative():
    assert model_function("Nu îmi place produsul") == "negativ"


def test_neutral():
    assert model_function("Produsul este ok") == "neutru"


def test_empty_input():
    assert model_function("") == "neutru"


def test_case_insensitivity():
    assert model_function("NU IMI PLACE") == "negativ"


def test_priority_logic():
    assert model_function("Nu îmi place") == "negativ"
