import sys
import os

# ca să poți importa din src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.service import model_function


def test_positive():
    result = model_function("Îmi place produsul")
    assert "pozitiv" in result.lower()


def test_negative():
    result = model_function("Nu îmi place produsul")
    assert "negativ" in result.lower()


def test_neutral():
    result = model_function("Produsul este ok")
    assert "neutru" in result.lower()


def test_empty_input():
    result = model_function("")
    assert result is not None


def test_case_insensitivity():
    assert model_function("NU IMI PLACE") == "negativ"


def test_priority_logic():
    assert model_function("Nu îmi place") == "negativ"
