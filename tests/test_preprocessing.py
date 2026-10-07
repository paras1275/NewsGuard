import pytest
from src.preprocessing import clean_text, tokenize_text, lemmatize_text, preprocess_article

def test_clean_text():
    sample = "<h3>SHOCKING!</h3> Check out http://example.com for 100% facts!!!"
    cleaned = clean_text(sample)
    assert "http" not in cleaned
    assert "<h3>" not in cleaned
    assert "!" not in cleaned
    assert cleaned.islower()

def test_tokenize_text():
    sample = "Breaking news report published today"
    tokens = tokenize_text(sample)
    assert isinstance(tokens, list)
    assert len(tokens) == 4
    assert "breaking" in tokens

def test_lemmatize_text():
    tokens = ["economists", "stated", "inflation", "decreased"]
    lemmas = lemmatize_text(tokens)
    assert isinstance(lemmas, list)
    assert len(lemmas) > 0

def test_preprocess_article():
    sample = "According to an official report released today, inflation decreased!"
    processed = preprocess_article(sample)
    assert isinstance(processed, str)
    assert len(processed) > 0
