import pytest
from src.predict import NewsPredictor

@pytest.fixture(scope="module")
def predictor():
    return NewsPredictor()

def test_predict_single_real(predictor):
    text = "According to an official report released by the Department of Commerce today, inflation rates decreased by 3% this quarter."
    result = predictor.predict_single(text)
    assert isinstance(result, dict)
    assert "label" in result
    assert result["label"] in ["REAL", "FAKE"]
    assert 0.0 <= result["credibility_score"] <= 100.0
    assert "confidence" in result
    assert "execution_time_ms" in result

def test_predict_single_fake(predictor):
    text = "SHOCKING Miracle ingredient cures all diseases instantly! Doctors are furious, share before deleted!"
    result = predictor.predict_single(text)
    assert isinstance(result, dict)
    assert result["label"] in ["REAL", "FAKE"]
    assert 0.0 <= result["credibility_score"] <= 100.0

def test_empty_input(predictor):
    with pytest.raises(ValueError):
        predictor.predict_single("")
