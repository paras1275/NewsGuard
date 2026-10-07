import pytest
import json
from app.flask_api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    res = client.get('/health')
    assert res.status_code == 200
    data = res.get_json()
    assert data['status'] == 'ok'

def test_model_info_endpoint(client):
    res = client.get('/model-info')
    assert res.status_code == 200
    data = res.get_json()
    assert 'model_name' in data
    assert 'best_cv_f1' in data

def test_predict_success(client):
    payload = {
        "text": "According to an official report released by the Department of Commerce, inflation rates decreased."
    }
    res = client.post('/predict', data=json.dumps(payload), content_type='application/json')
    assert res.status_code == 200
    data = res.get_json()
    assert 'label' in data
    assert 'credibility_score' in data
    assert 0.0 <= data['credibility_score'] <= 100.0

def test_predict_empty_text(client):
    payload = {"text": ""}
    res = client.post('/predict', data=json.dumps(payload), content_type='application/json')
    assert res.status_code == 400
    data = res.get_json()
    assert 'error' in data

def test_predict_missing_text_field(client):
    payload = {"invalid_field": "some text"}
    res = client.post('/predict', data=json.dumps(payload), content_type='application/json')
    assert res.status_code == 400

def test_predict_too_short(client):
    payload = {"text": "hello world"}
    res = client.post('/predict', data=json.dumps(payload), content_type='application/json')
    assert res.status_code == 400
