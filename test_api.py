import requests
import json
import pytest
import time

# --- Configurazione Test ---
# Presuppone che l'API sia in esecuzione su localhost:8000
API_URL = "http://localhost:8000/predict" 
API_KEY = "SUPER_SECRET_TOKEN_12345" 

HEADERS = {
    "X-Api-Key": API_KEY,
    "Content-Type": "application/json"
}

# Casi di test aggiornati alla lingua inglese
TEST_CASES = [
    ("This is a phenomenal result, I'm thrilled!", 1, "Positive"), 
    ("The wait was unacceptable and the product was defective.", 0, "Negative"), 
    ("I highly recommend this service.", 1, "Positive"),
    ("The documentation is confusing and incomplete.", 0, "Negative"),
]

# Attende l'avvio del server FastAPI prima di eseguire i test
time.sleep(2) 

def test_health_check():
    """Verifica che l'endpoint /health sia operativo."""
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.mark.parametrize("text, expected_prediction, expected_sentiment", TEST_CASES)
def test_predict_sentiment_success(text, expected_prediction, expected_sentiment):
    """Verifica predizione e struttura della risposta con chiave corretta (test di integrazione)."""
    payload = json.dumps({"text": text})
    response = requests.post(API_URL, headers=HEADERS, data=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == expected_prediction
    assert data["sentiment"] == expected_sentiment
    assert data["processing_time_ms"] > 0

def test_predict_sentiment_unauthorized():
    """Verifica che l'accesso senza la chiave API corretta sia negato (401)."""
    unauthorized_headers = {"X-Api-Key": "INCORRECT_KEY", "Content-Type": "application/json"}
    payload = json.dumps({"text": "Authentication failure test"})
    response = requests.post(API_URL, headers=unauthorized_headers, data=payload)
    
    assert response.status_code == 401
    assert "Chiave API non valida o mancante" in response.json()["detail"]