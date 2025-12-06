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

# ============================================================================
# TEST CASES - Copertura di tutti gli scenari possibili
# ============================================================================

# ✅ CASI POSITIVI - Sentiment chiaramente positivo
POSITIVE_CLEAR_CASES = [
    ("This product exceeded all my expectations! The quality is outstanding and I'm extremely satisfied.", 1, "Positive"),
    ("Absolutely love this service! Fast delivery, great customer support, and excellent value for money.", 1, "Positive"),
    ("Best purchase I've made this year. Highly recommend to everyone, works perfectly!", 1, "Positive"),
    ("Amazing experience from start to finish. The team was professional and the results are fantastic.", 1, "Positive"),
    ("I'm thrilled with this purchase! Top quality, arrived early, and looks even better than the photos.", 1, "Positive"),
]

# ✅ CASI POSITIVI - Sentiment moderatamente positivo
POSITIVE_MODERATE_CASES = [
    ("Pretty good overall. Does what it's supposed to do and I'm happy with it.", 1, "Positive"),
    ("Solid product, good quality for the price. Would buy again.", 1, "Positive"),
    ("Works well and arrived on time. No complaints so far.", 1, "Positive"),
    ("Nice experience, staff was helpful and the service was decent.", 1, "Positive"),
]

# ❌ CASI NEGATIVI - Sentiment chiaramente negativo (MANCAVANO!)
NEGATIVE_CLEAR_CASES = [
    ("Terrible product, complete waste of money! Broke after two days and customer service is useless.", 0, "Negative"),
    ("Absolutely horrible experience. Poor quality, late delivery, and rude staff. Never buying again!", 0, "Negative"),
    ("Worst purchase ever! Cheap materials, doesn't work as advertised, total garbage.", 0, "Negative"),
    ("Do not buy this! It's a scam, completely defective and they won't refund my money.", 0, "Negative"),
    ("Extremely disappointed and angry. This is junk and the company doesn't care about customers.", 0, "Negative"),
]

# ❌ CASI NEGATIVI - Sentiment moderatamente negativo (MANCAVANO!)
NEGATIVE_MODERATE_CASES = [
    ("Not very satisfied. Quality is below average and it doesn't quite meet my expectations.", 0, "Negative"),
    ("Disappointing purchase. It works but feels cheap and overpriced.", 0, "Negative"),
    ("Expected better quality for this price. Somewhat unhappy with my decision.", 0, "Negative"),
    ("Mediocre at best. Had issues from the start and support was slow to respond.", 0, "Negative"),
]

# ⚠️ EDGE CASES - Casi limite e corner cases
EDGE_CASES = [
    # Testo molto breve
    ("Good", 1, "Positive"),
    ("Bad", 0, "Negative"),
    ("Okay", 1, "Positive"),  # Neutro tendente positivo
    
    # Testo con errori ortografici
    ("Grate prodct, verry hapy with purchas!", 1, "Positive"),
    ("Terible qualiti, wont recomend to anyoen", 0, "Negative"),
    
    # Testo con punteggiatura mista
    ("Great!!! Love it!!!!", 1, "Positive"),
    ("Awful... just awful...", 0, "Negative"),
    
    # Testo con numeri e simboli
    ("5/5 stars! Excellent product #recommended", 1, "Positive"),
    ("1/10 would not buy again... terrible $$$", 0, "Negative"),
]

# Combina tutti i test cases
ALL_TEST_CASES = (
    POSITIVE_CLEAR_CASES + 
    POSITIVE_MODERATE_CASES + 
    NEGATIVE_CLEAR_CASES + 
    NEGATIVE_MODERATE_CASES + 
    EDGE_CASES
)

# Attende l'avvio del server FastAPI prima di eseguire i test
time.sleep(2) 

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_health_check():
    """Verifica che l'endpoint /health sia operativo."""
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    print("✅ Health check passed")

@pytest.mark.parametrize("text, expected_prediction, expected_sentiment", ALL_TEST_CASES)
def test_predict_sentiment_all_cases(text, expected_prediction, expected_sentiment):
    """
    Test completo per tutte le casistiche di sentiment:
    - Positivi chiari e moderati
    - Negativi chiari e moderati  
    - Edge cases (brevi, errori, simboli)
    """
    payload = json.dumps({"text": text})
    response = requests.post(API_URL, headers=HEADERS, data=payload)
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert "prediction" in data, "Missing 'prediction' in response"
    assert "sentiment" in data, "Missing 'sentiment' in response"
    assert "processing_time_ms" in data, "Missing 'processing_time_ms' in response"
    
    # Verifica predizione corretta
    assert data["prediction"] == expected_prediction, \
        f"Expected prediction {expected_prediction}, got {data['prediction']} for text: '{text}'"
    
    # Verifica sentiment label
    assert data["sentiment"] == expected_sentiment, \
        f"Expected sentiment '{expected_sentiment}', got '{data['sentiment']}' for text: '{text}'"
    
    # Verifica processing time positivo
    assert data["processing_time_ms"] > 0, "Processing time should be positive"
    
    print(f"✅ Test passed for: '{text[:50]}...' -> {expected_sentiment}")

def test_predict_sentiment_unauthorized():
    """Verifica che l'accesso senza la chiave API corretta sia negato (401)."""
    unauthorized_headers = {"X-Api-Key": "INCORRECT_KEY", "Content-Type": "application/json"}
    payload = json.dumps({"text": "Authentication failure test"})
    response = requests.post(API_URL, headers=unauthorized_headers, data=payload)
    
    assert response.status_code == 401
    assert "Chiave API non valida o mancante" in response.json()["detail"]
    print("✅ Unauthorized access correctly blocked")

def test_predict_empty_text():
    """Verifica gestione di testo vuoto."""
    payload = json.dumps({"text": ""})
    response = requests.post(API_URL, headers=HEADERS, data=payload)
    
    # L'API dovrebbe gestire testo vuoto (può restituire 400 o fare una predizione default)
    # Questo dipende dall'implementazione dell'API
    assert response.status_code in [200, 400, 422], \
        f"Expected 200, 400 or 422 for empty text, got {response.status_code}"
    print(f"✅ Empty text handled with status code: {response.status_code}")

def test_predict_very_long_text():
    """Verifica gestione di testo molto lungo."""
    long_text = "This is an amazing product! " * 100  # Testo lungo ripetuto
    payload = json.dumps({"text": long_text})
    response = requests.post(API_URL, headers=HEADERS, data=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "sentiment" in data
    print("✅ Very long text handled correctly")

def test_predict_special_characters():
    """Verifica gestione di caratteri speciali."""
    special_text = "Great product!!! ❤️ 😊 #awesome @company"
    payload = json.dumps({"text": special_text})
    response = requests.post(API_URL, headers=HEADERS, data=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1  # Dovrebbe essere positivo
    print("✅ Special characters handled correctly")

def test_predict_mixed_sentiment():
    """Verifica gestione di sentiment misto (positivo + negativo)."""
    mixed_text = "The product is great but the delivery was terrible and customer service was awful"
    payload = json.dumps({"text": mixed_text})
    response = requests.post(API_URL, headers=HEADERS, data=payload)
    
    assert response.status_code == 200
    data = response.json()
    # Il modello dovrebbe fare una scelta (positivo o negativo)
    assert data["prediction"] in [0, 1]
    print(f"✅ Mixed sentiment handled: {data['sentiment']}")

def test_predict_numbers_only():
    """Verifica gestione di input con solo numeri."""
    payload = json.dumps({"text": "12345"})
    response = requests.post(API_URL, headers=HEADERS, data=payload)
    
    assert response.status_code in [200, 400, 422]
    print(f"✅ Numbers-only input handled with status code: {response.status_code}")

def test_predict_malformed_json():
    """Verifica gestione di JSON malformato."""
    malformed_payload = '{"text": "test"'  # JSON incompleto
    response = requests.post(
        API_URL, 
        headers=HEADERS, 
        data=malformed_payload
    )
    
    assert response.status_code in [400, 422], \
        f"Expected 400 or 422 for malformed JSON, got {response.status_code}"
    print("✅ Malformed JSON correctly rejected")

def test_response_time_performance():
    """Verifica che il tempo di risposta sia ragionevole (< 1 secondo)."""
    payload = json.dumps({"text": "This is a test for performance"})
    
    start_time = time.time()
    response = requests.post(API_URL, headers=HEADERS, data=payload)
    end_time = time.time()
    
    response_time = end_time - start_time
    
    assert response.status_code == 200
    assert response_time < 1.0, f"Response time too slow: {response_time:.3f}s"
    print(f"✅ Response time: {response_time:.3f}s (acceptable)")

def test_consistency_multiple_requests():
    """Verifica consistenza delle predizioni su richieste multiple dello stesso testo."""
    text = "This is an excellent product that I highly recommend!"
    payload = json.dumps({"text": text})
    
    results = []
    for _ in range(5):
        response = requests.post(API_URL, headers=HEADERS, data=payload)
        assert response.status_code == 200
        results.append(response.json()["prediction"])
    
    # Tutte le predizioni dovrebbero essere identiche
    assert all(r == results[0] for r in results), "Predictions are inconsistent"
    print("✅ Predictions are consistent across multiple requests")

# ============================================================================
# SUMMARY
# ============================================================================
"""
COPERTURA TEST COMPLETA:

✅ Sentiment Positivi:
   - Chiari (5 casi)
   - Moderati (4 casi)

❌ Sentiment Negativi:
   - Chiari (5 casi) 
   - Moderati (4 casi)

⚠️ Edge Cases:
   - Testo breve
   - Errori ortografici
   - Punteggiatura mista
   - Numeri e simboli
   - Testo vuoto
   - Testo molto lungo
   - Caratteri speciali
   - Sentiment misto
   - Solo numeri
   - JSON malformato

🔒 Security & Performance:
   - Test autenticazione
   - Test performance
   - Test consistenza

✅ Tutti i test sono stati implementati con successo per garantire la robustezza e l'affidabilità dell'API di analisi del sentiment.
"""