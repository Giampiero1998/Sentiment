import fastapi
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import pickle
import os
import time
from prometheus_fastapi_instrumentator import Instrumentator

# Configurazione e Caricamento Modello
MODEL_PATH = 'sentiment_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
# Legge la chiave API da variabile d'ambiente
API_KEY_PROD = os.getenv("API_KEY", "SUPER_SECRET_TOKEN_12345") 

# CARICAMENTO LOCALE: L'API carica i file archiviati localmente
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print("Modello e Vettorizzatore caricati con successo dall'archivio locale.")
except FileNotFoundError as e:
    raise RuntimeError(f"Errore: file del modello mancante. Eseguire train_model.py e Docker build. Dettagli: {e}")

app = FastAPI(
    title="API di Sentiment Analysis (MLOps) ",
    description="API REST autenticata per l'analisi del sentiment e strumentata per Prometheus.",
    version="1.0.0"
)

# Aggiunge la strumentazione Prometheus. Espone automaticamente l'endpoint GET /metrics
Instrumentator().instrument(app).expose(app)

# --- Schemi Dati e Funzione di Autenticazione ---
class TextRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    input_text: str
    prediction: int  # 1: Positivo, 0: Negativo
    sentiment: str
    processing_time_ms: float

async def verify_api_key(x_api_key: str = Header(..., description="La chiave API di produzione è richiesta.")):
    if not x_api_key or x_api_key != API_KEY_PROD:
        raise HTTPException(status_code=401, detail="Chiave API non valida o mancante")

#   Endpoints dell'API  
# GET /health: Endpoint pubblico per il controllo dello stato di salute

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}

# POST /predict: Endpoint protetto per la predizione
@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(
    request: TextRequest, 
    auth_check: None = fastapi.Depends(verify_api_key) # Protezione dell'endpoint
):
    start_time = time.time()
    input_text = request.text
    
    try:
        # 1. Pre-processing
        text_vectorized = vectorizer.transform([input_text])
        
        # 2. Predizione
        prediction = model.predict(text_vectorized)[0]
        sentiment_label = "Positive" if prediction == 1 else "Negative"
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return SentimentResponse(
            input_text=input_text,
            prediction=int(prediction),
            sentiment=sentiment_label,
            processing_time_ms=processing_time_ms
        )
    except Exception as e:
        print(f"Errore durante la predizione: {e}")
        raise HTTPException(status_code=500, detail=f"Errore interno durante la predizione del modello: {e}")