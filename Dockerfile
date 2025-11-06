# Usa una immagine base Python ottimizzata
FROM python:3.10-slim

# Imposta la directory di lavoro all'interno del container
WORKDIR /app

# Copia le dipendenze e installale
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia i file del modello archiviati localmente (dal runner CI/CD) nell'immagine
COPY api.py .
COPY sentimentanalysismodel.pkl .
COPY sentimentanalysis_vectorizer.pkl .

# Variabile d'ambiente API_KEY, letta dall'API per l'autenticazione
ENV API_KEY="SUPER_SECRET_TOKEN_12345"

# Espone la porta su cui uvicorn (FastAPI) si metterà in ascolto
EXPOSE 8000

# Comando per avviare l'API utilizzando Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]