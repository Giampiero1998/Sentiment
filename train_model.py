import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pickle
import mlflow
import os

# --- 1. Configurazione MLflow ---
# L'URI di tracciamento può essere impostato tramite variabile d'ambiente MLFLOW_TRACKING_URI.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Sentiment_Analysis_Production")

# --- 2. Preparazione Dati (Simulazione - INGLESE) ---
# Dati di esempio per l'addestramento del modello in inglese.
data = {
    'text': [
        "I love this product, it's fantastic!", 
        "Terrible experience, I will not buy it again.",
        "The service was decent, but nothing exceptional.",
        "I am extremely satisfied with the result.",
        "What a disappointment, money thrown away.",
        "This film is an absolute masterpiece.",
        "The shipping was slow and the packaging broken."
    ],
    'sentiment': [1, 0, 0, 1, 0, 1, 0] # 1: Positive, 0: Negative
}
df = pd.DataFrame(data)

# Parametri del modello e del training
MAX_FEATURES = 100
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Avvia una nuova run di MLflow
with mlflow.start_run() as run:
    
    # Registrazione dei parametri in MLflow
    mlflow.log_param("language", "english")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("model_type", "LogisticRegression")
    
    # --- 3. Pre-processing e Addestramento ---
    X = df['text']
    y = df['sentiment']

    # Vettorizzazione TF-IDF
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_vectorized = vectorizer.fit_transform(X)

    # Split dei dati
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Addestramento del modello
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # --- 4. Valutazione e Logging delle Metriche ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Log delle metriche di test
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_f1_score", f1)
    
    # --- 5. Serializzazione e Archiviazione Locale ---
    # NOMI FILE AGGIORNATI
    MODEL_PATH = 'sentimentanalysismodel.pkl'
    VECTORIZER_PATH = 'sentimentanalysis_vectorizer.pkl'

    # SALVATAGGIO LOCALE: Archivia i file .pkl sul filesystem per il Dockerfile
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Registra i file .pkl come artefatti di MLflow per la governance (tracking)
    mlflow.log_artifact(MODEL_PATH)
    mlflow.log_artifact(VECTORIZER_PATH)
    
    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"Modello ({MODEL_PATH}) e Vettorizzatore archiviati in locale e tracciati su MLflow.")