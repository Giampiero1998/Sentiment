import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pickle
import mlflow
import os

# 1. Configurazione MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Sentiment_Analysis_Production")

# 2. Preparazione dati di addestramento
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

# Parametri del modello
MAX_FEATURES = 100
RANDOM_STATE = 42
TEST_SIZE = 0.3
MIN_F1_SCORE_THRESHOLD = 0.85 # Soglia minima per il Quality Gate

# Inizio della run MLflow
with mlflow.start_run() as run:
    
    # Log dei parametri
    mlflow.log_param("language", "english")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("model_type", "LogisticRegression")
    
    # 3. Pre-processing e Addestramento
    X = df['text']
    y = df['sentiment']

    # Vettorizzazione del testo
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_vectorized = vectorizer.fit_transform(X)

    # Split dei dati
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Addestramento del modello
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 4. Valutazione e logging delle metriche
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Log delle metriche di test
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_f1_score", f1)
    
    # Salva l'F1-Score in un file per il Quality Gate
    METRICS_FILE = 'model_metrics.txt'
    with open(METRICS_FILE, 'w') as f:
        f.write(str(f1))
    
    print(f"F1-Score salvato in {METRICS_FILE}: {f1}")

    # 5. Serializzazione e archiviazione locale
    MODEL_PATH = 'sentimentanalysismodel.pkl'
    VECTORIZER_PATH = 'sentimentanalysis_vectorizer.pkl'

    # Archivia i file .pkl sul filesystem per il Docker container
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Log dei file .pkl come artifact in MLflow
    mlflow.log_artifact(MODEL_PATH)
    mlflow.log_artifact(VECTORIZER_PATH)
    mlflow.log_artifact(METRICS_FILE) # Log del file delle metriche
    
    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"Modello ({MODEL_PATH}) e Vettorizzatore archiviati in locale e tracciati su MLflow.")