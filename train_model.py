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

# 2. Preparazione dati di addestramento (DATASET AMPLIATO)
data = {
    'text': [
        # Positivi
        "I love this product, it's fantastic!", 
        "I am extremely satisfied with the result.",
        "This film is an absolute masterpiece.",
        "This is a phenomenal result, I'm thrilled!", 
        "I highly recommend this service.",           
        "The staff was incredibly kind and helpful.",
        "Working with this API is extremely pleasant.",
        "Excellent quality, fast delivery.",
        "Fantastic service, very helpful.",
        "Amazing experience, will buy again!",
        "Best purchase I've ever made.",
        "Outstanding quality and service.",
        "Absolutely love it, highly recommended!",
        "Perfect in every way.",
        "Exceeded all my expectations.",
        "Wonderful product, great value.",
        "Impressive results, very satisfied.",
        "Brilliant service, fast response.",
        "Superb quality, worth every penny.",
        "Delighted with this purchase.",
        "Incredible product, works perfectly.",
        "Five stars, would recommend to anyone.",
        "Exceptional service and quality.",
        "Top notch product, very happy.",
        "Great experience from start to finish.",
        
        # Negativi
        "Terrible experience, I will not buy it again.",
        "The service was decent, but nothing exceptional.",
        "What a disappointment, money thrown away.",
        "The shipping was slow and the packaging broken.",
        "I found the response time unacceptable.",
        "I regret spending my money on this cheap imitation.",
        "The product is complete garbage.",
        "I will be returning this immediately.",
        "The quality is very poor and it broke.",
        "Waste of money, terrible quality.",
        "Disappointed, not as described.",
        "Poor service, would not recommend.",
        "Broken on arrival, very upset.",
        "Awful experience, never again.",
        "Substandard quality, falling apart.",
        "Not worth the price, disappointed.",
        "Horrible product, doesn't work.",
        "Terrible customer service, unhelpful.",
        "Complete failure, avoid at all costs.",
        "Worst purchase ever made.",
        "Defective product, poor quality.",
        "Unsatisfied, requesting refund.",
        "Bad experience overall.",
        "Low quality, breaks easily.",
        "Regret buying this, waste of time."
    ],
    'sentiment': [
        # 25 positivi (1)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # 25 negativi (0)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
}
df = pd.DataFrame(data)

# Parametri del modello
MAX_FEATURES = 100
RANDOM_STATE = 42
TEST_SIZE = 0.3  # 30% test = 15 campioni
MIN_F1_SCORE_THRESHOLD = 0.85

# Inizio della run MLflow
with mlflow.start_run() as run:
    
    # Log dei parametri
    mlflow.log_param("language", "english")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("dataset_size", len(df))
    
    # 3. Pre-processing e Addestramento
    X = df['text']
    y = df['sentiment']

    # Vettorizzazione del testo
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_vectorized = vectorizer.fit_transform(X)

    # Split dei dati
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Addestramento del modello
    model = LogisticRegression(max_iter=1000)
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
    print(f"Accuracy: {accuracy}")
    print(f"Test set size: {len(y_test)} samples")

    # 5. Serializzazione e archiviazione locale
    MODEL_PATH = 'sentiment_model.pkl'
    VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

    # Archivia i file .pkl sul filesystem per il Docker container
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"Modello ({MODEL_PATH}) e vettorizzatore archiviati in locale e tracciati su MLflow.")
    
    # Log dei file .pkl come artifact in MLflow
    mlflow.log_artifact(MODEL_PATH)
    mlflow.log_artifact(VECTORIZER_PATH)
    mlflow.log_artifact(METRICS_FILE)