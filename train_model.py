import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle
import mlflow
import os

# 1. Configurazione MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Sentiment_Analysis_Production")

# 2. Dataset più ampio e bilanciato per migliorare la generalizzazione
data = {
    'text': [
        #  Sentiment positivi (50)
        "I love this product, it's absolutely fantastic and amazing!",
        "Extremely satisfied with the exceptional quality and service!",
        "This is a masterpiece, exceeded all my expectations!",
        "Phenomenal results, I'm thrilled and very happy!",
        "Highly recommend this excellent service to everyone!",
        "The staff was incredibly kind, helpful and professional!",
        "Working with this API is extremely pleasant and efficient!",
        "Excellent quality, fast delivery, perfect packaging!",
        "Fantastic service, very helpful and responsive team!",
        "Amazing experience, will definitely buy again!",
        "Best purchase I've ever made in my life!",
        "Outstanding quality and exceptional customer service!",
        "Absolutely love it, highly recommended to friends!",
        "Perfect in every way, couldn't be happier!",
        "Exceeded all expectations, truly remarkable product!",
        "Wonderful product, great value for money!",
        "Impressive results, very satisfied customer here!",
        "Brilliant service, fast response and helpful!",
        "Superb quality, worth every single penny!",
        "Delighted with this purchase, amazing quality!",
        "Incredible product, works perfectly as described!",
        "Five stars, would recommend to absolutely anyone!",
        "Exceptional service and outstanding quality!",
        "Top notch product, very happy customer!",
        "Great experience from start to finish!",
        "Loving every aspect of this product!",
        "Beautifully designed and works flawlessly!",
        "Impressive performance, highly satisfied!",
        "Exceptional value, great purchase decision!",
        "Remarkable quality, exceeded expectations!",
        "Fabulous product, truly outstanding!",
        "Magnificent service, very professional!",
        "Splendid quality, absolutely delighted!",
        "Wonderful experience, highly recommended!",
        "Excellent in every way, love it!",
        "Superior quality, great customer service!",
        "Impressive product, works wonderfully!",
        "Fantastic value, very pleased!",
        "Outstanding performance, highly satisfied!",
        "Brilliant product, exceeded hopes!",
        "Marvelous quality, great experience!",
        "Superb service, very helpful team!",
        "Excellent purchase, very happy!",
        "Amazing quality, love everything about it!",
        "Perfect solution, works great!",
        "Incredible value, highly recommend!",
        "Exceptional product, very impressed!",
        "Wonderful quality, great service!",
        "Outstanding experience, will return!",
        "Brilliant quality, absolutely fantastic!",
        
        #  Sentiment negativi (50) 
        "Terrible experience, will never buy again, awful!",
        "Very disappointed, poor quality, waste of money!",
        "What a complete disappointment, money thrown away!",
        "Slow shipping, broken packaging, terrible service!",
        "Unacceptable response time, very poor service!",
        "Regret buying this cheap imitation, horrible!",
        "Complete garbage, broke immediately, useless!",
        "Returning this immediately, terrible quality!",
        "Very poor quality, broke after one use!",
        "Total waste of money, terrible quality!",
        "Extremely disappointed, not as described at all!",
        "Poor service, would never recommend to anyone!",
        "Broken on arrival, very upset and angry!",
        "Awful experience, will never use again!",
        "Substandard quality, falling apart already!",
        "Not worth the price, very disappointed!",
        "Horrible product, doesn't work at all!",
        "Terrible customer service, completely unhelpful!",
        "Complete failure, avoid at all costs!",
        "Worst purchase I've ever made!",
        "Defective product, very poor quality!",
        "Unsatisfied, requesting immediate refund!",
        "Bad experience overall, very unhappy!",
        "Low quality, breaks very easily!",
        "Regret buying this, complete waste of time!",
        "Disappointing quality, not recommended!",
        "Useless product, doesn't work properly!",
        "Very poor experience, terrible service!",
        "Awful quality, broke immediately!",
        "Completely dissatisfied, waste of money!",
        "Horrible experience, never again!",
        "Poor performance, very disappointed!",
        "Terrible product, not worth buying!",
        "Bad quality, fell apart quickly!",
        "Disappointing purchase, regret it!",
        "Useless and overpriced, terrible!",
        "Very unhappy, poor quality!",
        "Awful service, not helpful at all!",
        "Worst quality, completely unusable!",
        "Terrible value, waste of money!",
        "Poor design, doesn't work well!",
        "Very bad experience, disappointed!",
        "Horrible quality, broke quickly!",
        "Awful product, not recommended!",
        "Disappointing service, very poor!",
        "Useless purchase, regret buying!",
        "Very poor quality, not good!",
        "Terrible experience, avoid this!",
        "Bad product, doesn't work!",
        "Disappointing overall, unhappy customer!"
    ],
    'sentiment': [1]*50 + [0]*50  # 50 positivi, 50 negativi
}

df = pd.DataFrame(data)

# Parametri del modello ottimizzati
MAX_FEATURES = 200  # Aumentato per catturare più pattern
RANDOM_STATE = 42
TEST_SIZE = 0.2  # Ridotto a 20% per più dati di training
C_REGULARIZATION = 1.0  # Parametro di regolarizzazione
MIN_F1_SCORE_THRESHOLD = 0.85

# Inizio della run MLflow
with mlflow.start_run() as run:
    
    # Log dei parametri
    mlflow.log_param("language", "english")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("dataset_size", len(df))
    mlflow.log_param("C_regularization", C_REGULARIZATION)
    
    # 3. Pre-processing e Addestramento
    X = df['text']
    y = df['sentiment']

    # Vettorizzazione ottimizzata con n-grams
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),  # Usa unigrammi e bigrammi
        min_df=1,
        sublinear_tf=True
    )
    X_vectorized = vectorizer.fit_transform(X)

    # Split stratificato per mantenere bilanciamento
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y
    )

    # Modello con parametri ottimizzati
    model = LogisticRegression(
        max_iter=1000,
        C=C_REGULARIZATION,
        class_weight='balanced',
        solver='liblinear'
    )
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
    
    print(f"\n{'='*60}")
    print(f"METRICHE DI VALUTAZIONE DEL MODELLO")
    print(f"{'='*60}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    print(f"{'='*60}\n")
    
    # Quality Gate check
    if f1 >= MIN_F1_SCORE_THRESHOLD:
        print(f"✅ Quality Gate PASSED: F1-Score ({f1:.4f}) >= Threshold ({MIN_F1_SCORE_THRESHOLD})")
    else:
        print(f"❌ Quality Gate FAILED: F1-Score ({f1:.4f}) < Threshold ({MIN_F1_SCORE_THRESHOLD})")

    # 5. Serializzazione e archiviazione locale
    MODEL_PATH = 'sentiment_model.pkl'
    VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

    # Archivia i file .pkl sul filesystem
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"Modello e vettorizzatore salvati e tracciati su MLflow.\n")
    
    # Log dei file come artifact in MLflow
    mlflow.log_artifact(MODEL_PATH)
    mlflow.log_artifact(VECTORIZER_PATH)
    mlflow.log_artifact(METRICS_FILE)