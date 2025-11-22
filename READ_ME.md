# 🎭 Sentiment Analysis MLOps Pipeline

Sistema completo di **MLOps** per il deploy automatizzato e il monitoraggio in tempo reale di un modello di **Sentiment Analysis** per recensioni in lingua inglese.

## 📋 Indice

- [Panoramica](#-panoramica)
- [Architettura](#-architettura)
- [Tecnologie Utilizzate](#-tecnologie-utilizzate)
- [Prerequisiti](#-prerequisiti)
- [Installazione e Setup](#-installazione-e-setup)
- [Utilizzo](#-utilizzo)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Monitoraggio](#-monitoraggio)
- [API Reference](#-api-reference)
- [Test](#-test)
- [Troubleshooting](#-troubleshooting)
- [Manutenzione](#-manutenzione)
- [Metriche del Modello](#-metriche-del-modello)
- [Struttura del Progetto](#-struttura-del-progetto)
- [Contributi](#-contributi)
- [Licenza](#-licenza)

## 🌟 Panoramica

Questo progetto implementa una **pipeline MLOps completa** per un modello di Sentiment Analysis che classifica recensioni in inglese come **Positive** o **Negative**.
Il sistema include:

- ✅ **Training automatizzato** con tracking MLflow
- ✅ **Quality Gate** basato su F1-Score (soglia: 0.85)
- ✅ **API REST** autenticata con FastAPI
- ✅ **CI/CD Pipeline** con Jenkins
- ✅ **Containerizzazione** con Docker
- ✅ **Orchestrazione** con Kubernetes
- ✅ **Monitoraggio** con Prometheus e Grafana
- ✅ **Rollback automatico** in caso di fallimento

### 🎯 Benefici Aziendali

- **Automazione completa**: Deploy automatico ad ogni commit con test e validazione
- **Affidabilità**: Quality gate e rollback automatico garantiscono alta qualità
- **Osservabilità**: Monitoraggio in tempo reale delle prestazioni del modello
- **Scalabilità**: Architettura cloud-native con Kubernetes

## 🏗️ Architettura

```
┌─────────────────────────────────────────────────────────────────┐
│                         GitHub Repository                       │
│                    (Source Code + Jenkinsfile)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ Git Push/Commit
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Jenkins CI/CD Pipeline                     │
│  ┌──────────────┐  ┌──────────┐  ┌────────┐  ┌──────────────┐   │
│  │   Training   │→ │  Tests   │→ │ Docker │→ │  Kubernetes  │   │
│  │ + Quality    │  │   API    │  │ Build  │  │    Deploy    │   │
│  │   Gate       │  │          │  │  Push  │  │              │   │
│  └──────────────┘  └──────────┘  └────────┘  └──────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ Deploy Success
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster (Docker Desktop)          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Sentiment Analysis Deployment                 │ │
│  │  ┌──────────────┐              ┌──────────────┐            │ │
│  │  │   Pod 1      │              │   Pod 2      │            │ │
│  │  │ FastAPI App  │◄────────────►│ FastAPI App  │            │ │
│  │  └──────────────┘              └──────────────┘            │ │
│  │         │                              │                   │ │
│  │         └──────────────┬───────────────┘                   │ │
│  │                        │                                   │ │
│  │                ┌───────▼────────┐                          │ │
│  │                │    Service     │                          │ │
│  │                │  NodePort 30080│                          │ │
│  │                └────────────────┘                          │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Scrape Metrics
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Monitoring Stack                           │
│  ┌──────────────┐              ┌──────────────┐                 │
│  │  Prometheus  │─────────────►│   Grafana    │                 │
│  │  (Scraper)   │   Metrics    │ (Dashboard)  │                 │
│  └──────────────┘              └──────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tecnologie Utilizzate

### Machine Learning & Data Science

- **Python 3.10** - Linguaggio principale
- **scikit-learn** - Training del modello (Logistic Regression + TF-IDF)
- **MLflow** - Experiment tracking e model registry
- **pandas** - Manipolazione dati

### API & Web Framework

- **FastAPI** - REST API framework
- **Uvicorn** - ASGI server
- **Pydantic** - Validazione dati
- **prometheus-fastapi-instrumentator** - Raccolta metriche

### DevOps & Infrastructure

- **Docker** - Containerizzazione
- **Kubernetes** - Container orchestration (Docker Desktop)
- **Jenkins** - CI/CD automation
- **Git/GitHub** - Version control

### Monitoring & Observability

- **Prometheus** - Raccolta metriche
- **Grafana** - Visualizzazione metriche
- **Custom Dashboards** - Real-time monitoring

### Testing

- **pytest** - Testing framework
- **requests** - HTTP testing

## 💻 Prerequisiti

### Requisiti Hardware Minimi

- **CPU**: 4 core (2.0 GHz o superiore)
- **RAM**: 8 GB (16 GB raccomandati)
- **Disk**: 20 GB spazio libero
- **OS**: Windows 10/11, macOS, o Linux

### Software Richiesto

1. **Docker Desktop** (versione 4.0+)
   - Con Kubernetes abilitato
   - Download: <https://www.docker.com/products/docker-desktop>

2. **Python 3.10**
   - Download: <https://www.python.org/downloads/>

3. **Git**
   - Download: <https://git-scm.com/downloads>

4. **Jenkins** (opzionale per esecuzione locale)
   - Download: <https://www.jenkins.io/download/>

5. **kubectl** (installato automaticamente dalla pipeline)
   - Verifica: `kubectl version --client`

### Account Richiesti

- **Docker Hub**: Per il registry delle immagini
- **GitHub**: Per il repository del codice

## 🚀 Installazione e Setup

### 1. Clone del Repository

```bash
git clone https://github.com/Giampiero1998/Sentiment.git
cd Sentiment
```

### 2. Setup Ambiente Python

```bash
# Crea virtual environment
python -m venv venv

# Attiva virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Installa dipendenze
pip install -r requirements.txt
```

### 3. Setup Docker Desktop

1. Avvia Docker Desktop
2. Abilita Kubernetes:
   - Settings → Kubernetes → Enable Kubernetes
   - Apply & Restart
3. Verifica: `kubectl cluster-info`

### 4. Setup Jenkins

#### Opzione A: Jenkins in Docker (Raccomandato)

```bash
docker run -d -p 8080:8080 -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --name jenkins \
  jenkins/jenkins:lts
```

#### Opzione B: Jenkins Locale

- Scarica e installa da <https://www.jenkins.io/download/>
- Avvia su <http://localhost:8080>

#### Configurazione Jenkins

1. **Installa Plugin**:
   - Docker Pipeline
   - Kubernetes CLI
   - Git

2. **Configura Credentials**:

   a) **Docker Hub** (ID: `docker-registry-creds`):

   ```
   Kind: Username with password
   Username: <il-tuo-username>
   Password: <il-tuo-docker-hub-token>
   ID: docker-registry-creds
   ```

   b) **Kubernetes Config** (ID: `k8s-kubeconfig-file`):

   ```
   Kind: Secret file
   File: <fresh_kubeconfig.yml con insecure-skip-tls-verify: true>
   ID: k8s-kubeconfig-file
   ```

3. **Crea Pipeline**:
   - New Item → Pipeline
   - Pipeline script from SCM
   - SCM: Git
   - Repository URL: <https://github.com/Giampiero1998/Sentiment>
   - Script Path: Jenkinsfile

### 5. Setup Kubernetes Secrets

```bash
# Secret per API Key
kubectl create secret generic sentiment-api-key \
  --from-literal=API_KEY='SUPER_SECRET_TOKEN_12345'

# Secret per Docker Registry
kubectl create secret docker-registry regcred \
  --docker-server=https://index.docker.io/v1/ \
  --docker-username=<USERNAME> \
  --docker-password=<PASSWORD> \
  --docker-email=<EMAIL>
```

### 6. Setup Prometheus

```bash
# Crea file prometheus.yml
cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'sentiment-api'
    scrape_interval: 10s
    metrics_path: '/metrics'
    static_configs:
      - targets: ['host.docker.internal:30080']
EOF

# Avvia Prometheus
docker run -d -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  --name prometheus \
  prom/prometheus
```

### 7. Setup Grafana

```bash
# Avvia Grafana
docker run -d -p 3000:3000 \
  --name grafana \
  grafana/grafana

# Accedi: http://localhost:3000
# Credenziali: admin / admin (cambiarle al primo accesso)
```

**Configurazione Grafana**:

1. Add Data Source → Prometheus
2. URL: `http://host.docker.internal:9090`
3. Save & Test
4. Import Dashboard: Copia il JSON da `Sentiment Analysis - Monitoring Dashboard-1763365257338.json`

## 📖 Utilizzo

### Training Locale del Modello

```bash
# Attiva virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Esegui training
python train_model.py
```

Output atteso:

```
============================================================
METRICHE DI VALUTAZIONE DEL MODELLO
============================================================
F1-Score: 0.9500
Accuracy: 0.9500
Training set: 80 samples
Test set: 20 samples

✅ Quality Gate PASSED: F1-Score (0.9500) >= Threshold (0.85)
```

### Avvio API in Locale

```bash
# Assicurati di aver eseguito il training prima
export API_KEY='SUPER_SECRET_TOKEN_12345'
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Test dell'API

#### Windows PowerShell

```powershell
# Health Check
curl http://localhost:8000/health

# Predict (Positivo)
Invoke-WebRequest -Uri "http://localhost:8000/predict" `
  -Method POST `
  -Headers @{"Content-Type"="application/json"; "x-api-key"="SUPER_SECRET_TOKEN_12345"} `
  -Body '{"text": "This movie is amazing!"}'

# Predict (Negativo)
Invoke-WebRequest -Uri "http://localhost:8000/predict" `
  -Method POST `
  -Headers @{"Content-Type"="application/json"; "x-api-key"="SUPER_SECRET_TOKEN_12345"} `
  -Body '{"text": "This movie is terrible!"}'

# Metrics
curl http://localhost:8000/metrics
```

#### Linux/Mac

```bash
# Health Check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: SUPER_SECRET_TOKEN_12345" \
  -d '{"text": "This movie is amazing!"}'

# Metrics
curl http://localhost:8000/metrics
```

### Deploy su Kubernetes

```bash
# Applica manifest
kubectl apply -f k8s_deployment.yml

# Verifica deployment
kubectl get deployments
kubectl get pods
kubectl get services

# Test API su Kubernetes
curl http://localhost:30080/health
```

## 🔄 CI/CD Pipeline

La pipeline Jenkins è configurata per eseguire automaticamente ad ogni commit:

### Stage 1: Model Training & Quality Gate

- Installa dipendenze Python
- Esegue training del modello
- Salva modello e vectorizer (.pkl)
- Traccia esperimento su MLflow
- **Quality Gate**: Verifica F1-Score >= 0.85
- ❌ **Blocco deploy** se score insufficiente

### Stage 2: Tests

- Avvia API con Uvicorn
- Esegue health check
- Esegue test pytest (integrazione + autenticazione)
- Verifica endpoint `/predict`

### Stage 3: Build Docker Image

- Build immagine Docker
- Tag con hash Git commit
- Push su Docker Hub (giampiero98/sentiment-api)
- Logout sicuro

### Stage 4: Deploy to Kubernetes

- Installa kubectl
- Configura kubeconfig
- Applica manifest K8s
- Verifica rollout
- **Rollback automatico** in caso di errore

### Trigger Pipeline

```bash
# Commit e push
git add .
git commit -m "Update model or code"
git push origin main

# Jenkins rileverà il commit e avvierà automaticamente la pipeline
```

### Monitoraggio Pipeline

Accedi a Jenkins: <http://localhost:8080>

- Visualizza log in tempo reale
- Controlla metriche di build
- Ricevi notifiche di successo/fallimento

## 📊 Monitoraggio

### Prometheus (<http://localhost:9090>)

**Targets**: <http://localhost:9090/targets>

- Verifica che `sentiment-api` sia **UP** (verde)

**Query Utili**:

```promql
# Total requests
http_requests_total{job="sentiment-api"}

# Request rate (req/s)
rate(http_requests_total{job="sentiment-api"}[5m])

# Average response time
rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])

# P95 latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# API health status
up{job="sentiment-api"}
```

### Grafana (<http://localhost:3000>)

**Dashboard Include**:

- 📈 **Request Rate** - Richieste al secondo per endpoint
- ⏱️ **Response Time** - Latenza (p50, p95)
- 📊 **Total Requests** - Contatore totale richieste
- 🎯 **Requests by Endpoint** - Distribuzione per endpoint
- 🚦 **API Health** - Stato UP/DOWN
- 📉 **HTTP Status Codes** - Distribuzione codici risposta

**Metriche Chiave**:

- Average response time: ~10-50ms
- P95 latency: <100ms
- Success rate: >99%

## 📚 API Reference

### Base URL

- **Locale**: `http://localhost:8000`
- **Kubernetes**: `http://localhost:30080`

### Endpoints

#### `GET /health`

Health check pubblico (non richiede autenticazione).

**Response**:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

#### `GET /metrics`

Metriche Prometheus (non richiede autenticazione).

**Response**: Formato Prometheus

```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{handler="/predict",method="POST"} 42.0
```

#### `POST /predict`

Predizione sentiment (richiede autenticazione).

**Headers**:

```
Content-Type: application/json
x-api-key: SUPER_SECRET_TOKEN_12345
```

**Request Body**:

```json
{
  "text": "This movie is amazing!"
}
```

**Response (200 OK)**:

```json
{
  "input_text": "This movie is amazing!",
  "prediction": 1,
  "sentiment": "Positive",
  "processing_time_ms": 12.34
}
```

**Error Responses**:

- `401 Unauthorized`: API key mancante o invalida
- `422 Unprocessable Entity`: JSON malformato
- `500 Internal Server Error`: Errore del modello

## 🧪 Test

### Esecuzione Test

```bash
# Avvia API
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Esegui test
pytest test_api.py -v

# Output atteso:
# test_api.py::test_health_check PASSED
# test_api.py::test_predict_sentiment_success[case1] PASSED
# test_api.py::test_predict_sentiment_success[case2] PASSED
# test_api.py::test_predict_sentiment_unauthorized PASSED
```

### Coverage Test

I test includono:

- ✅ Health check endpoint
- ✅ Predizione sentiment positivo
- ✅ Predizione sentiment negativo
- ✅ Autenticazione fallita (401)
- ✅ Validazione tempo di risposta

## 🔧 Troubleshooting

### Problema: Pod in `CreateContainerConfigError`

**Causa**: Secret Kubernetes mancante

**Soluzione**:

```bash
kubectl create secret generic sentiment-api-key \
  --from-literal=API_KEY='SUPER_SECRET_TOKEN_12345'

kubectl delete pod -l app=sentiment-api
```

### Problema: Prometheus mostra target DOWN

**Causa**: Porta errata o API non raggiungibile

**Soluzione**:

1. Verifica che l'API sia accessibile: `curl http://localhost:30080/health`
2. Controlla `prometheus.yml`: porta deve essere `30080`
3. Riavvia Prometheus: `docker restart prometheus`

### Problema: Jenkins pipeline fallisce su kubectl

**Causa**: Kubeconfig non configurato correttamente

**Soluzione**:

1. Usa kubeconfig con `insecure-skip-tls-verify: true`
2. Verifica credential ID: `k8s-kubeconfig-file`
3. Server deve essere: `https://kubernetes.docker.internal:6443`

### Problema: Docker build fallisce

**Causa**: File modello mancante

**Soluzione**:

```bash
# Esegui training prima del build
python train_model.py

# Verifica che esistano:
ls -la sentiment_model.pkl tfidf_vectorizer.pkl
```

### Problema: API restituisce 401

**Causa**: API key errata

**Soluzione**:

- Usa header: `x-api-key: SUPER_SECRET_TOKEN_12345`
- Verifica secret K8s: `kubectl get secret sentiment-api-key -o yaml`

## 🔄 Manutenzione

### Aggiornamento Modello

1. Modifica dataset in `train_model.py`
2. Esegui training: `python train_model.py`
3. Commit e push:

```bash
git add train_model.py
git commit -m "Update training dataset"
git push origin main
```

4. La pipeline eseguirà automaticamente il deploy

### Scaling Deployment

```bash
# Aumenta repliche
kubectl scale deployment sentiment-analysis-deployment --replicas=5

# Verifica
kubectl get pods -l app=sentiment-api
```

### Visualizzazione Logs

```bash
# Logs dei pod
kubectl logs -l app=sentiment-api --tail=100 -f

# Logs di un pod specifico
kubectl logs <pod-name> -f

# Logs Jenkins (Docker)
docker logs jenkins -f
```

### Backup MLflow

```bash
# Backup database MLflow
cp mlruns.db mlruns_backup_$(date +%Y%m%d).db

# Backup completo esperimenti
tar -czf mlruns_backup.tar.gz mlruns/ mlruns.db
```

### Rollback Manuale

```bash
# Lista deployment history
kubectl rollout history deployment/sentiment-analysis-deployment

# Rollback all'ultima versione
kubectl rollout undo deployment/sentiment-analysis-deployment

# Rollback a revisione specifica
kubectl rollout undo deployment/sentiment-analysis-deployment --to-revision=2
```

## 📈 Metriche del Modello

### Training Performance

| Metrica | Valore | Threshold     |
|---------|--------|---------------|
| **F1-Score** | 0.95 | >= 0.85✅  |
| **Accuracy** | 0.95 | -          |
| **Precision** | 0.95| -          |
| **Recall** | 0.95   | -          |

### Dataset

- **Dimensione totale**: 100 recensioni
- **Training set**: 80 recensioni (80%)
- **Test set**: 20 recensioni (20%)
- **Bilanciamento**: 50% positive, 50% negative
- **Lingua**: Inglese
- **Dominio**: Recensioni prodotti e servizi

### Modello

- **Algoritmo**: Logistic Regression
- **Vectorization**: TF-IDF (max_features=200, ngram_range=(1,2))
- **Regolarizzazione**: C=1.0, class_weight='balanced'
- **Framework**: scikit-learn

## 📁 Struttura del Progetto

```
Sentiment/
├── api.py                          # REST API FastAPI
├── train_model.py                  # Script training modello
├── test_api.py                     # Test suite
├── requirements.txt                # Dipendenze Python
├── Dockerfile                      # Container definition
├── Jenkinsfile                     # CI/CD pipeline
├── k8s_deployment.yml              # Kubernetes manifest
├── prometheus.yml                  # Configurazione Prometheus
├── .gitignore                      # File ignorati da Git
├── README.md                       # Questa documentazione
│
├── mlruns/                         # MLflow experiments (git-ignored)
├── mlruns.db                       # MLflow database (git-ignored)
├── sentiment_model.pkl             # Modello serializzato (git-ignored)
├── tfidf_vectorizer.pkl            # Vectorizer (git-ignored)
└── model_metrics.txt               # F1-Score per Quality Gate (git-ignored)
```

## 🤝 Contributi

Questo progetto è stato sviluppato come progetto didattico per il corso MLOps.

Per contribuire:

1. Fork del repository
2. Crea branch feature: `git checkout -b feature/NuovaFeature`
3. Commit modifiche: `git commit -m 'Add NuovaFeature'`
4. Push: `git push origin feature/NuovaFeature`
5. Apri Pull Request

## 👨‍💻 Autore

**Giampiero**

- GitHub: [@Giampiero1998](https://github.com/Giampiero1998)
- Repository: <https://github.com/Giampiero1998/Sentiment>

## 🙏 Contributi

- Dataset: Custom training set basato su recensioni sintetiche
- Framework: FastAPI, scikit-learn, MLflow
- Infrastructure: Docker, Kubernetes, Jenkins, Prometheus, Grafana
- Corso: Progetto MLOps - Profession.AI

---

**📌 Nota**: Questo progetto è stato sviluppato a scopo didattico.
Per uso in produzione, considerare ulteriori hardening di sicurezza, scaling avanzato e disaster recovery.

**🔗 Link Utili**:

- API Locale: <http://localhost:8000/docs>
- Kubernetes API: <http://localhost:30080>
- Jenkins: <http://localhost:8080>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000>
