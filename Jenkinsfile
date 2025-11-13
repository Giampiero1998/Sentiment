pipeline {
    agent any
    
    // Variabili d'ambiente globali
    environment {
        MIN_F1_SCORE_THRESHOLD = '0.85' 
        API_KEY = 'SUPER_SECRET_TOKEN_12345'
        DOCKER_IMAGE_NAME = 'sentiment-api'
        DOCKER_REGISTRY = 'docker.io/giampiero98'
        MLFLOW_TRACKING_URI = 'sqlite:///mlruns.db' 
    }

    stages {   
        // Fase 1 e 2: Training del modello e validazione della qualità 
        stage('Model Training & Quality Gate') {
            agent {
                docker {
                    image 'python:3.10-slim'
                    args '-u root'
                }
            }
            steps {
                script {
                    echo 'Installing project dependencies from requirements.txt...'
                    sh 'pip install -r requirements.txt'
                    //Pulisce eventuali modelli vecchi
                    echo 'Cleaning up old model files...'
                    sh 'rm -f sentiment_model.pkl tfidf_vectorizer.pkl model_metrics.txt || true'

                    // 1. Esegue il training e salva l'F1-Score in model_metrics.txt
                    echo 'Starting model training and logging to MLflow...'
                    sh "export MLFLOW_TRACKING_URI='${MLFLOW_TRACKING_URI}' && python3 train_model.py"
                    
                    // 2. Implementazione del Quality Gate (Eseguito all'interno del container)
                    echo 'Reading F1-Score from model_metrics.txt...'
                    def f1_score = sh(script: "cat model_metrics.txt", returnStdout: true).trim()
                    
                    if (f1_score.toFloat() < MIN_F1_SCORE_THRESHOLD.toFloat()) {
                        error("❌ Deployment BLOCKED: F1-Score (${f1_score}) è sotto la soglia (${MIN_F1_SCORE_THRESHOLD}).")
                    } else {
                        echo "✅ Quality Gate Passed: F1-Score (${f1_score}) è accettabile."
                    }
                }
            }
        }
        
        // Fase 2: Test e sanity check dell'API con Docker
        stage('Tests') {
            agent {
                // Usa un container Python per eseguire i test
                docker {
                    image 'python:3.10-slim'
                    args '-u root'
                }
            }
            steps {
                script {
                    sh '''
                    echo "Installing system dependencies (Debian) and Python requirements..."
                    # Installa utility di sistema necessarie per uvicorn, curl e pkill
                    apt-get update
                    apt-get install -y curl procps coreutils
                    pip install -r requirements.txt
                    '''

                    echo 'Running application and API tests...'
                    sh "export API_KEY='${API_KEY}' && uvicorn api:app --host 0.0.0.0 --port 8000 &"
                    
                    // Ciclo curl per testare attivamente l'endpoint dell'API
                    sh '''
                    echo "Performing sanity check on the API endpoint..."
                    timeout 30s bash -c \
                    'while ! curl -s http://0.0.0.0:8000/health | grep -q "ok"; do \
                    echo -n "Waiting for API to be healthy"; \
                    sleep 1; \
                    done; \
                    echo ""; \
                    echo "API is healthy!"'
                    '''
                    
                    // Esegue i test con pytest
                    echo "Executing pytest for API tests..."
                    sh "pytest"
                    
                    // Termina il server Uvicorn
                    echo "Stopping Uvicorn server..."
                    sh "pkill uvicorn || true" 
                }
            }
        }
        
        // Fase 3: Building e versioning dell'Immagine Docker
        stage('Build Docker Image') {
            agent any  // ✅ FIX: Usa l'agent principale con Docker già installato
            steps {
                script {
                    // Ottiene l'hash del commit Git per il versioning
                    def GIT_COMMIT_TAG = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
                    echo "Using Git Commit Hash as tag: ${GIT_COMMIT_TAG}"

                    // Definisce il tag completo dell'immagine Docker
                    def DOCKER_IMAGE_FULL_TAG = "${DOCKER_REGISTRY}/${DOCKER_IMAGE_NAME}:${GIT_COMMIT_TAG}"
                    
                    // Debug e build dell'immagine Docker
                    echo "Building Docker image with full tag: ${DOCKER_IMAGE_FULL_TAG}"
                    sh "docker build -t ${DOCKER_IMAGE_FULL_TAG} ."

                    // Push sicuro tramite credenziali
                    withCredentials([usernamePassword(credentialsId: 'docker-registry-creds', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                        sh "echo \$DOCKER_PASS | docker login -u \$DOCKER_USER --password-stdin ${DOCKER_REGISTRY}"

                        sh "docker push ${DOCKER_IMAGE_FULL_TAG}"
                        echo "Docker image pushed successfully: ${DOCKER_IMAGE_FULL_TAG}"
                        
                        sh "docker logout ${DOCKER_REGISTRY}"
                    }

                    // Aggiorna le variabili d'ambiente per i passaggi successivi
                    env.DOCKER_IMAGE_TAG = GIT_COMMIT_TAG
                    env.DOCKER_IMAGE_FULL_TAG = DOCKER_IMAGE_FULL_TAG
                }
            }
        }

        // Fase 4: Deploy su Kubernetes
        stage('Deploy to Kubernetes') {
            agent any  // ✅ FIX: Usa l'agent principale
            steps {
                script {
                    echo 'Deploying to Kubernetes cluster...'
                    
                    // 1. Sostituisce il placeholder nell'YAML con il tag corretto dell'immagine
                    sh "sed 's|IMAGE_PLACEHOLDER|${env.DOCKER_IMAGE_FULL_TAG}|g' k8s_deployment.yml > k8s_deployment_final.yml"
                    
                    // 2. Applica la configurazione a Kubernetes
                    sh "kubectl apply -f k8s_deployment_final.yml"
                    echo "Deployment completed for version: ${env.DOCKER_IMAGE_FULL_TAG}"
                }
            }

            // Gestione del rollback in caso di fallimento del deploy
            post {
                failure {
                    echo "🚨 Deployment fallito! Avvio il rollback automatico..."
                    sh "kubectl rollout undo deployment/sentiment-analysis-deployment"
                    echo "✅ Rollback completato. Ripristinata la versione precedente."
                }
            }
        }
    }
    // Gestione degli esiti della pipeline
    post {
        success {
            echo '✅ Pipeline completed successfully!'
        }
        failure {
            echo '❌ Pipeline failed. Check logs for details.'
        }
    }
}