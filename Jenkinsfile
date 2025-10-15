pipeline {
    agent any
    
    // Variabili d'ambiente globali
    environment {
        MIN_F1_SCORE_THRESHOLD = '0.85' 
        API_KEY = 'SUPER_SECRET_TOKEN_12345'
        DOCKER_IMAGE_NAME = 'sentiment-api'
        DOCKER_REGISTRY = 'tuo_registry_docker'
        MLFLOW_TRACKING_URI = 'sqlite:///mlruns.db' 
    }

    stages {
        
         //Fase 1: Training del modello e validazione della qualità
        stage('Model Training') {
            steps {
                script {
                    echo 'Starting model training and logging to MLflow...'
                    //Esporta l'URI di MLflow e avvia lo script di training
                    sh "export MLFLOW_TRACKING_URI='${MLFLOW_TRACKING_URI}' && python3 train_model.py"
                }
            }
        }
        
        //Implementazione del Quality Gate
        stage('Validate Model Quality') {
            steps {
                script {
                    echo 'Reading F1-Score from model_metrics.txt...'
                    // 1. Legge l'F1-Score dal file generato da train_model.py
                    def f1_score = sh(script: "cat model_metrics.txt", returnStdout: true).trim()
                    
                    // 2. Confronto con la soglia definita
                    if (f1_score.toFloat() < MIN_F1_SCORE_THRESHOLD.toFloat()) {
                        error("❌ Deployment BLOCKED: F1-Score (${f1_score}) is below threshold (${MIN_F1_SCORE_THRESHOLD}). Model quality insufficient.")
                    } else {
                        echo "✅ Quality Gate Passed: F1-Score (${f1_score}) is acceptable."
                    }
                }
            }
        }
        
        // Fase 2: Test e sanity check dell'API
        stage('Tests') {
            steps {
                script {
                    echo 'Running application and API tests...'
                    sh "export API_KEY='${API_KEY}' && uvicorn api:app --host 0.0.0.0 --port 8000 &"
                    
                    // Attende qualche secondo per assicurarsi che l'API sia avviata
                    sleep 5
                    
                    // Esegue i test con pytest
                    sh "pytest"
                    
                    // Termina il server Uvicorn
                    sh "pkill -f 'uvicorn api:app'" 
                }
            }
        }
        
        // Fase 3: Building e versioning dell'Immagine Docker
        stage('Build Docker Image') {
            steps {
                script {
                    // Ottiene l'hash del commit Git per il versioning
                    def GIT_COMMIT_TAG = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
                    echo "Using Git Commit Hash as tag: ${GIT_COMMIT_TAG}"

                    // Definisce il tag completo dell'immagine Docker
                    def DOCKER_IMAGE_FULL_TAG = "${DOCKER_REGISTRY}/${DOCKER_IMAGE_NAME}:${GIT_COMMIT_TAG}"
                    
                    // Build dell'immagine Docker
                    sh "docker build -t ${DOCKER_IMAGE_FULL_TAG} ."

                    // Push dell'immagine al registry Docker
                    echo "Docker image built and pushed: ${DOCKER_IMAGE_FULL_TAG}"
                    sh "docker push ${DOCKER_IMAGE_FULL_TAG}"

                    // Push sicuro tramite credenziali definite dall'identificativo 'docker-registry-creds'
                    withCredentials([usernamePassword(credentialsId: 'docker-registry-creds', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                        sh "echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin ${DOCKER_REGISTRY}"
                        sh "docker push ${DOCKER_IMAGE_FULL_TAG}"
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
            steps {
                script {
                    echo 'Deploying to Kubernetes cluster...'
                    
                    // 1. Sostituisce il placeholder nell'YAML con il tag corretto dell'immagine
                    sh "sed 's|IMAGE_PLACEHOLDER|${DOCKER_IMAGE_FULL_TAG}|g' k8s_deployment.yml > k8s_deployment_final.yml"
                    
                    // 2. Applica la configurazione a Kubernetes
                    sh "kubectl apply -f k8s_deployment_final.yml"
                    echo "Deployment completed for version: ${DOCKER_IMAGE_FULL_TAG}"
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
            echo 'Pipeline completed successfully.'
        }
        failure {
            echo 'Pipeline failed. Check logs for details.'
        }
    }
}
