pipeline {
    agent any 

    // Variabili Globali per la pipeline
    environment {
        // Nomi e configurazioni di build
        DOCKER_IMAGE_NAME = "sentiment-api"
        K8S_NAMESPACE = "default" 
        
        // Credenziali e servizi
        MLFLOW_TRACKING_URI = "http://mlflow-server:5000" 
        API_KEY = "SUPER_SECRET_TOKEN_12345" 
        
        // Variabili per notifica (da configurare in Jenkins)
        SLACK_CHANNEL = "#mlops-notifications"
    }

    stages {
        stage('Checkout Code') {
            steps {
                echo "1. Clonaggio del codice dal SCM (GitHub/GitLab)."
                checkout scm
            }
        }

        stage('Model Training & Local Archive') {
            steps {
                script {
                    echo "2. Addestramento del modello, archiviazione locale dei file .pkl e tracciamento su MLflow."
                    sh "export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}"
                    sh 'python3 train_model.py'
                }
            }
        }

        stage('Tests') {
            steps {
                echo "3. Esecuzione dei test (unitari e di integrazione dell'API)."
                script {
                    // Avvia l'API in background per i test di integrazione
                    sh "export API_KEY=${API_KEY} && uvicorn api:app --host 0.0.0.0 --port 8000 &"
                    sh "sleep 5" // Tempo per l'avvio del server
                    
                    // Esegue i test funzionali con Pytest
                    sh "pytest test_api.py"
                    
                    // Killa il processo API
                    sh "kill \$(lsof -t -i:8000)"
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "4. Costruzione dell'immagine Docker con i file .pkl archiviati localmente."
                script {
                    sh "docker build -t ${DOCKER_IMAGE_NAME}:${env.BUILD_ID} ."
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                echo "5. Deploy al cluster Kubernetes integrato in Docker Desktop (NodePort: 30080)."
                script {
                    // Sostituisce il placeholder dell'immagine nel manifest K8s
                    sh "sed 's|IMAGE_PLACEHOLDER|${DOCKER_IMAGE_NAME}:${env.BUILD_ID}|g' k8s_deployment.yml > final_k8s_deployment.yml"
                    
                    // Applicazione del manifest (richiede kubectl configurato per Docker Desktop)
                    // sh "kubectl apply -f final_k8s_deployment.yml --namespace ${K8S_NAMESPACE}"
                    echo "Deploy K8s simulato completato. L'immagine ${DOCKER_IMAGE_NAME}:${env.BUILD_ID} è stata specificata."
                }
            }
        }
    }
    
    // Post-Esecuzione: Notifiche
    post {
        always {
            echo "Pipeline terminata. Stato: ${currentBuild.result}"
        }
        success {
            echo "Pipeline SUCCESS. Invia notifica su ${SLACK_CHANNEL}."
            // slackSend channel: SLACK_CHANNEL, color: 'good', message: "Pipeline MLOps #${BUILD_NUMBER} SUCCESS."
        }
        failure {
            echo "Pipeline FAILURE. Invia allarme su ${SLACK_CHANNEL}."
            // slackSend channel: SLACK_CHANNEL, color: 'danger', message: "Pipeline MLOps #${BUILD_NUMBER} FAILED."
        }
    }
}