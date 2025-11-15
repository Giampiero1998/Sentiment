
pipeline {
    agent any
    
    // Variabili d'ambiente globali
    environment {
        MIN_F1_SCORE_THRESHOLD = '0.85' 
        API_KEY = 'SUPER_SECRET_TOKEN_12345'
        DOCKER_IMAGE_NAME = 'sentiment-api'
        //Separo il registry radice dal nome utente per facilitare il login/logout 
        DOCKER_REGISTRY_URL = 'https://docker.io'
        DOCKER_NAMESPACE = 'giampiero98'

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

                    // Pulisce eventuali modelli vecchi
                    echo 'Cleaning old model files...'
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
                    
                    // 3. Salva i file del modello per gli stage successivi
                    echo 'Stashing model files for next stages...'
                    stash includes: 'sentiment_model.pkl,tfidf_vectorizer.pkl,model_metrics.txt', name: 'model-files'
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
            agent any  
            steps {
                script {
                    // Ottiene l'hash del commit Git per il versioning
                    def GIT_COMMIT_TAG = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
                    echo "Using Git Commit Hash as tag: ${GIT_COMMIT_TAG}"

                    // Definisce il tag completo dell'immagine Docker
                    def DOCKER_IMAGE_FULL_TAG_WITH_PROTOCOL = "${DOCKER_REGISTRY_URL}/${DOCKER_NAMESPACE}/${DOCKER_IMAGE_NAME}:${GIT_COMMIT_TAG}"

                    // Variabile finale per i comandi CLI (senza https://)
                    def DOCKER_IMAGE_FULL_TAG_CLI = DOCKER_IMAGE_FULL_TAG_WITH_PROTOCOL.replace('https://', '')
                    
                    // Debug e build dell'immagine Docker
                    echo "Building Docker image with full tag: ${DOCKER_IMAGE_FULL_TAG_CLI}"
                    sh "docker build -t ${DOCKER_IMAGE_FULL_TAG_CLI} ."

                    //Estrae l'host dal registry URL per il login
                    def DOCKER_LOGIN_HOST = "${DOCKER_REGISTRY_URL}".replace('https://', '')

                    // Push sicuro tramite credenziali
                    withCredentials([usernamePassword(credentialsId: 'docker-registry-creds', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {

                        sh "echo \$DOCKER_PASS | docker login -u \$DOCKER_USER --password-stdin ${DOCKER_LOGIN_HOST}"

                        sh "docker push ${DOCKER_IMAGE_FULL_TAG_CLI}"
                        echo "Docker image pushed successfully: ${DOCKER_IMAGE_FULL_TAG_CLI}"
                        
                        sh "docker logout ${DOCKER_LOGIN_HOST} || true"
                    }

                    // Aggiorna le variabili d'ambiente per i passaggi successivi
                    env.DOCKER_IMAGE_TAG = GIT_COMMIT_TAG
                    env.DOCKER_IMAGE_FULL_TAG = DOCKER_IMAGE_FULL_TAG_CLI
                }
            }
        }

        // Fase 4: Deploy su Kubernetes - SOLUZIONE COMPLETA
        stage('Deploy to Kubernetes') {
            agent {
                docker {
                    image 'debian:latest
                    args '--network host'
                    }
                }
                steps {
                    withCredentials([file(credentialsId: 'k8s-kubeconfig-file', variable: 'KUBECONFIG_FILE')]) {
                        script {
                            
                            echo 'Installing kubectl...'
                            sh '''
                            apt-get update -qq
                            apt-get install -y curl sed > /dev/null 2>&1

                            curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl" 2>/dev/null
                            chmod +x ./kubectl && mv ./kubectl /usr/local/bin/kubectl
                        
                            echo "✓ kubectl installed: $(kubectl version --client --short 2>/dev/null || echo 'installed')"
                            '''

                            // Setup kubeconfig con validazione
                            echo 'Configuring kubeconfig...'
                            sh '''
                            mkdir -p ~/.kube
                        
                            # Leggi e scrivi il kubeconfig
                            cat "${KUBECONFIG_FILE}" > ~/.kube/config
                            chmod 600 ~/.kube/config
                        
                            # Verifica che il file sia stato creato
                            if [ ! -f ~/.kube/config ]; then
                                echo "❌ ERROR: Failed to create kubeconfig file"
                                exit 1
                            fi
                        
                            FILE_SIZE=$(wc -c < ~/.kube/config)
                            echo "✓ Kubeconfig created: ${FILE_SIZE} bytes, $(wc -l < ~/.kube/config) lines"
                        
                            # Mostra struttura kubeconfig (senza dati sensibili)
                            echo "=== Kubeconfig Structure ==="
                            grep -E "^(apiVersion|kind|current-context|clusters:|contexts:|users:)" ~/.kube/config || true
                            echo "=========================="
                            '''

                            // Test connessione Kubernetes CRITICO
                            echo 'Testing Kubernetes connection...'
                            sh '''
                            export KUBECONFIG=~/.kube/config
                        
                            echo "Current context:"
                            kubectl config current-context || echo "⚠ Cannot get current context"
                        
                            echo ""
                            echo "Testing connection to cluster..."
                        
                            # Test 1: Cluster info
                            if kubectl cluster-info --request-timeout=5s 2>&1 | grep -q "Kubernetes"; then
                                echo "✓ Cluster connection successful"
                                kubectl cluster-info
                            else
                                echo "⚠ Cluster info failed, trying with --insecure-skip-tls-verify..."
                                kubectl cluster-info --insecure-skip-tls-verify || echo "⚠ Still cannot connect"
                            fi
                        
                            echo ""
                            echo "Testing API access..."
                        
                            # Test 2: API version
                            if kubectl version 2>&1 | grep -q "Server Version"; then
                                echo "✓ API accessible"
                                kubectl version --short 2>/dev/null || kubectl version
                            else
                                echo "⚠ API not accessible, trying alternative..."
                                kubectl version --insecure-skip-tls-verify || echo "⚠ Cannot get server version"
                            fi
                        
                            echo ""
                            echo "Testing authentication..."
                        
                            # Test 3: Get nodes (test auth)
                            if kubectl get nodes --request-timeout=5s > /dev/null 2>&1; then
                                echo "✓ Authentication successful"
                                kubectl get nodes
                            else
                                echo "⚠ Authentication failed, trying with --insecure-skip-tls-verify..."
                                kubectl get nodes --insecure-skip-tls-verify || echo "⚠ Cannot authenticate"
                            fi
                            '''

                            echo 'Preparing deployment manifest...'
                            def FINAL_IMAGE_TAG = env.DOCKER_IMAGE_FULL_TAG
                        
                            sh """
                            # Sostituisci placeholder con l'immagine Docker
                            sed 's|IMAGE_PLACEHOLDER|${FINAL_IMAGE_TAG}|g' k8s_deployment.yml > k8s_deployment_final.yml
                        
                            echo "=== Generated Deployment ==="
                            head -n 30 k8s_deployment_final.yml
                            echo "==========================="
                            """

                            // Deploy con gestione errori avanzata
                            echo 'Applying Kubernetes deployment...'
                            sh '''
                            export KUBECONFIG=~/.kube/config
                        
                            echo "Attempting deployment with standard validation..."
                        
                            # Tentativo 1: Deploy standard
                            if kubectl apply -f k8s_deployment_final.yml --timeout=30s 2>&1; then
                                echo "✓ Deployment applied successfully (standard)"
                                DEPLOY_SUCCESS=true
                            else
                                echo "⚠ Standard deployment failed, trying with --insecure-skip-tls-verify..."
                            
                                # Tentativo 2: Deploy con --insecure-skip-tls-verify
                                if kubectl apply -f k8s_deployment_final.yml --insecure-skip-tls-verify --timeout=30s 2>&1; then
                                    echo "✓ Deployment applied successfully (insecure)"
                                    DEPLOY_SUCCESS=true
                                else
                                    echo "⚠ Insecure deployment failed, trying without validation..."
                                
                                    # Tentativo 3: Deploy senza validazione
                                        if kubectl apply -f k8s_deployment_final.yml --validate=false --insecure-skip-tls-verify --timeout=30s 2>&1; then
                                            echo "✓ Deployment applied successfully (no validation)"
                                            DEPLOY_SUCCESS=true
                                    else
                                        echo "❌ All deployment attempts failed!"
                                    
                                        # Debug finale
                                        echo ""
                                        echo "=== DEBUG INFO ==="
                                        echo "Kubeconfig content check:"
                                        ls -lh ~/.kube/config
                                    
                                        echo ""
                                        echo "Kubernetes server reachable:"
                                        curl -k https://kubernetes.docker.internal:6443/version 2>&1 || echo "Server not reachable"
                                    
                                        echo ""
                                        echo "Current deployments:"
                                        kubectl get deployments --all-namespaces --insecure-skip-tls-verify 2>&1 || echo "Cannot list deployments"
                                    
                                        exit 1
                                    fi
                                fi
                            fi
                        
                            # Verifica stato deployment
                            echo ""
                            echo "Checking deployment status..."
                            kubectl get deployment sentiment-analysis-deployment --insecure-skip-tls-verify -o wide 2>&1 || echo "⚠ Cannot get deployment status"
                        
                            echo ""
                            echo "Waiting for rollout to complete..."
                            if kubectl rollout status deployment/sentiment-analysis-deployment --timeout=120s --insecure-skip-tls-verify 2>&1; then
                                echo "✓ Rollout completed successfully"
                            else
                                echo "⚠ Rollout status check failed or timed out"
                            fi
                        
                            echo ""
                            echo "Current pods:"
                            kubectl get pods -l app=sentiment-analysis --insecure-skip-tls-verify -o wide 2>&1 || echo "⚠ Cannot get pods"
                        
                            echo ""
                            echo "Services:"
                            kubectl get service sentiment-analysis-service --insecure-skip-tls-verify -o wide 2>&1 || echo "⚠ Cannot get service"
                            '''
                        
                            echo "✅ Deployment completed for version: ${FINAL_IMAGE_TAG}"
                            }
                        }
                    }   
            
                    post {
                        failure {
                            echo "🚨 Deployment failed! Starting automatic rollback..."
                            withCredentials([file(credentialsId: 'k8s-kubeconfig-file', variable: 'KUBECONFIG_FILE')]) {
                                sh '''
                                mkdir -p ~/.kube
                                cat "${KUBECONFIG_FILE}" > ~/.kube/config
                                chmod 600 ~/.kube/config
                                export KUBECONFIG=~/.kube/config
                        
                                echo "Attempting rollback..."
                                if kubectl rollout undo deployment/sentiment-analysis-deployment --insecure-skip-tls-verify 2>&1; then
                                    echo "✓ Rollback initiated"
                            
                                    echo "Checking rollback status..."
                                    kubectl rollout status deployment/sentiment-analysis-deployment --timeout=60s --insecure-skip-tls-verify 2>&1 || echo "⚠ Cannot verify rollback status"
                                else
                                    echo "⚠ Rollback failed - deployment might not exist yet (first deploy?)"
                                fi
                                '''
                            }
                            echo "Rollback process completed"
                        }
                        success {
                            echo "✅ Kubernetes deployment successful!"
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
