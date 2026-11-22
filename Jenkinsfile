pipeline {
    agent any

    stages {
        stage('Check Infra & Cluster') {
            steps {
                sh 'terraform -version && minikube status && kubectl get nodes'
            }
        }
        stage('Apply Base Manifests') {
            steps {
                sh 'kubectl apply -k manifests && kubectl get pods'
            }
        }
        stage('Pipeline & MLflow Registry') {
            steps {
                sh '/home/abdullah/k8s-mlops-assignment/.venv/bin/python ml_pipeline/pipeline_and_registry.py'
            }
        }
        stage('Deploy Replicas & Verify') {
            steps {
                sh 'kubectl apply -f deployment/model-deployment.yaml && kubectl get pods -l app=iris-model -o wide'
            }
        }
    }
}
