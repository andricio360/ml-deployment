# ML Deployment

Project to deploy ML model using Docker and Kubernetes. Take Iris ML Model, try three different types of model, versioning them with mlflow and detecting drift using Evidently AI for each prediction.

## Kubernetes Deployment

To deploy the ML model using Kubernetes, follow these steps:

1. Start Minikube with Docker as the driver:
minikube start --driver=docker

2. Apply the Kubernetes deployment configuration:
kubectl apply -f deployment.yaml

3. Apply the Kubernetes service configuration:
kubectl apply -f service.yaml

## Display URL

To display the URL of the deployed service, use the following command:
minikube service iris-service --url
