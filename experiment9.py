import requests
import json
import pandas as pd
import mlflow

# Set up MLflow tracking URI to use HTTP server
mlflow.set_tracking_uri("http://localhost:5002")

# Prepare test data (same format as training)
test_data = {
    "dataframe_split": {
        "columns": [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
        "data": [
            [5.1, 3.5, 1.4, 0.2],  # setosa
            [6.2, 2.9, 4.3, 1.3],  # versicolor
            [7.3, 2.9, 6.3, 1.8],  # virginica
        ],
    }
}

# Use MLflow's direct prediction API instead of REST server
try:
    predictions = mlflow.pyfunc.load_model("models:/iris_classifier@production").predict(
        pd.DataFrame(test_data["dataframe_split"]["data"], columns=test_data["dataframe_split"]["columns"])
    )
    print("Predictions:", predictions)
except Exception as e:
    print(f"Error making predictions: {e}")

# Health check - try to connect to tracking server if available
try:
    health = requests.get("http://localhost:5002/health", timeout=2)
    if health.status_code == 200:
        print("Health status:", health.status_code)  # Should be 200
    else:
        print(f"Health check failed: {health.status_code}, {health.text}")
except requests.exceptions.RequestException:
    print("Health check: MLflow tracking server not available")

# Model info - try to get from tracking server if available
try:
    info = requests.get("http://localhost:5002/api/2.0/mlflow/registered-models/get", 
                       params={"name": "iris_classifier"}, timeout=2)
    if info.status_code == 200:
        print("Model info:", info.json())
    else:
        print(f"Model info request failed: {info.status_code}, {info.text}")
except requests.exceptions.RequestException:
    print("Model info: MLflow tracking server not available")