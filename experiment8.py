import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

mlflow.set_tracking_uri('http://localhost:5001')

# Load sample data
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Enable sklearn autologging with model registration
mlflow.sklearn.autolog(registered_model_name="iris_classifier")

# Train model - MLflow automatically logs everything
with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # Autologging automatically captures:
    # - Model artifacts
    # - Training parameters (n_estimators, random_state, etc.)
    # - Training metrics (score on training data)
    # - Model signature (inferred from training data)
    # - Input example

    # Optional: Log additional custom metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", accuracy)

    print(f"Run ID: {run.info.run_id}")
    print("Model automatically logged and registered!")
    