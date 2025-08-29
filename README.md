# MLflow Pipeline Project

A comprehensive machine learning pipeline project demonstrating MLflow integration for experiment tracking, model management, and deployment. This project includes various experiments with different ML frameworks and showcases best practices for ML lifecycle management.

## 🚀 Features

- **Experiment Tracking**: Comprehensive MLflow integration for tracking experiments, parameters, and metrics
- **Model Comparison**: Compare different ML algorithms (Random Forest, Neural Networks, etc.)
- **Hyperparameter Optimization**: Automated hyperparameter tuning using Hyperopt
- **Model Deployment**: Ready-to-deploy models with MLflow serving
- **Data Generation**: Synthetic data generation for reproducible experiments
- **Feature Analysis**: Feature importance analysis and model interpretability

## 📁 Project Structure

```
mlflow-pipeline/
├── wine_prediction.py          # Main wine quality prediction with MLflow
├── experiment1.py              # Basic MLflow experiment setup
├── experiment2.py              # Model training and logging
├── experiment3.py              # Parameter tracking
├── experiment4.py              # Model comparison
├── experiment4_local.py        # Local MLflow tracking
├── experiment5.py              # Artifact logging
├── experiment6.py              # Model registration
├── experiment7.py              # Model deployment
├── experiment8.py              # Custom metrics
├── experiment9.py              # Nested runs
├── experiment10.py             # Apple sales forecasting setup
├── experiment11.py             # Apple sales data generation
├── experiment12.py             # Apple sales model training
├── promote_model.py            # Model promotion utilities
├── main.py                     # Project entry point
├── .venv/                      # Virtual environment
├── mlruns/                     # MLflow tracking data
├── mlartifacts/                # MLflow artifacts
└── README.md                   # This file
```

## 🛠️ Setup

### Prerequisites

- Python 3.8+
- pip or conda
- MLflow server (optional, for remote tracking)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mlflow-pipeline
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install mlflow scikit-learn pandas numpy hyperopt matplotlib
   ```

4. **Start MLflow server (optional)**
   ```bash
   mlflow server --host 0.0.0.0 --port 8080
   ```

## 🎯 Quick Start

### Wine Quality Prediction

The main example demonstrates wine quality prediction with comprehensive MLflow integration:

```bash
python wine_prediction.py
```

This script:
- Generates synthetic wine quality data
- Trains Random Forest and MLP models
- Performs hyperparameter optimization
- Logs experiments, metrics, and models to MLflow
- Provides feature importance analysis

### Basic Experiment

Run a simple MLflow experiment:

```bash
python experiment1.py
```

## 📊 Experiments Overview

### Core Experiments

| File | Description |
|------|-------------|
| `experiment1.py` | Basic MLflow setup and experiment creation |
| `experiment2.py` | Model training with parameter and metric logging |
| `experiment3.py` | Advanced parameter tracking and model comparison |
| `experiment4.py` | Model comparison across different algorithms |
| `experiment5.py` | Artifact logging (plots, data, models) |
| `experiment6.py` | Model registration and versioning |
| `experiment7.py` | Model deployment and serving |
| `experiment8.py` | Custom metrics and evaluation |
| `experiment9.py` | Nested runs and experiment organization |

### Domain-Specific Experiments

| File | Description |
|------|-------------|
| `experiment10.py` | Apple sales forecasting experiment setup |
| `experiment11.py` | Apple sales data generation with seasonality |
| `experiment12.py` | Apple sales prediction model training |

## 🔧 Configuration

### MLflow Tracking

The project supports both local and remote MLflow tracking:

- **Local**: Uses `mlruns/` directory (default)
- **Remote**: Configure with `mlflow.set_tracking_uri("http://127.0.0.1:8080")`

### Environment Variables

```bash
# For local development
export MLFLOW_TRACKING_URI="sqlite:///mlruns.db"

# For remote tracking
export MLFLOW_TRACKING_URI="http://127.0.0.1:8080"

# For model serving (avoid pyenv issues)
export MLFLOW_CONDA_HOME=""
export MLFLOW_PYTHON_BIN=""
```

## 🚀 Model Deployment

### Serve a Model

```bash
# Serve the best model from wine prediction
mlflow models serve -m "models:/wine-quality-predictor/1" --port 5002 --env-manager local
```

### Make Predictions

```python
import mlflow
import pandas as pd

# Load the model
model = mlflow.sklearn.load_model("models:/wine-quality-predictor/1")

# Make predictions
data = pd.DataFrame({
    'fixed_acidity': [7.0],
    'volatile_acidity': [0.3],
    'citric_acid': [0.4],
    'residual_sugar': [6.0],
    'chlorides': [0.05],
    'free_sulfur_dioxide': [35],
    'total_sulfur_dioxide': [140],
    'density': [0.994],
    'pH': [3.2],
    'sulphates': [0.5],
    'alcohol': [10.5]
})

prediction = model.predict(data)
print(f"Predicted wine quality: {prediction[0]:.2f}")
```

## 📈 MLflow UI

Access the MLflow tracking UI to view experiments:

```bash
# Start MLflow UI
mlflow ui --port 5000

# Or if using remote server
mlflow ui --backend-store-uri http://127.0.0.1:8080 --port 5000
```

Navigate to `http://localhost:5000` to view:
- Experiment runs and metrics
- Model artifacts and versions
- Parameter comparisons
- Feature importance plots

## 🔍 Key Features Demonstrated

### 1. Hyperparameter Optimization
- Uses Hyperopt for Bayesian optimization
- Tests multiple model types (Random Forest, MLP)
- Logs all trials and best parameters

### 2. Model Comparison
- Compares different algorithms
- Tracks performance metrics
- Visualizes results

### 3. Feature Importance
- Analyzes feature contributions
- Logs importance scores
- Provides interpretability

### 4. Model Versioning
- Registers models with versions
- Promotes models across stages
- Manages model lifecycle

## 🛠️ Troubleshooting

### Common Issues

1. **Pyenv Error**: When serving models, use `--env-manager local`
   ```bash
   mlflow models serve -m "models:/model-name/1" --env-manager local
   ```

2. **Port Conflicts**: Change ports if needed
   ```bash
   mlflow server --port 8081
   mlflow ui --port 5001
   ```

3. **Virtual Environment**: Ensure you're in the correct environment
   ```bash
   source .venv/bin/activate
   ```

## 📚 Dependencies

- **mlflow**: Experiment tracking and model management
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **hyperopt**: Hyperparameter optimization
- **matplotlib**: Plotting and visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your experiments or improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review MLflow documentation
3. Open an issue in the repository

---

**Happy Experimenting! 🧪📊**