import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import mlflow
from mlflow.models import infer_signature
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import os

# Suppress any potential logging issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Starting wine prediction with scikit-learn and MLflow...")

def generate_wine_data(n_samples=4898):
    """Generate synthetic wine quality data similar to the original dataset"""
    np.random.seed(42)
    
    # Generate features similar to wine quality dataset
    data = {
        'fixed_acidity': np.random.normal(6.85, 0.84, n_samples),
        'volatile_acidity': np.random.normal(0.28, 0.10, n_samples),
        'citric_acid': np.random.normal(0.33, 0.12, n_samples),
        'residual_sugar': np.random.normal(6.39, 5.07, n_samples),
        'chlorides': np.random.normal(0.046, 0.022, n_samples),
        'free_sulfur_dioxide': np.random.normal(35.31, 17.01, n_samples),
        'total_sulfur_dioxide': np.random.normal(138.36, 42.50, n_samples),
        'density': np.random.normal(0.994, 0.003, n_samples),
        'pH': np.random.normal(3.19, 0.15, n_samples),
        'sulphates': np.random.normal(0.49, 0.11, n_samples),
        'alcohol': np.random.normal(10.51, 1.23, n_samples),
    }
    
    # Generate quality scores (3-9) based on features
    quality = (
        5.0  # base quality
        + 0.1 * data['alcohol']
        - 0.5 * data['volatile_acidity']
        + 0.2 * data['citric_acid']
        - 0.1 * data['total_sulfur_dioxide']
        + np.random.normal(0, 0.5, n_samples)
    )
    
    # Clip quality to 3-9 range and round
    quality = np.clip(quality, 3, 9)
    quality = np.round(quality).astype(int)
    
    data['quality'] = quality
    
    return pd.DataFrame(data)

# Generate synthetic wine data
print("Generating synthetic wine data...")
data = generate_wine_data()
print(f"✓ Data generated: {data.shape}")

# Create train/validation/test splits
train, test = train_test_split(data, test_size=0.25, random_state=42)
train_x = train.drop(["quality"], axis=1).values
train_y = train[["quality"]].values.ravel()
test_x = test.drop(["quality"], axis=1).values
test_y = test[["quality"]].values.ravel()

# Further split training data for validation
train_x, valid_x, train_y, valid_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42
)

print(f"✓ Data split: train={train_x.shape}, valid={valid_x.shape}, test={test_x.shape}")

# Scale the features
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
valid_x_scaled = scaler.transform(valid_x)
test_x_scaled = scaler.transform(test_x)

# Create model signature for deployment
signature = infer_signature(train_x_scaled, train_y)


def create_and_train_model(model_type, **params):
    """
    Create and train a model with specified hyperparameters.
    """
    print(f"Training {model_type} model...")
    
    if model_type == "rf":
        model = RandomForestRegressor(**params, random_state=42)
    elif model_type == "mlp":
        model = MLPRegressor(**params, random_state=42, max_iter=200)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train the model
    model.fit(train_x_scaled, train_y)
    
    # Evaluate on validation set
    val_pred = model.predict(valid_x_scaled)
    val_rmse = np.sqrt(mean_squared_error(valid_y, val_pred))
    
    return {
        "model": model,
        "val_rmse": val_rmse,
        "model_type": model_type,
        "params": params,
    }


def objective(params):
    """
    Objective function for hyperparameter optimization.
    This function will be called by Hyperopt for each trial.
    """
    print(f"Trial with params: {params}")
    
    with mlflow.start_run(nested=True):
        # Log hyperparameters being tested
        mlflow.log_params(params)
        
        # Determine model type and extract parameters
        model_type = params.pop("model_type", "rf")  # Get model type
        print(f"Model type: {model_type}")
        
        # Filter parameters based on model type
        if model_type == "rf":
            # Only use Random Forest parameters
            rf_params = {k: v for k, v in params.items() if k in ["n_estimators", "max_depth"]}
            result = create_and_train_model(model_type, **rf_params)
        elif model_type == "mlp":
            # Only use MLP parameters
            mlp_params = {k: v for k, v in params.items() if k in ["hidden_layer_sizes", "learning_rate_init"]}
            result = create_and_train_model(model_type, **mlp_params)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        # Log training results
        mlflow.log_metrics({
            "val_rmse": result["val_rmse"],
        })
        mlflow.log_params({
            "model_type": result["model_type"],
        })
        
        # Log the trained model
        mlflow.sklearn.log_model(result["model"], name="model", signature=signature)
        
        # Log feature importance if it's a Random Forest
        if result["model_type"] == "rf":
            feature_names = data.drop("quality", axis=1).columns
            importances = result["model"].feature_importances_
            for i, (name, importance) in enumerate(zip(feature_names, importances)):
                mlflow.log_metric(f"feature_importance_{name}", importance)
        
        # Return loss for Hyperopt (it minimizes)
        print(f"Trial completed - RMSE: {result['val_rmse']:.4f}")
        return {"loss": result["val_rmse"], "status": STATUS_OK}


# Define search space for hyperparameters
search_space = {
    "model_type": hp.choice("model_type", ["rf", "mlp"]),
    "n_estimators": hp.choice("n_estimators", [50, 100, 200]),
    "max_depth": hp.choice("max_depth", [5, 10, 15, None]),
    "hidden_layer_sizes": hp.choice("hidden_layer_sizes", [(32, 16), (64, 32), (16, 8)]),
    "learning_rate_init": hp.loguniform("learning_rate_init", np.log(1e-4), np.log(1e-1)),
}

print("Search space defined:")
print("- Model types: Random Forest, MLP")
print("- Random Forest: n_estimators, max_depth")
print("- MLP: hidden_layer_sizes, learning_rate_init")

# Create or set experiment
experiment_name = "wine-quality-optimization"
mlflow.set_experiment(experiment_name)

print(f"Starting hyperparameter optimization experiment: {experiment_name}")
print("This will run 10 trials to find optimal hyperparameters...")

with mlflow.start_run(run_name="hyperparameter-sweep-sklearn"):
    # Log experiment metadata
    mlflow.log_params({
        "optimization_method": "Tree-structured Parzen Estimator (TPE)",
        "max_evaluations": 10,
        "objective_metric": "validation_rmse",
        "dataset": "synthetic-wine-quality",
        "model_framework": "scikit-learn",
    })
    
    # Run optimization
    print("Starting optimization...")
    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=trials,
        verbose=True,
    )
    
    # Find and log best results
    best_trial = min(trials.results, key=lambda x: x["loss"])
    best_rmse = best_trial["loss"]
    
    # Log optimization results
    mlflow.log_params({
        "best_model_type": best_params["model_type"],
        "best_rmse": best_rmse,
    })
    mlflow.log_metrics({
        "best_val_rmse": best_rmse,
        "total_trials": len(trials.trials),
        "optimization_completed": 1,
    })
    
    print(f"Optimization completed!")
    print(f"Best RMSE: {best_rmse:.4f}")
    print(f"Best model type: {best_params['model_type']}")
    
    # Train final model on full training data
    print("Training final model on full dataset...")
    best_model_type = ["rf", "mlp"][best_params["model_type"]]  # Convert index to string
    if best_model_type == "rf":
        final_model_params = {k: v for k, v in best_params.items() if k in ["n_estimators", "max_depth"]}
    else:
        final_model_params = {k: v for k, v in best_params.items() if k in ["hidden_layer_sizes", "learning_rate_init"]}
    final_model = create_and_train_model(best_model_type, **final_model_params)
    
    # Test on test set
    test_pred = final_model["model"].predict(test_x_scaled)
    test_rmse = np.sqrt(mean_squared_error(test_y, test_pred))
    
    mlflow.log_metric("test_rmse", test_rmse)
    print(f"Final test RMSE: {test_rmse:.4f}")
    
    # Log final model
    mlflow.sklearn.log_model(final_model["model"], name="final_model", signature=signature)
    
    print("✓ All results logged to MLflow dashboard!")