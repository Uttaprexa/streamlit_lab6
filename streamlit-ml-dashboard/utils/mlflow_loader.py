"""
MLflow utility functions for loading models and experiments
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from pathlib import Path

def setup_mlflow():
    """Setup MLflow tracking URI"""
    mlflow.set_tracking_uri("file:./mlruns")

def check_mlflow_exists():
    """Check if MLflow tracking data exists"""
    mlruns_path = Path('mlruns')
    return mlruns_path.exists() and any(mlruns_path.iterdir())

def get_all_experiments():
    """Get all MLflow experiments"""
    try:
        setup_mlflow()
        client = MlflowClient()
        experiments = client.search_experiments()
        return experiments
    except Exception as e:
        print(f"Error getting experiments: {e}")
        return []

def get_experiment_runs(experiment_id):
    """Get all runs from an experiment"""
    try:
        setup_mlflow()
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=["start_time DESC"]
        )
        return runs
    except Exception as e:
        print(f"Error getting runs: {e}")
        return pd.DataFrame()

def load_model_from_run(run_id):
    """Load a model from MLflow run"""
    try:
        setup_mlflow()
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_best_model(experiment_name, metric='accuracy'):
    """Get the best model from an experiment"""
    try:
        setup_mlflow()
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if not experiment:
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        
        if len(runs) == 0:
            return None
        
        best_run_id = runs.iloc[0]['run_id']
        return load_model_from_run(best_run_id)
        
    except Exception as e:
        print(f"Error getting best model: {e}")
        return None

def get_run_metrics(run_id):
    """Get metrics from a specific run"""
    try:
        setup_mlflow()
        client = MlflowClient()
        run = client.get_run(run_id)
        return run.data.metrics
    except Exception as e:
        print(f"Error getting metrics: {e}")
        return {}

def get_run_params(run_id):
    """Get parameters from a specific run"""
    try:
        setup_mlflow()
        client = MlflowClient()
        run = client.get_run(run_id)
        return run.data.params
    except Exception as e:
        print(f"Error getting params: {e}")
        return {}