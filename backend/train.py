"""
Module d'entraînement des modèles avec MLflow
Tâche 3 - Machine Learning Avancée
"""
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import os
import time
from typing import Dict, Any, Tuple

# Configuration MLflow
MLFLOW_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_URI)

def get_model(algorithm: str, random_state: int = 42):
    """
    Retourne le modèle correspondant à l'algorithme choisi
    
    Args:
        algorithm: Nom de l'algorithme
        random_state: Graine aléatoire
        
    Returns:
        Modèle sklearn
    """
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        "SVR": SVR(kernel="rbf", C=10, epsilon=0.1),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=random_state, verbosity=0)
    }
    return models.get(algorithm, LinearRegression())

def train_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                algorithm: str, params: Dict[str, Any] = None, experiment_name: str = "cineml_experiments"):
    """
    Entraîne un modèle et log les résultats dans MLflow
    
    Args:
        X_train: Features d'entraînement
        X_test: Features de test
        y_train: Target d'entraînement
        y_test: Target de test
        algorithm: Nom de l'algorithme
        params: Hyperparamètres
        experiment_name: Nom de l'expérience MLflow
        
    Returns:
        metrics: Dictionnaire des métriques
        model: Modèle entraîné
    """
    # Créer ou récupérer l'expérience MLflow
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    start_time = time.time()
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{algorithm}_training"):
        # Logger les paramètres
        mlflow.log_param("algorithm", algorithm)
        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)
        
        # Obtenir et entraîner le modèle
        model = get_model(algorithm)
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Métriques (dans l'espace original)
        y_test_original = np.expm1(y_test)
        y_pred_original = np.expm1(y_pred)
        
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test, y_pred)
        
        # Logger les métriques
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("train_time", time.time() - start_time)
        
        # Sauvegarder le modèle
        model_path = f"../models/{algorithm}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        print(f"✅ {algorithm} - RMSE: ${rmse/1e6:.1f}M, R²: {r2:.4f}")
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "train_time": time.time() - start_time
        }
        
        return metrics, model

if __name__ == "__main__":
    # Test du module
    from data_loader import load_tmdb_data, clean_data, get_features_target
    from preprocessing import split_data, scale_data
    
    print("=" * 50)
    print("Test de train.py")
    print("=" * 50)
    
    # Chargement des données
    df = load_tmdb_data()
    df_clean = clean_data(df)
    X, y, features = get_features_target(df_clean)
    
    # Split et scaling
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, _ = scale_data(X_train, X_test)
    
    # Test avec XGBoost
    metrics, model = train_model(X_train_scaled, X_test_scaled, y_train, y_test, "XGBoost")
    print(f"Métriques: {metrics}")
