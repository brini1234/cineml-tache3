"""
Module d'évaluation des modèles
Tâche 3 - Machine Learning Avancée
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model_path: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Évalue un modèle sauvegardé
    
    Args:
        model_path: Chemin vers le modèle .pkl
        X_test: Features de test
        y_test: Target de test
        
    Returns:
        Dictionnaire des métriques
    """
    # Charger le modèle
    model = joblib.load(model_path)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Métriques dans l'espace original
    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    
    print(f"📊 Évaluation: RMSE=${rmse/1e6:.1f}M, MAE=${mae/1e6:.1f}M, R²={r2:.4f}")
    return metrics

def compare_models(models_dir: str = "../models", X_test: np.ndarray = None, y_test: np.ndarray = None) -> pd.DataFrame:
    """
    Compare tous les modèles dans un dossier
    
    Args:
        models_dir: Dossier contenant les modèles .pkl
        X_test: Features de test
        y_test: Target de test
        
    Returns:
        DataFrame des comparaisons
    """
    results = []
    
    for model_file in os.listdir(models_dir):
        if model_file.endswith('.pkl') and model_file != 'scaler.pkl':
            model_path = os.path.join(models_dir, model_file)
            model_name = model_file.replace('.pkl', '')
            
            try:
                metrics = evaluate_model(model_path, X_test, y_test)
                results.append({
                    "model": model_name,
                    "rmse_millions": metrics["rmse"] / 1e6,
                    "mae_millions": metrics["mae"] / 1e6,
                    "r2": metrics["r2"]
                })
            except Exception as e:
                print(f"❌ Erreur avec {model_name}: {e}")
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("r2", ascending=False)
    
    print("\n" + "=" * 60)
    print("📊 COMPARAISON DES MODÈLES")
    print("=" * 60)
    print(df_results.to_string(index=False))
    
    return df_results

if __name__ == "__main__":
    from data_loader import load_tmdb_data, clean_data, get_features_target
    from preprocessing import split_data, scale_data
    
    print("=" * 50)
    print("Test de evaluate.py")
    print("=" * 50)
    
    # Chargement des données
    df = load_tmdb_data()
    df_clean = clean_data(df)
    X, y, features = get_features_target(df_clean)
    
    # Split et scaling
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, _ = scale_data(X_train, X_test)
    
    # Comparer les modèles
    compare_models("../models", X_test_scaled, y_test)
