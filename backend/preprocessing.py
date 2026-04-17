"""
Module de prétraitement des données
Tâche 3 - Machine Learning Avancée
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """
    Divise les données en train/test
    
    Args:
        X: Features
        y: Target
        test_size: Proportion pour le test
        random_state: Graine aléatoire
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"✅ Split: Train={len(X_train)} | Test={len(X_test)}")
    return X_train, X_test, y_train, y_test

def scale_data(X_train: np.ndarray, X_test: np.ndarray, scaler_path: str = "../models/scaler.pkl"):
    """
    Standardise les données
    
    Args:
        X_train: Features d'entraînement
        X_test: Features de test
        scaler_path: Chemin pour sauvegarder le scaler
        
    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Sauvegarder le scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    
    print(f"✅ Scaling terminé, scaler sauvegardé dans {scaler_path}")
    return X_train_scaled, X_test_scaled, scaler

if __name__ == "__main__":
    # Test du module
    from data_loader import load_tmdb_data, clean_data, get_features_target
    
    print("=" * 50)
    print("Test de preprocessing.py")
    print("=" * 50)
    
    df = load_tmdb_data()
    df_clean = clean_data(df)
    X, y, features = get_features_target(df_clean)
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    print(f"Train scaled: {X_train_scaled.shape}")
    print(f"Test scaled: {X_test_scaled.shape}")
