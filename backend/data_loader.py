"""
Module de chargement des données TMDB
Tâche 3 - Machine Learning Avancée
"""
import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, List

def load_tmdb_data(csv_path: str = "data/tmdb_5000_movies.csv") -> pd.DataFrame:
    """
    Charge le dataset TMDB depuis le fichier CSV
    
    Args:
        csv_path: Chemin vers le fichier CSV
        
    Returns:
        DataFrame pandas contenant les données
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Fichier non trouvé: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✅ Dataset chargé: {len(df)} films")
    print(f"📊 Colonnes: {list(df.columns)[:10]}...")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données TMDB
    
    Args:
        df: DataFrame brut
        
    Returns:
        DataFrame nettoyé
    """
    df_clean = df.copy()
    
    # Conversion numérique
    numeric_cols = ['budget', 'revenue', 'popularity', 'runtime', 'vote_average', 'vote_count']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    # Filtrage films valides (budget ET revenue > 1M$)
    df_clean = df_clean[(df_clean['budget'] > 1_000_000) & (df_clean['revenue'] > 1_000_000)]
    
    # Suppression des outliers extrêmes
    df_clean = df_clean[df_clean['revenue'] <= df_clean['revenue'].quantile(0.999)]
    
    print(f"✅ Après nettoyage: {len(df_clean)} films valides")
    return df_clean

def get_features_target(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extrait les features et la target
    
    Args:
        df: DataFrame nettoyé
        
    Returns:
        X: Features (numpy array)
        y: Target (numpy array)
        feature_names: Liste des noms des features
    """
    # Transformation log
    df['log_budget'] = np.log1p(df['budget'])
    df['log_revenue'] = np.log1p(df['revenue'])
    df['log_popularity'] = np.log1p(df['popularity'])
    df['log_vote_count'] = np.log1p(df['vote_count'])
    
    # Features sélectionnées
    feature_cols = [
        'log_budget', 'popularity', 'runtime', 'vote_average', 'vote_count',
        'log_popularity', 'log_vote_count'
    ]
    
    # Ajouter les colonnes disponibles
    available_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[available_cols].values
    y = df['log_revenue'].values
    
    print(f"✅ Features ({len(available_cols)}): {available_cols}")
    return X, y, available_cols

if __name__ == "__main__":
    # Test du module
    print("=" * 50)
    print("Test de data_loader.py")
    print("=" * 50)
    df = load_tmdb_data()
    df_clean = clean_data(df)
    X, y, features = get_features_target(df_clean)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
