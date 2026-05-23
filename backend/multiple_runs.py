"""
Script pour lancer plusieurs configurations de Random Forest
et les comparer dans MLflow
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configuration
MLFLOW_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_URI)
EXPERIMENT = "Tache5_Comparaison"

# Importer nos fonctions
from data_loader import load_tmdb_data, clean_data, get_features_target
from preprocessing import split_data, scale_data

print("=" * 60)
print("TÂCHE 5 - COMPARAISON DE MODÈLES")
print("=" * 60)

# Chargement des données
print("\n📊 Chargement des données...")
df = clean_data(load_tmdb_data())
X, y, FEATURES = get_features_target(df)
X_train, X_test, y_train, y_test = split_data(X, y)
X_tr_sc, X_te_sc, scaler = scale_data(X_train, X_test)

print(f"✅ Données prêtes: {X_train.shape[0]} train, {X_test.shape[0]} test")

# Configurations à tester
configs = [
    {"name": "rf_baseline", "n_estimators": 50, "max_depth": 3},
    {"name": "rf_deep", "n_estimators": 200, "max_depth": 10},
    {"name": "rf_large", "n_estimators": 300, "max_depth": 15},
]

# Créer l'expérience
mlflow.set_experiment(EXPERIMENT)

print("\n🚀 Lancement des expérimentations...")
print("-" * 60)

# Boucle sur les configurations
for cfg in configs:
    print(f"\n🔵 Test: {cfg['name']}")
    print(f"   n_estimators={cfg['n_estimators']}, max_depth={cfg['max_depth']}")
    
    with mlflow.start_run(run_name=cfg["name"]):
        # Log des paramètres
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", cfg["n_estimators"])
        mlflow.log_param("max_depth", cfg["max_depth"])
        mlflow.log_param("random_state", 42)
        
        # Entraînement
        model = RandomForestRegressor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_tr_sc, y_train)
        
        # Prédictions
        y_pred = model.predict(X_te_sc)
        y_pred_orig = np.expm1(y_pred)
        y_test_orig = np.expm1(y_test)
        
        # Calcul des métriques
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)) / 1e6
        mae = mean_absolute_error(y_test_orig, y_pred_orig) / 1e6
        r2 = r2_score(y_test, y_pred)
        
        # Log des métriques
        mlflow.log_metric("rmse_test_M", rmse)
        mlflow.log_metric("mae_test_M", mae)
        mlflow.log_metric("r2_test", r2)
        
        # Log du modèle
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print(f"   ✅ R² = {r2:.4f} | RMSE = {rmse:.1f}M$")

print("\n" + "=" * 60)
print("✅ Toutes les expérimentations sont terminées!")
print("📊 Lancez MLflow UI pour voir les résultats:")
print("   mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("=" * 60)
