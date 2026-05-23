"""
Script pour trouver le meilleur run automatiquement
"""

from mlflow.tracking import MlflowClient
import mlflow

MLFLOW_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

print("=" * 60)
print("RECHERCHE DU MEILLEUR MODÈLE")
print("=" * 60)

# Récupérer l'expérience
experiment = client.get_experiment_by_name("Tache5_Comparaison")

if experiment:
    print(f"\n✅ Expérience trouvée: {experiment.name}")
    
    # Rechercher les runs triés par R² décroissant
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.r2_test DESC"],
        max_results=5
    )
    
    print(f"\n📊 Classement des modèles (par R²):")
    print("-" * 70)
    
    for i, run in enumerate(runs):
        r2 = run.data.metrics.get("r2_test", 0)
        rmse = run.data.metrics.get("rmse_test_M", 0)
        n_estimators = run.data.params.get("n_estimators", "?")
        max_depth = run.data.params.get("max_depth", "?")
        
        print(f"{i+1}. {run.info.run_name}")
        print(f"   R² = {r2:.4f} | RMSE = {rmse:.1f}M$")
        print(f"   Paramètres: n={n_estimators}, depth={max_depth}")
        print()
    
    # Meilleur run
    best_run = runs[0]
    print("=" * 60)
    print("🏆 MEILLEUR MODÈLE:")
    print(f"   Run ID: {best_run.info.run_id}")
    print(f"   Run Name: {best_run.info.run_name}")
    print(f"   R²: {best_run.data.metrics.get('r2_test', 0):.4f}")
    print(f"   RMSE: {best_run.data.metrics.get('rmse_test_M', 0):.1f}M$")
    print("=" * 60)
    
else:
    print("❌ Expérience 'Tache5_Comparaison' non trouvée")
    print("   Exécutez d'abord: python3 multiple_runs.py")
