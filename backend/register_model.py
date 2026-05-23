"""
Script pour enregistrer le meilleur modèle dans MLflow Registry
"""

from mlflow.tracking import MlflowClient
import mlflow

MLFLOW_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

MODEL_NAME = "CineML_Revenue_Predictor"

print("=" * 60)
print("MODEL REGISTRY - ENREGISTREMENT")
print("=" * 60)

# 1. Récupérer le meilleur run
experiment = client.get_experiment_by_name("Tache5_Comparaison")

if not experiment:
    print("❌ Expérience non trouvée. Exécutez d'abord multiple_runs.py")
    exit()

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.r2_test DESC"],
    max_results=1
)

if not runs:
    print("❌ Aucun run trouvé")
    exit()

best_run = runs[0]
best_run_id = best_run.info.run_id
best_r2 = best_run.data.metrics.get("r2_test", 0)

print(f"\n📊 Meilleur run trouvé:")
print(f"   ID: {best_run_id}")
print(f"   R²: {best_r2:.4f}")

# 2. Enregistrer le modèle
print(f"\n📝 Enregistrement du modèle '{MODEL_NAME}'...")
model_uri = f"runs:/{best_run_id}/random_forest_model"

try:
    registered = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"✅ Modèle enregistré - Version: {registered.version}")
except Exception as e:
    print(f"⚠️ Le modèle existe peut-être déjà. Version suivante...")
    # Récupérer la dernière version
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    next_version = int(versions[-1].version) + 1 if versions else 1
    registered = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"✅ Nouvelle version créée: {registered.version}")

# 3. Ajouter description et tags
client.update_registered_model(
    name=MODEL_NAME,
    description="Modèle Random Forest pour prédire le revenu des films (TMDB 5000)"
)

client.set_model_version_tag(
    name=MODEL_NAME,
    version=registered.version,
    key="validated_by",
    value="equipe_data"
)

client.set_model_version_tag(
    name=MODEL_NAME,
    version=registered.version,
    key="r2_score",
    value=str(round(best_r2, 4))
)

print("✅ Description et tags ajoutés")

# 4. Promouvoir en Staging
print(f"\n📤 Promotion en Staging...")
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=registered.version,
    stage="Staging",
    archive_existing_versions=False
)
print("✅ Modèle promu en Staging")

# 5. Vérifier seuil pour Production
SEUIL_PRODUCTION = 0.60  # Seuil pour R²

print(f"\n📊 Vérification du seuil de production:")
print(f"   R² du modèle: {best_r2:.4f}")
print(f"   Seuil requis: {SEUIL_PRODUCTION}")

if best_r2 >= SEUIL_PRODUCTION:
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=registered.version,
        stage="Production"
    )
    print(f"✅ Modèle v{registered.version} promu en Production!")
else:
    print(f"❌ Modèle non promu: R² {best_r2:.3f} < seuil {SEUIL_PRODUCTION}")

print("\n" + "=" * 60)
print("✅ MODEL REGISTRY - TERMINÉ")
print("=" * 60)
