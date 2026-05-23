"""
Serveur API simple pour le modèle Random Forest
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")

print("=" * 50)
print("🚀 DÉMARRAGE DU SERVEUR")
print("=" * 50)

# Charger le modèle depuis MLflow
print("\n📊 Chargement du modèle...")

# Récupérer le meilleur run
from mlflow.tracking import MlflowClient
client = MlflowClient()

exp = client.get_experiment_by_name('Tache5_Comparaison')
if exp:
    runs = client.search_runs(exp.experiment_id, order_by=['metrics.r2_test DESC'], max_results=1)
    if runs:
        best_run = runs[0]
        run_id = best_run.info.run_id
        print(f"✅ Run trouvé: {best_run.info.run_name}")
        print(f"   Run ID: {run_id}")
        print(f"   R²: {best_run.data.metrics.get('r2_test', 0):.4f}")
        
        # Charger le modèle
        model_uri = f"runs:/{run_id}/random_forest_model"
        model = mlflow.sklearn.load_model(model_uri)
        print("✅ Modèle chargé avec succès!")
    else:
        print("❌ Aucun run trouvé")
        exit()
else:
    print("❌ Expérience Tache5_Comparaison non trouvée")
    exit()

print("\n" + "=" * 50)
print("🌐 Serveur démarré sur http://localhost:1234")
print("📋 Endpoint: POST /invocations")
print("=" * 50)

@app.route('/invocations', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Format attendu par MLflow
        if 'dataframe_split' in data:
            columns = data['dataframe_split']['columns']
            values = data['dataframe_split']['data']
            df_input = pd.DataFrame(values, columns=columns)
        else:
            df_input = pd.DataFrame(data['data'])
        
        # Prédiction
        predictions = model.predict(df_input)
        
        return jsonify({"predictions": predictions.tolist()})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1234, debug=False, threaded=True)
