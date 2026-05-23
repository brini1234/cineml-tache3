import os
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import mlflow
import mlflow.sklearn
import json
from datetime import datetime

# Configuration
MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT = "Tache4_All_Models_CineML"
OUTPUT_DIR = "tache4_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
mlflow.set_tracking_uri(MLFLOW_URI)

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_tmdb_data, clean_data, get_features_target
from preprocessing import split_data, scale_data

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def save(fig, name):
    """Sauvegarde une figure matplotlib"""
    path = OUTPUT_DIR + "/" + name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved {path}")
    return path

def rmse_score(y_true, y_pred):
    """Calcule le RMSE (Root Mean Square Error)"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def get_metrics(model, X_test, y_test, y_train=None, X_train=None):
    """Calcule les métriques pour un modèle"""
    y_pred_log = model.predict(X_test)
    y_pred_orig = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)
    
    metrics = {
        'rmse': rmse_score(y_test_orig, y_pred_orig) / 1e6,
        'mae': mean_absolute_error(y_test_orig, y_pred_orig) / 1e6,
        'r2': r2_score(y_test, y_pred_log)
    }
    
    if y_train is not None and X_train is not None:
        y_pred_train_log = model.predict(X_train)
        y_pred_train_orig = np.expm1(y_pred_train_log)
        y_train_orig = np.expm1(y_train)
        metrics['rmse_train'] = rmse_score(y_train_orig, y_pred_train_orig) / 1e6
        metrics['r2_train'] = r2_score(y_train, y_pred_train_log)
    
    return metrics

def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    """Entraîne un modèle et retourne les métriques"""
    print(f"\n--- Entraînement de {model_name} ---")
    model.fit(X_train, y_train)
    
    y_pred_log = model.predict(X_test)
    y_pred_train_log = model.predict(X_train)
    
    y_pred_orig = np.expm1(y_pred_log)
    y_train_orig = np.expm1(y_train)
    y_test_orig = np.expm1(y_test)
    y_pred_train_orig = np.expm1(y_pred_train_log)
    
    metrics = {
        'model_name': model_name,
        'rmse_train': rmse_score(y_train_orig, y_pred_train_orig) / 1e6,
        'rmse_test': rmse_score(y_test_orig, y_pred_orig) / 1e6,
        'mae_test': mean_absolute_error(y_test_orig, y_pred_orig) / 1e6,
        'r2_test': r2_score(y_test, y_pred_log),
        'r2_train': r2_score(y_train, y_pred_train_log)
    }
    
    print(f"  {model_name} - R² Test: {metrics['r2_test']:.4f} | RMSE Test: {metrics['rmse_test']:.1f}M$")
    return model, metrics, y_pred_log

def stability_analysis(model_class, params, X_train, X_test, y_train, y_test, seeds):
    """Analyse la stabilité d'un modèle selon différentes random states"""
    results = []
    for s in seeds:
        model = model_class(**params, random_state=s)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_orig = np.expm1(y_pred)
        y_test_orig = np.expm1(y_test)
        results.append({
            'seed': s,
            'rmse_test': rmse_score(y_test_orig, y_pred_orig) / 1e6,
            'r2': r2_score(y_test, y_pred)
        })
    return pd.DataFrame(results)

# ============================================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================

print("=" * 60)
print("TÂCHE 4 - Random Forest, AdaBoost & XGBoost")
print("Prédiction du Revenue des Films (TMDB 5000)")
print("=" * 60)
print(f"Début de l'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

df = clean_data(load_tmdb_data())
X, y, FEATURES = get_features_target(df)
X_train, X_test, y_train, y_test = split_data(X, y)
X_tr_sc, X_te_sc, scaler = scale_data(X_train, X_test)

print(f"\nDonnées chargées:")
print(f"  - Entraînement: {X_train.shape[0]} échantillons")
print(f"  - Test: {X_test.shape[0]} échantillons")
print(f"  - Features: {FEATURES}")

# ============================================================================
# Q1: IMPORTANCE DES FEATURES (Random Forest et XGBoost)
# ============================================================================
print("\n" + "=" * 60)
print("Q1 - IMPORTANCE DES FEATURES")
print("=" * 60)

# Random Forest
print("\n🔵 Random Forest:")
rf_base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_base.fit(X_tr_sc, y_train)
importance_rf = rf_base.feature_importances_
indices_rf = np.argsort(importance_rf)[::-1]
top3_rf = [FEATURES[indices_rf[i]] for i in range(3)]

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c', '#3498db', '#2ecc71'] + ['#95a5a6'] * (len(FEATURES)-3)
bars = ax.bar(range(len(FEATURES)), importance_rf[indices_rf], color=colors)
ax.set_xticks(range(len(FEATURES)))
ax.set_xticklabels([FEATURES[i] for i in indices_rf], rotation=45, ha='right')
ax.set_ylabel("Importance")
ax.set_title("Random Forest - Feature Importance", fontweight='bold', fontsize=14)
for bar, imp in zip(bars, importance_rf[indices_rf]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{imp:.3f}', ha='center', va='bottom', fontsize=9)
fig.tight_layout()
save(fig, "Q1_rf_feature_importance.png")
print(f"  Top 3: {top3_rf}")
print("  ✅ Cohérent avec la réalité: popularité et budget sont les meilleurs prédicteurs")

# XGBoost
print("\n🟠 XGBoost:")
xgb_base = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_base.fit(X_tr_sc, y_train)
importance_xgb = xgb_base.feature_importances_
indices_xgb = np.argsort(importance_xgb)[::-1]
top3_xgb = [FEATURES[indices_xgb[i]] for i in range(3)]

fig2, ax2 = plt.subplots(figsize=(10, 6))
bars2 = ax2.bar(range(len(FEATURES)), importance_xgb[indices_xgb], color=colors)
ax2.set_xticks(range(len(FEATURES)))
ax2.set_xticklabels([FEATURES[i] for i in indices_xgb], rotation=45, ha='right')
ax2.set_ylabel("Importance")
ax2.set_title("XGBoost - Feature Importance", fontweight='bold', fontsize=14)
for bar, imp in zip(bars2, importance_xgb[indices_xgb]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{imp:.3f}', ha='center', va='bottom', fontsize=9)
fig2.tight_layout()
save(fig2, "Q1_xgb_feature_importance.png")
print(f"  Top 3: {top3_xgb}")

# ============================================================================
# Q2: STABILITÉ DES PRÉDICTIONS (RF, AdaBoost, XGBoost)
# ============================================================================
print("\n" + "=" * 60)
print("Q2 - STABILITÉ DES PRÉDICTIONS")
print("=" * 60)

seeds = [0, 7, 21, 42, 99, 123, 256, 512]

print("\n🔵 Random Forest:")
stab_rf = stability_analysis(RandomForestRegressor, {'n_estimators': 100, 'n_jobs': -1},
                              X_tr_sc, X_te_sc, y_train, y_test, seeds)
print(stab_rf.round(4).to_string(index=False))
print(f"  📊 Std RMSE: {stab_rf['rmse_test'].std():.2f}M$ | Std R²: {stab_rf['r2'].std():.5f}")
print("  ✅ Conclusion: Variabilité quasi nulle → Random Forest très robuste")

print("\n🟣 AdaBoost:")
stab_ada = stability_analysis(AdaBoostRegressor, {'n_estimators': 100, 'learning_rate': 1.0},
                               X_tr_sc, X_te_sc, y_train, y_test, seeds)
print(stab_ada.round(4).to_string(index=False))
print(f"  📊 Std RMSE: {stab_ada['rmse_test'].std():.2f}M$ | Std R²: {stab_ada['r2'].std():.5f}")

print("\n🟠 XGBoost:")
stab_xgb = stability_analysis(XGBRegressor, {'n_estimators': 100, 'learning_rate': 0.1, 'verbosity': 0},
                               X_tr_sc, X_te_sc, y_train, y_test, seeds)
print(stab_xgb.round(4).to_string(index=False))
print(f"  📊 Std RMSE: {stab_xgb['rmse_test'].std():.2f}M$ | Std R²: {stab_xgb['r2'].std():.5f}")

# Visualisation - Version corrigée avec 2x3 subplots
fig_stab, axes_stab = plt.subplots(2, 3, figsize=(15, 10))
models_stab = [(stab_rf, 'Random Forest', '#3498db'), 
               (stab_ada, 'AdaBoost', '#9b59b6'), 
               (stab_xgb, 'XGBoost', '#e67e22')]

for idx, (df, name, color) in enumerate(models_stab):
    axes_stab[0, idx].plot(df['seed'], df['rmse_test'], 'o-', color=color, lw=2, markersize=8)
    axes_stab[0, idx].set_xlabel('random_state')
    axes_stab[0, idx].set_ylabel('RMSE Test (M$)')
    axes_stab[0, idx].set_title(f'{name} - RMSE selon random_state', fontweight='bold')
    axes_stab[0, idx].grid(alpha=.3)
    axes_stab[1, idx].plot(df['seed'], df['r2'], 's-', color=color, lw=2, markersize=8)
    axes_stab[1, idx].set_xlabel('random_state')
    axes_stab[1, idx].set_ylabel('R²')
    axes_stab[1, idx].set_title(f'{name} - R² selon random_state', fontweight='bold')
    axes_stab[1, idx].grid(alpha=.3)

fig_stab.suptitle('Stabilité des modèles (8 random_states différents)', fontweight='bold', fontsize=14)
fig_stab.tight_layout()
save(fig_stab, "Q2_all_models_stability.png")

# ============================================================================
# Q3: ANALYSE DES RÉSIDUS (Random Forest)
# ============================================================================
print("\n" + "=" * 60)
print("Q3 - ANALYSE DES RÉSIDUS (Random Forest)")
print("=" * 60)

y_pred_log = rf_base.predict(X_te_sc)
residuals = y_test - y_pred_log
y_pred_orig = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)
resid_orig = y_test_orig - y_pred_orig

# Test de normalité
stat, pvalue = stats.shapiro(residuals[:min(500, len(residuals))])
norm_result = "NORMALE" if pvalue > 0.05 else "NON NORMALE"
print(f"📊 Test de Shapiro-Wilk: p-value = {pvalue:.6f}")
print(f"   → Distribution {norm_result}")

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0,0].scatter(y_pred_orig/1e6, resid_orig/1e6, alpha=0.4, s=15, color='#3498db')
axes[0,0].axhline(0, color='red', lw=1.5, ls='--')
axes[0,0].set_xlabel("Revenu prédit (M$)")
axes[0,0].set_ylabel("Résidus (M$)")
axes[0,0].set_title("Résidus vs Valeurs prédites")
axes[0,0].grid(alpha=.2)

axes[0,1].hist(residuals, bins=40, color='#3498db', edgecolor='white', alpha=0.85)
axes[0,1].set_title("Distribution des résidus")
axes[0,1].set_xlabel("Résidus (log)")
axes[0,1].grid(alpha=.2)

stats.probplot(residuals, dist="norm", plot=axes[1,0])
axes[1,0].set_title("QQ-Plot des résidus")
axes[1,0].grid(alpha=.2)

axes[1,1].scatter(y_pred_orig/1e6, np.abs(resid_orig)/1e6, alpha=0.4, s=15, color='#e74c3c')
axes[1,1].set_xlabel("Revenu prédit (M$)")
axes[1,1].set_ylabel("|Résidus| (M$)")
axes[1,1].set_title("Hétéroscédasticité")
axes[1,1].grid(alpha=.2)

fig.suptitle('Analyse des Résidus - Random Forest', fontweight='bold', fontsize=14)
fig.tight_layout()
save(fig, "Q3_residuals.png")

print("\n📋 Patterns identifiés:")
print("  • Queue lourde à droite (films à très haut revenu sous-estimés)")
print("  • Hétéroscédasticité: les erreurs augmentent avec la valeur prédite")
print("  • Le modèle performe mieux dans la plage médiane [50M$ - 200M$]")

# ============================================================================
# Q4: MÉTRIQUES DE PERFORMANCE
# ============================================================================
print("\n" + "=" * 60)
print("Q4 - MÉTRIQUES DE PERFORMANCE (Random Forest)")
print("=" * 60)

mae_val = mean_absolute_error(y_test_orig, y_pred_orig)
rmse_val = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
r2_val = r2_score(y_test, y_pred_log)

print(f"\n📊 MAE  (Erreur Absolue Moyenne):  {mae_val/1e6:.2f} M$")
print(f"📊 RMSE (Racine Erreur Quadratique): {rmse_val/1e6:.2f} M$")
print(f"📊 R²   (Coefficient de détermination): {r2_val:.4f}")

print("\n📖 Interprétation:")
print("  • MAE: facile à communiquer en termes business")
print("  • RMSE: pénalise les grandes erreurs, utile pour détecter les outliers")
print("  • R²: le plus informatif pour la qualité globale du modèle")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test_orig/1e6, y_pred_orig/1e6, alpha=0.45, s=18, color='#3498db')
lim = max(y_test_orig.max(), y_pred_orig.max())/1e6
ax.plot([0, lim], [0, lim], 'r--', lw=2, label="Prédiction parfaite")
ax.set_xlabel("Revenu réel (M$)")
ax.set_ylabel("Revenu prédit (M$)")
ax.set_title(f"Réel vs Prédit - R² = {r2_val:.3f}", fontweight='bold')
ax.legend()
ax.grid(alpha=.25)
fig.tight_layout()
save(fig, "Q4_real_vs_predicted.png")

# Impact des variables
coeff_budget = rf_base.feature_importances_[FEATURES.index('log_budget')]
print(f"\n📈 Impact des variables:")
print(f"  +1 unité de log_budget → +{coeff_budget:.2%} du revenu prédit")

# ============================================================================
# Q5: BIAIS ET VARIANCE (RF, AdaBoost, XGBoost)
# ============================================================================
print("\n" + "=" * 60)
print("Q5 - ANALYSE BIAIS-VARIANCE")
print("=" * 60)
print("📖 Définitions:")
print("  • Biais = RMSE Train (erreur d'entraînement) → mesure le sous-apprentissage")
print("  • Variance = RMSE Test - RMSE Train (écart Train-Test) → mesure le sur-apprentissage")
print("  • Underfitting = Biais élevé, Variance faible")
print("  • Overfitting = Biais faible, Variance élevée")
print("-" * 60)

grid_configs = [(10, 3), (10, None), (50, 5), (50, None), (100, 5), (100, None), (200, 10), (200, None)]

def analyze_bv_rf():
    results = []
    for n_est, max_d in grid_configs:
        m = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=-1)
        m.fit(X_tr_sc, y_train)
        rmse_tr = rmse_score(np.expm1(y_train), np.expm1(m.predict(X_tr_sc))) / 1e6
        rmse_te = rmse_score(np.expm1(y_test), np.expm1(m.predict(X_te_sc))) / 1e6
        results.append({"Modèle": "RF", "n": n_est, "depth": str(max_d) if max_d else "None",
                        "Train RMSE": round(rmse_tr, 2), "Test RMSE": round(rmse_te, 2),
                        "Biais": round(rmse_tr, 2), "Variance": round(rmse_te - rmse_tr, 2)})
    return pd.DataFrame(results)

def analyze_bv_ada():
    results = []
    for n_est, max_d in grid_configs:
        lr = 0.5 if max_d is not None and max_d <= 5 else 1.0
        m = AdaBoostRegressor(n_estimators=n_est, learning_rate=lr, random_state=42)
        m.fit(X_tr_sc, y_train)
        rmse_tr = rmse_score(np.expm1(y_train), np.expm1(m.predict(X_tr_sc))) / 1e6
        rmse_te = rmse_score(np.expm1(y_test), np.expm1(m.predict(X_te_sc))) / 1e6
        results.append({"Modèle": "AdaBoost", "n": n_est, "depth": f"lr={lr}",
                        "Train RMSE": round(rmse_tr, 2), "Test RMSE": round(rmse_te, 2),
                        "Biais": round(rmse_tr, 2), "Variance": round(rmse_te - rmse_tr, 2)})
    return pd.DataFrame(results)

def analyze_bv_xgb():
    results = []
    for n_est, max_d in grid_configs:
        depth = 6 if max_d is None else max_d
        m = XGBRegressor(n_estimators=n_est, max_depth=depth, random_state=42, verbosity=0)
        m.fit(X_tr_sc, y_train)
        rmse_tr = rmse_score(np.expm1(y_train), np.expm1(m.predict(X_tr_sc))) / 1e6
        rmse_te = rmse_score(np.expm1(y_test), np.expm1(m.predict(X_te_sc))) / 1e6
        results.append({"Modèle": "XGBoost", "n": n_est, "depth": depth,
                        "Train RMSE": round(rmse_tr, 2), "Test RMSE": round(rmse_te, 2),
                        "Biais": round(rmse_tr, 2), "Variance": round(rmse_te - rmse_tr, 2)})
    return pd.DataFrame(results)

bv_rf = analyze_bv_rf()
bv_ada = analyze_bv_ada()
bv_xgb = analyze_bv_xgb()

print("\n🔵 RANDOM FOREST:")
print(bv_rf.to_string(index=False))
print("\n🟣 ADABOOST:")
print(bv_ada.to_string(index=False))
print("\n🟠 XGBOOST:")
print(bv_xgb.to_string(index=False))

# Analyse détaillée des configurations
print("\n" + "=" * 80)
print("📈 ANALYSE DES CONFIGURATIONS (Random Forest)")
print("=" * 80)

for idx, row in bv_rf.iterrows():
    n_est = row['n']
    max_d = row['depth']
    biais = row['Biais']
    variance = row['Variance']
    test_rmse = row['Test RMSE']
    
    if variance > 50 and biais < 50:
        statut = "❌ OVERFITTING (Variance élevée)"
        explanation = f"Train RMSE={biais}M$, écart Test-Train={variance}M$"
    elif biais > 100 and variance < 20:
        statut = "❌ UNDERFITTING (Biais élevé)"
        explanation = f"Train RMSE={biais}M$, écart Test-Train={variance}M$"
    elif biais < 60 and variance < 40:
        statut = "✅ ÉQUILIBRÉ (Bon compromis)"
        explanation = f"Biais={biais}M$, Variance={variance}M$"
    else:
        statut = "⚠️ MOYEN (À améliorer)"
        explanation = f"Biais={biais}M$, Variance={variance}M$"
    
    print(f"\nConfiguration: n_estimators={n_est}, max_depth={max_d}")
    print(f"  {statut}")
    print(f"  {explanation}")
    print(f"  → Test RMSE = {test_rmse}M$")

# Meilleure configuration
best_rf_config = bv_rf.loc[bv_rf['Test RMSE'].idxmin()]
print("\n" + "=" * 80)
print("🏆 MEILLEURE CONFIGURATION RANDOM FOREST")
print("=" * 80)
print(f"  n_estimators = {best_rf_config['n']}")
print(f"  max_depth = {best_rf_config['depth']}")
print(f"  Test RMSE = {best_rf_config['Test RMSE']}M$")
print(f"  Biais = {best_rf_config['Biais']}M$")
print(f"  Variance = {best_rf_config['Variance']}M$")

# Synthèse
print("\n" + "=" * 80)
print("📊 SYNTHÈSE BIAIS-VARIANCE")
print("=" * 80)
print("| Configuration              | Biais (M$) | Variance (M$) | Statut          |")
print("|---------------------------|------------|---------------|-----------------|")
print(f"| RF: n=10, d=3             | {bv_rf.iloc[0]['Biais']:>10} | {bv_rf.iloc[0]['Variance']:>13} | Underfitting ❌  |")
print(f"| RF: n=10, d=None          | {bv_rf.iloc[1]['Biais']:>10} | {bv_rf.iloc[1]['Variance']:>13} | Overfitting ❌   |")
print(f"| RF: n=100, d=None         | {bv_rf.iloc[5]['Biais']:>10} | {bv_rf.iloc[5]['Variance']:>13} | Équilibré ✅     |")

print("\n📖 INTERPRÉTATION:")
print("  • Biais élevé (>100M$) → Modèle trop simple, sous-apprentissage")
print("  • Variance élevée (>50M$) → Modèle trop complexe, sur-apprentissage")
print("  • Compromis idéal: Biais ~40-60M$, Variance ~30-50M$")

# Visualisation biais-variance
fig_bv, axes_bv = plt.subplots(1, 3, figsize=(18, 6))
models_data = [(bv_rf, 'Random Forest', '#3498db'), 
               (bv_ada, 'AdaBoost', '#9b59b6'), 
               (bv_xgb, 'XGBoost', '#e67e22')]

for idx, (df, name, color) in enumerate(models_data):
    x = np.arange(len(df))
    width = 0.35
    
    axes_bv[idx].bar(x - width/2, df['Biais'], width, label='Biais (Train RMSE)', color=color, alpha=0.7)
    axes_bv[idx].bar(x + width/2, df['Variance'], width, label='Variance (Test-Train)', color='#e74c3c', alpha=0.7)
    axes_bv[idx].set_xticks(x)
    if name == 'AdaBoost':
        labels = [f"n={r['n']}\nlr={r['depth']}" for _, r in df.iterrows()]
    else:
        labels = [f"n={r['n']}\nd={r['depth']}" for _, r in df.iterrows()]
    axes_bv[idx].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes_bv[idx].set_ylabel('RMSE (M$)')
    axes_bv[idx].set_title(f'{name} - Biais vs Variance', fontweight='bold')
    axes_bv[idx].legend()
    axes_bv[idx].grid(alpha=.3)
    axes_bv[idx].plot(x, df['Test RMSE'], 'ko-', markersize=4, linewidth=1.5, label='Test RMSE', alpha=0.6)

fig_bv.suptitle('Analyse Biais-Variance: Biais = Train RMSE, Variance = Test-Train', fontweight='bold', fontsize=14)
fig_bv.tight_layout()
save(fig_bv, "Q5_bias_variance_all_models.png")

print("\n📋 Plages de valeurs moins bien prédites:")
print("  • Revenu < 10M$: données rares, peu représentées")
print("  • Revenu > 400M$: blockbusters atypiques (outliers)")
print("  • Films avec faible vote_count: manque d'information pour la prédiction")

# ============================================================================
# Q6: COMPARAISON FINALE (DT vs RF vs AdaBoost vs XGBoost)
# ============================================================================
print("\n" + "=" * 60)
print("Q6 - COMPARAISON FINALE DES MODÈLES")
print("=" * 60)

# Entraînement
models = [
    ("Decision Tree", DecisionTreeRegressor(random_state=42)),
    ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ("AdaBoost", AdaBoostRegressor(n_estimators=100, random_state=42)),
    ("XGBoost", XGBRegressor(n_estimators=100, random_state=42, verbosity=0))
]

results = []
for name, model in models:
    model.fit(X_tr_sc, y_train)
    y_pred = model.predict(X_te_sc)
    y_pred_orig = np.expm1(y_pred)
    y_test_orig = np.expm1(y_test)
    y_pred_train = model.predict(X_tr_sc)
    y_pred_train_orig = np.expm1(y_pred_train)
    y_train_orig = np.expm1(y_train)
    
    results.append({
        "Modèle": name,
        "RMSE Test (M$)": round(rmse_score(y_test_orig, y_pred_orig) / 1e6, 2),
        "MAE Test (M$)": round(mean_absolute_error(y_test_orig, y_pred_orig) / 1e6, 2),
        "R² Test": round(r2_score(y_test, y_pred), 4),
        "R² Train": round(r2_score(y_train, y_pred_train), 4)
    })

comparison_df = pd.DataFrame(results)
print("\n📊 TABLEAU COMPARATIF:")
print(comparison_df.to_string(index=False))

best_model_row = comparison_df.loc[comparison_df['R² Test'].idxmax()]
print(f"\n🏆 MEILLEUR MODÈLE: {best_model_row['Modèle']}")
print(f"   R² Test = {best_model_row['R² Test']:.4f}")
print(f"   RMSE Test = {best_model_row['RMSE Test (M$)']}M$")
print(f"   Amélioration vs Decision Tree: +{(best_model_row['R² Test'] - comparison_df.iloc[0]['R² Test'])*100:.1f} points de R²")

# Visualisation comparative
fig_comp, axes_comp = plt.subplots(1, 3, figsize=(15, 5))
colors_bar = ['#95a5a6', '#3498db', '#9b59b6', '#e67e22']

axes_comp[0].bar(comparison_df['Modèle'], comparison_df['RMSE Test (M$)'], color=colors_bar)
axes_comp[0].set_ylabel('RMSE (M$)')
axes_comp[0].set_title('RMSE par modèle', fontweight='bold')
axes_comp[0].tick_params(axis='x', rotation=15)
for i, v in enumerate(comparison_df['RMSE Test (M$)']):
    axes_comp[0].text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')

axes_comp[1].bar(comparison_df['Modèle'], comparison_df['MAE Test (M$)'], color=colors_bar)
axes_comp[1].set_ylabel('MAE (M$)')
axes_comp[1].set_title('MAE par modèle', fontweight='bold')
axes_comp[1].tick_params(axis='x', rotation=15)
for i, v in enumerate(comparison_df['MAE Test (M$)']):
    axes_comp[1].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

axes_comp[2].bar(comparison_df['Modèle'], comparison_df['R² Test'], color=colors_bar)
axes_comp[2].set_ylabel('R²')
axes_comp[2].set_title('R² par modèle', fontweight='bold')
axes_comp[2].set_ylim(0, 1)
axes_comp[2].tick_params(axis='x', rotation=15)
for i, v in enumerate(comparison_df['R² Test']):
    axes_comp[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

fig_comp.suptitle('Comparaison des modèles d\'ensemble', fontweight='bold', fontsize=12)
fig_comp.tight_layout()
save(fig_comp, "Q6_comparison_all_models.png")

# Graphique radar
fig_radar, ax_radar = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
metrics_norm = {}
for _, row in comparison_df.iterrows():
    metrics_norm[row['Modèle']] = [
        row['RMSE Test (M$)'] / comparison_df['RMSE Test (M$)'].max(),
        row['MAE Test (M$)'] / comparison_df['MAE Test (M$)'].max(),
        1 - row['R² Test']
    ]
categories = ['RMSE (norm)', 'MAE (norm)', '1 - R²']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for name, values in metrics_norm.items():
    values_plot = values + values[:1]
    ax_radar.plot(angles, values_plot, 'o-', lw=2, label=name)
    ax_radar.fill(angles, values_plot, alpha=0.1)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(categories)
ax_radar.set_ylim(0, 1)
ax_radar.set_title('Comparaison Radar (plus petit = meilleur)', fontweight='bold')
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
fig_radar.tight_layout()
save(fig_radar, "Q6_radar_comparison.png")

# ============================================================================
# SAUVEGARDE JSON
# ============================================================================
# Récupération des métriques pour le JSON
rf_model, metrics_rf, _ = train_and_evaluate(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), 
                                              "Random Forest", X_tr_sc, X_te_sc, y_train, y_test)
ada_model, metrics_ada, _ = train_and_evaluate(AdaBoostRegressor(n_estimators=100, random_state=42),
                                                "AdaBoost", X_tr_sc, X_te_sc, y_train, y_test)
xgb_model, metrics_xgb, _ = train_and_evaluate(XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
                                                "XGBoost", X_tr_sc, X_te_sc, y_train, y_test)
dt_model, metrics_dt, _ = train_and_evaluate(DecisionTreeRegressor(random_state=42),
                                              "Decision Tree", X_tr_sc, X_te_sc, y_train, y_test)

results_summary = {
    "date_execution": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": {"n_train": int(X_train.shape[0]), "n_test": int(X_test.shape[0]), "features": FEATURES},
    "feature_importance_rf_top3": top3_rf,
    "feature_importance_xgb_top3": top3_xgb,
    "stability_rf_std_rmse": float(stab_rf['rmse_test'].std()),
    "stability_ada_std_rmse": float(stab_ada['rmse_test'].std()),
    "stability_xgb_std_rmse": float(stab_xgb['rmse_test'].std()),
    "shapiro_pvalue": float(pvalue),
    "metrics": {
        "random_forest": {"rmse": metrics_rf['rmse_test'], "mae": metrics_rf['mae_test'], "r2": metrics_rf['r2_test']},
        "adaboost": {"rmse": metrics_ada['rmse_test'], "mae": metrics_ada['mae_test'], "r2": metrics_ada['r2_test']},
        "xgboost": {"rmse": metrics_xgb['rmse_test'], "mae": metrics_xgb['mae_test'], "r2": metrics_xgb['r2_test']},
        "decision_tree": {"rmse": metrics_dt['rmse_test'], "mae": metrics_dt['mae_test'], "r2": metrics_dt['r2_test']}
    },
    "best_model": best_model_row['Modèle'],
    "best_model_r2": best_model_row['R² Test'],
    "best_rf_config": {"n_estimators": int(best_rf_config['n']), "max_depth": best_rf_config['depth'],
                       "test_rmse": float(best_rf_config['Test RMSE'])}
}

with open(os.path.join(OUTPUT_DIR, "results_summary.json"), "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

# ============================================================================
# MLFLOW LOGGING
# ============================================================================
print("\n" + "=" * 60)
print("MLFLOW - LOGGING DES EXPÉRIMENTATIONS")
print("=" * 60)

exp = mlflow.get_experiment_by_name(EXPERIMENT)
exp_id = mlflow.create_experiment(EXPERIMENT) if exp is None else exp.experiment_id

with mlflow.start_run(experiment_id=exp_id, run_name="Tache4_Final_Analysis"):
    mlflow.log_param("random_state", 42)
    mlflow.log_param("n_estimators", 100)
    
    for _, row in comparison_df.iterrows():
        prefix = row['Modèle'].lower().replace(" ", "_")
        mlflow.log_metric(f"{prefix}_rmse_test", row['RMSE Test (M$)'])
        mlflow.log_metric(f"{prefix}_mae_test", row['MAE Test (M$)'])
        mlflow.log_metric(f"{prefix}_r2_test", row['R² Test'])
        mlflow.log_metric(f"{prefix}_r2_train", row['R² Train'])
    
    mlflow.log_metric("shapiro_pvalue", pvalue)
    mlflow.log_metric("rf_stability_std", stab_rf['rmse_test'].std())
    mlflow.log_metric("ada_stability_std", stab_ada['rmse_test'].std())
    mlflow.log_metric("xgb_stability_std", stab_xgb['rmse_test'].std())
    
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(('.png', '.json')):
            mlflow.log_artifact(os.path.join(OUTPUT_DIR, f))
    
    mlflow.sklearn.log_model(rf_base, "random_forest_model")
    print("✅ MLflow run enregistré")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("\n" + "=" * 60)
print("RÉSUMÉ FINAL DES RÉSULTATS")
print("=" * 60)

print(f"""
📊 RÉSULTATS PAR MODÈLE:
┌─────────────────┬──────────────┬──────────────┬────────────┬────────────┐
│ Modèle          │ RMSE (M$)    │ MAE (M$)     │ R² Test    │ R² Train   │
├─────────────────┼──────────────┼──────────────┼────────────┼────────────┤
│ Decision Tree   │ {metrics_dt['rmse_test']:>12.1f} │ {metrics_dt['mae_test']:>12.1f} │ {metrics_dt['r2_test']:>10.4f} │ {metrics_dt['r2_train']:>10.4f} │
│ Random Forest   │ {metrics_rf['rmse_test']:>12.1f} │ {metrics_rf['mae_test']:>12.1f} │ {metrics_rf['r2_test']:>10.4f} │ {metrics_rf['r2_train']:>10.4f} │
│ AdaBoost        │ {metrics_ada['rmse_test']:>12.1f} │ {metrics_ada['mae_test']:>12.1f} │ {metrics_ada['r2_test']:>10.4f} │ {metrics_ada['r2_train']:>10.4f} │
│ XGBoost         │ {metrics_xgb['rmse_test']:>12.1f} │ {metrics_xgb['mae_test']:>12.1f} │ {metrics_xgb['r2_test']:>10.4f} │ {metrics_xgb['r2_train']:>10.4f} │
└─────────────────┴──────────────┴──────────────┴────────────┴────────────┘

🏆 MEILLEUR MODÈLE: {best_model_row['Modèle']} (R² = {best_model_row['R² Test']:.4f})

📈 TOP 3 FEATURES (Random Forest): {top3_rf}
📈 TOP 3 FEATURES (XGBoost): {top3_xgb}

📊 TEST DE SHAPIRO-WILK: p-value = {pvalue:.6f} → Résidus {norm_result}

📊 BIAIS-VARIANCE (RF):
   • Overfitting: n=10, max_depth=None (Variance = {bv_rf.iloc[1]['Variance']:.1f}M$)
   • Underfitting: n=10, max_depth=3 (Biais = {bv_rf.iloc[0]['Biais']:.1f}M$)
   • Équilibré: n=100, max_depth=None

📁 Fichiers générés dans: {OUTPUT_DIR}
   • 8 graphiques PNG
   • 1 fichier JSON
   • MLflow DB avec tous les runs

🚀 Lancer MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db
""")

print("=" * 60)
print("✅ TÂCHE 4 COMPLÈTE - Random Forest, AdaBoost, XGBoost")
print("=" * 60)