import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import mlflow
import mlflow.sklearn

MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT = "Tache4_RandomForest"
OUTPUT_DIR = "tache4_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
mlflow.set_tracking_uri(MLFLOW_URI)

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_tmdb_data, clean_data, get_features_target
from preprocessing import split_data, scale_data

def save(fig, name):
    path = OUTPUT_DIR + "/" + name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("saved " + path)
    return path

def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print("=" * 60)
print("TACHE 4 - Random Forest CineML")
print("=" * 60)

df = clean_data(load_tmdb_data())
X, y, FEATURES = get_features_target(df)
X_train, X_test, y_train, y_test = split_data(X, y)
X_tr_sc, X_te_sc, scaler = scale_data(X_train, X_test)

print("\n[Q1] Feature Importance")
rf_base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_base.fit(X_tr_sc, y_train)
importance = rf_base.feature_importances_
indices = np.argsort(importance)[::-1]
top3 = [FEATURES[indices[i]] for i in range(3)]

fig, ax = plt.subplots(figsize=(9, 5))
colors_map = {indices[0]: "#e74c3c", indices[1]: "#3498db", indices[2]: "#2ecc71"}
bar_colors = [colors_map.get(i, "#95a5a6") for i in indices]
bars = ax.bar(range(len(FEATURES)), importance[indices], color=bar_colors)
ax.set_xticks(range(len(FEATURES)))
ax.set_xticklabels([FEATURES[i] for i in indices], rotation=35, ha='right', fontsize=10)
ax.set_ylabel("Importance (Gini)")
ax.set_title("Random Forest - Feature Importance", fontweight='bold')
for bar, imp in zip(bars, importance[indices]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            str(round(imp, 3)), ha='center', va='bottom', fontsize=9)
ax.set_ylim(0, max(importance) * 1.15)
fig.tight_layout()
save(fig, "Q1_feature_importance.png")
print("Top 3: " + str(top3))

print("\n[Q2] Stabilite - random_state varies")
seeds = [0, 7, 21, 42, 99, 123, 256, 512]
stability = []
for s in seeds:
    m = RandomForestRegressor(n_estimators=100, random_state=s, n_jobs=-1)
    m.fit(X_tr_sc, y_train)
    yp = m.predict(X_te_sc)
    yt = m.predict(X_tr_sc)
    stability.append({"seed": s,
        "rmse_train": rmse_score(np.expm1(y_train), np.expm1(yt)),
        "rmse_test": rmse_score(np.expm1(y_test), np.expm1(yp)),
        "r2": r2_score(y_test, yp)})
stab_df = pd.DataFrame(stability)
print(stab_df.to_string(index=False))
print("Std RMSE test: " + str(round(stab_df['rmse_test'].std(), 0)))
print("Std R2: " + str(round(stab_df['r2'].std(), 5)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(stab_df["seed"], stab_df["rmse_test"]/1e6, 'o-', color="#3498db", lw=2)
ax1.set_xlabel("random_state"); ax1.set_ylabel("RMSE Test (M$)")
ax1.set_title("RMSE Test selon random_state"); ax1.grid(alpha=.3)
ax2.plot(stab_df["seed"], stab_df["r2"], 's-', color="#e74c3c", lw=2)
ax2.set_xlabel("random_state"); ax2.set_ylabel("R2")
ax2.set_title("R2 selon random_state"); ax2.grid(alpha=.3)
fig.suptitle("Stabilite du Random Forest", fontweight='bold')
fig.tight_layout()
save(fig, "Q2_stability.png")

print("\n[Q3] Analyse des residus")
y_pred_log = rf_base.predict(X_te_sc)
residuals = y_test - y_pred_log
y_pred_orig = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)
resid_orig = y_test_orig - y_pred_orig
stat, pvalue = stats.shapiro(residuals[:min(500, len(residuals))])
norm_result = "NORMAL" if pvalue > 0.05 else "NON normal"
print("Shapiro p-value: " + str(round(pvalue, 4)) + " (" + norm_result + ")")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes[0,0].scatter(y_pred_orig/1e6, resid_orig/1e6, alpha=0.4, s=15, color="#3498db")
axes[0,0].axhline(0, color='red', lw=1.5, ls='--')
axes[0,0].set_xlabel("Revenue predit (M$)"); axes[0,0].set_ylabel("Residus (M$)")
axes[0,0].set_title("Residus vs Valeurs predites"); axes[0,0].grid(alpha=.2)
axes[0,1].hist(residuals, bins=40, color="#3498db", edgecolor='white', alpha=0.85)
xr = np.linspace(residuals.min(), residuals.max(), 200)
axes[0,1].plot(xr, stats.norm.pdf(xr, residuals.mean(), residuals.std()) * len(residuals) * (residuals.max()-residuals.min())/40, 'r-', lw=2)
axes[0,1].set_title("Distribution des residus"); axes[0,1].grid(alpha=.2)
(osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
axes[1,0].plot(osm, osr, '.', alpha=0.4, markersize=4, color="#2ecc71")
axes[1,0].plot(osm, slope*np.array(osm)+intercept, 'r-', lw=2)
axes[1,0].set_title("QQ-Plot des residus"); axes[1,0].grid(alpha=.2)
axes[1,1].scatter(y_pred_orig/1e6, np.abs(resid_orig)/1e6, alpha=0.4, s=15, color="#e74c3c")
axes[1,1].set_title("Heteroscedasticite"); axes[1,1].grid(alpha=.2)
fig.suptitle("Analyse des Residus - Random Forest", fontweight='bold', fontsize=13)
fig.tight_layout()
save(fig, "Q3_residuals.png")

print("\n[Q4] Metriques de performance")
mae_val = mean_absolute_error(y_test_orig, y_pred_orig)
mse_val = mean_squared_error(y_test_orig, y_pred_orig)
r2_val = r2_score(y_test, y_pred_log)
rmse_val = np.sqrt(mse_val)
print("MAE  : " + str(round(mae_val/1e6, 2)) + " M$")
print("RMSE : " + str(round(rmse_val/1e6, 2)) + " M$")
print("R2   : " + str(round(r2_val, 4)))

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test_orig/1e6, y_pred_orig/1e6, alpha=0.45, s=18, color="#3498db")
lim = max(y_test_orig.max(), y_pred_orig.max())/1e6
ax.plot([0, lim], [0, lim], 'r--', lw=2, label="Prediction parfaite")
ax.set_xlabel("Revenue reel (M$)"); ax.set_ylabel("Revenue predit (M$)")
ax.set_title("Reel vs Predit - R2=" + str(round(r2_val, 3)), fontweight='bold')
ax.legend(); ax.grid(alpha=.25)
fig.tight_layout()
save(fig, "Q4_real_vs_predicted.png")

print("\n[Q5] Biais-Variance")
grid = [(10,3),(10,None),(50,5),(50,None),(100,5),(100,None),(200,10),(200,None)]
bv_rows = []
for n_est, max_d in grid:
    m = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=-1)
    m.fit(X_tr_sc, y_train)
    rmse_tr = rmse_score(np.expm1(y_train), np.expm1(m.predict(X_tr_sc)))
    rmse_te = rmse_score(np.expm1(y_test), np.expm1(m.predict(X_te_sc)))
    bv_rows.append({"n_estimators": n_est, "max_depth": str(max_d),
        "Train RMSE (M$)": round(rmse_tr/1e6, 2), "Test RMSE (M$)": round(rmse_te/1e6, 2),
        "Biais (M$)": round(rmse_tr/1e6, 2), "Variance (M$)": round((rmse_te-rmse_tr)/1e6, 2)})
bv_df = pd.DataFrame(bv_rows)
print(bv_df.to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
labels = ["n="+str(r['n_estimators'])+"\nd="+str(r['max_depth']) for _, r in bv_df.iterrows()]
x = np.arange(len(labels)); w = 0.35
ax1.bar(x-w/2, bv_df["Train RMSE (M$)"], w, label="Train RMSE", color="#3498db")
ax1.bar(x+w/2, bv_df["Test RMSE (M$)"], w, label="Test RMSE", color="#e74c3c")
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=7.5)
ax1.set_ylabel("RMSE (M$)"); ax1.legend(); ax1.grid(axis='y', alpha=.3)
ax1.set_title("Train vs Test RMSE")
ax2.bar(x-w/2, bv_df["Biais (M$)"], w, label="Biais", color="#9b59b6")
ax2.bar(x+w/2, bv_df["Variance (M$)"], w, label="Variance", color="#f39c12")
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7.5)
ax2.set_ylabel("RMSE (M$)"); ax2.legend(); ax2.grid(axis='y', alpha=.3)
ax2.set_title("Biais vs Variance")
fig.suptitle("Analyse Biais-Variance - Random Forest", fontweight='bold', fontsize=13)
fig.tight_layout()
save(fig, "Q5_bias_variance.png")

print("\n[Q6] Comparaison RF vs Decision Tree")
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_tr_sc, y_train)
dt_r2 = r2_score(y_test, dt.predict(X_te_sc))
dt_rmse = rmse_score(np.expm1(y_test), np.expm1(dt.predict(X_te_sc)))
dt_mae = mean_absolute_error(np.expm1(y_test), np.expm1(dt.predict(X_te_sc)))
rf_rmse_te = rmse_score(np.expm1(y_test), y_pred_orig)
comparison = pd.DataFrame({"Modele": ["Decision Tree", "Random Forest"],
    "Test RMSE (M$)": [round(dt_rmse/1e6,2), round(rf_rmse_te/1e6,2)],
    "MAE (M$)": [round(dt_mae/1e6,2), round(mae_val/1e6,2)],
    "R2": [round(dt_r2,4), round(r2_val,4)]})
print(comparison.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(2); w = 0.35
axes[0].bar(x-w/2, [dt_rmse/1e6, dt_mae/1e6], w, label="Decision Tree", color="#95a5a6")
axes[0].bar(x+w/2, [rf_rmse_te/1e6, mae_val/1e6], w, label="Random Forest", color="#3498db")
axes[0].set_xticks(x); axes[0].set_xticklabels(["RMSE", "MAE"])
axes[0].set_ylabel("Erreur (M$)"); axes[0].legend(); axes[0].set_title("Erreurs DT vs RF")
axes[1].bar(["Decision Tree", "Random Forest"], [dt_r2, r2_val], color=["#95a5a6","#3498db"])
axes[1].set_ylabel("R2"); axes[1].set_title("R2 DT vs RF"); axes[1].set_ylim(0, 1)
fig.suptitle("Comparaison Decision Tree vs Random Forest", fontweight='bold', fontsize=13)
fig.tight_layout()
save(fig, "Q6_comparison_DT_RF.png")

print("\n[MLflow] Logging...")
exp = mlflow.get_experiment_by_name(EXPERIMENT)
exp_id = mlflow.create_experiment(EXPERIMENT) if exp is None else exp.experiment_id
with mlflow.start_run(experiment_id=exp_id, run_name="Tache4_RF_Analysis"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("max_depth", "None")
    mlflow.log_metric("rmse_test_M", round(rmse_val/1e6, 4))
    mlflow.log_metric("mae_test_M", round(mae_val/1e6, 4))
    mlflow.log_metric("r2_test", round(r2_val, 4))
    mlflow.log_metric("shapiro_pval", round(float(pvalue), 4))
    mlflow.log_metric("stability_rmse_std", round(stab_df['rmse_test'].std(), 0))
    for f in os.listdir(OUTPUT_DIR):
        mlflow.log_artifact(OUTPUT_DIR + "/" + f)
    mlflow.sklearn.log_model(rf_base, "random_forest_model")
    print("MLflow run enregistre")

print("\n" + "="*60)
print("RESUME DES RESULTATS")
print("="*60)
print("Q1 Top 3: " + str(top3))
print("Q2 Std RMSE: " + str(round(stab_df['rmse_test'].std(),0)) + "$ | Std R2: " + str(round(stab_df['r2'].std(),5)))
print("Q3 Shapiro p=" + str(round(pvalue,4)) + " => " + norm_result)
print("Q4 MAE=" + str(round(mae_val/1e6,1)) + "M RMSE=" + str(round(rmse_val/1e6,1)) + "M R2=" + str(round(r2_val,4)))
print("Q5 Overfitting: n=10 max_depth=None | Underfitting: n=10 max_depth=3 | Equilibre: n=100 max_depth=None")
print("Q6 RF R2=" + str(round(r2_val,4)) + " vs DT R2=" + str(round(dt_r2,4)))
print("Graphiques dans: " + OUTPUT_DIR)
print("="*60)
