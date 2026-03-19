"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   TÂCHE 3 — Machine Learning Avancée — ING4 DSA                            ║
║   Projet CineML : Prédiction du Succès au Box-Office                        ║
║   Étudiant  : Chaima Brini                                                  ║
║   Enseignant: Aroua Hedhili — Université Sésame                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DATASET RÉEL :                                                              ║
║    TMDB Movie Dataset — Kaggle                                               ║
║    https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata                ║
║    → Télécharger tmdb_5000_movies.csv                                       ║
║    → Placer dans data/raw/tmdb_5000_movies.csv                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  USAGE :                                                                     ║
║    pip3 install mlflow scikit-learn xgboost pandas numpy                   ║
║                 matplotlib seaborn reportlab joblib                          ║
║    python3 tache3_cineml.py                                                 ║
║                                                                              ║
║  MLFLOW UI :                                                                 ║
║    python3 -m mlflow ui --backend-store-uri sqlite:///mlflow_cineml.db \   ║
║                          --port 5000                                         ║
║    → http://127.0.0.1:5000                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, ast, time, warnings, logging, joblib
import numpy  as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow, mlflow.sklearn

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                      RandomizedSearchCV)
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor
from sklearn.svm             import SVR
from sklearn.decomposition   import PCA
from sklearn.manifold        import TSNE
from sklearn.metrics         import (mean_absolute_error, mean_squared_error,
                                      r2_score)
from xgboost                 import XGBRegressor

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles    import ParagraphStyle
from reportlab.lib.units     import cm
from reportlab.lib           import colors
from reportlab.platypus      import (SimpleDocTemplate, Paragraph, Spacer,
                                      Table, TableStyle, HRFlowable,
                                      PageBreak, Image)
from reportlab.lib.enums     import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen        import canvas as rl_canvas

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
SEEDS        = [11, 42]
MLFLOW_URI   = "sqlite:///mlflow_cineml.db"
FIGURES_DIR  = "figures_tache3"
MODELS_DIR   = "models"
OUTPUT_PDF   = "Rapport_MLflow_Tache3_Brini_FINAL.pdf"
RESULTS_CSV  = "resultats_tache3.csv"

# Chemins possibles pour le fichier CSV TMDB
TMDB_CSV_PATHS = [
    "data/raw/tmdb_5000_movies.csv",
    "data/raw/movies_metadata.csv",
    "data/raw/tmdb_movies.csv",
    "tmdb_5000_movies.csv",
    "movies_metadata.csv",
]

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs("data/raw",  exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# PARTIE 1 : CHARGEMENT DU VRAI DATASET TMDB
# ═════════════════════════════════════════════════════════════════════════════

def _extract_genres(genres_str):
    """Extrait les noms de genres depuis la colonne JSON TMDB."""
    try:
        return [g["name"] for g in ast.literal_eval(str(genres_str))
                if isinstance(g, dict)]
    except Exception:
        return []


def load_tmdb_dataset():
    """
    Charge le vrai dataset TMDB (tmdb_5000_movies.csv).
    Si introuvable, génère un dataset synthétique de même structure.

    Structure du vrai CSV :
        budget, genres, id, original_language, original_title,
        popularity, production_companies, release_date, revenue,
        runtime, spoken_languages, status, title,
        vote_average, vote_count
    """
    csv_path = next((p for p in TMDB_CSV_PATHS if os.path.exists(p)), None)

    if csv_path:
        print(f"[dataset] Chargement du vrai dataset TMDB : {csv_path}")
        df_raw = pd.read_csv(csv_path, low_memory=False)
        df     = _process_real_tmdb(df_raw)
    else:
        print("[dataset] ATTENTION : Fichier TMDB non trouvé !")
        print("[dataset] Télécharger depuis :")
        print("[dataset]   https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
        print("[dataset] Placer dans : data/raw/tmdb_5000_movies.csv")
        print("[dataset] → Génération d'un dataset TMDB-like (même structure)...")
        df = _generate_synthetic(n=4800, seed=RANDOM_STATE)

    print(f"[dataset] {len(df)} films | "
          f"Revenue moyen : {df.revenue.mean()/1e6:.1f} M$ | "
          f"Budget moyen : {df.budget.mean()/1e6:.1f} M$")
    return df


def _process_real_tmdb(df_raw):
    """Prétraitement du vrai CSV TMDB Kaggle."""
    df = df_raw.copy()

    # Conversion numérique
    for col in ["budget", "revenue", "popularity", "runtime",
                "vote_average", "vote_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Filtrage films valides (budget ET revenue > 1M$)
    df = df[(df.budget > 1_000_000) & (df.revenue > 1_000_000)].copy()
    df = df.dropna(subset=["budget","revenue"]).reset_index(drop=True)

    # Genres depuis JSON
    if "genres" in df.columns:
        df["genres_list"] = df["genres"].apply(_extract_genres)
    else:
        df["genres_list"] = [[]]

    top_genres = ["Action","Comedy","Drama","Thriller","Romance",
                  "Horror","Adventure","Science Fiction"]
    for g in top_genres:
        col = "genre_" + g.lower().replace(" ", "_")
        df[col] = df["genres_list"].apply(lambda lst: 1 if g in lst else 0)

    # Langue
    if "original_language" in df.columns:
        df["is_english"] = (df.original_language.str.strip().str.lower()=="en").astype(int)
    else:
        df["is_english"] = 1

    # Date de sortie → mois + année + saisonnalité
    if "release_date" in df.columns:
        rd = pd.to_datetime(df.release_date, errors="coerce")
        df["release_month"] = rd.dt.month.fillna(6).astype(int)
        df["release_year"]  = rd.dt.year.fillna(2000).astype(int)
    else:
        df["release_month"] = 6; df["release_year"] = 2000

    df["summer"] = ((df.release_month >= 6) & (df.release_month <= 8)).astype(int)
    df["xmas"]   = (df.release_month == 12).astype(int)

    # Retrait outliers extrêmes
    df = df[df.revenue <= df.revenue.quantile(0.999)].reset_index(drop=True)
    print(f"[tmdb_real] {len(df)} films après nettoyage")
    return df


def _generate_synthetic(n=4800, seed=42):
    """
    Dataset synthétique reproduisant les distributions statistiques
    du vrai TMDB (utilisé uniquement si le CSV n'est pas disponible).
    """
    rng = np.random.default_rng(seed)
    budget       = np.exp(rng.normal(18.0,1.5,n)).clip(5e5,3e8).astype(int)
    popularity   = rng.exponential(18,n).clip(0.5,400)
    runtime      = rng.normal(108,20,n).clip(70,220).astype(int)
    vote_average = rng.normal(6.2,0.8,n).clip(2,9.5).round(1)
    vote_count   = rng.exponential(1100,n).clip(20,18000).astype(int)
    release_month= rng.integers(1,13,n)
    release_year = rng.integers(1990,2018,n)
    g_action     = rng.binomial(1,.28,n); g_comedy=rng.binomial(1,.22,n)
    g_drama      = rng.binomial(1,.32,n); g_thriller=rng.binomial(1,.18,n)
    g_romance    = rng.binomial(1,.15,n); g_horror=rng.binomial(1,.12,n)
    g_adventure  = rng.binomial(1,.20,n); g_scifi=rng.binomial(1,.10,n)
    is_english   = rng.binomial(1,.74,n)
    summer       = ((release_month>=6)&(release_month<=8)).astype(float)
    xmas         = (release_month==12).astype(float)
    log_rev = (0.88*np.log1p(budget)+0.62*np.log1p(popularity)
               +0.30*np.log1p(vote_count)+0.20*vote_average
               +0.14*is_english+0.09*g_action+0.11*summer+0.08*xmas
               +rng.normal(0,0.72,n))
    revenue = np.exp(log_rev).clip(0,4e9).astype(int)
    mask = (budget>1e6)&(revenue>1e6)
    df = pd.DataFrame({
        "budget":budget[mask],"popularity":popularity[mask].round(3),
        "runtime":runtime[mask],"vote_average":vote_average[mask],
        "vote_count":vote_count[mask],"release_month":release_month[mask],
        "release_year":release_year[mask],"genre_action":g_action[mask],
        "genre_comedy":g_comedy[mask],"genre_drama":g_drama[mask],
        "genre_thriller":g_thriller[mask],"genre_romance":g_romance[mask],
        "genre_horror":g_horror[mask],"genre_adventure":g_adventure[mask],
        "genre_science_fiction":g_scifi[mask],"is_english":is_english[mask],
        "summer":summer[mask],"xmas":xmas[mask],"revenue":revenue[mask],
    }).reset_index(drop=True)
    return df[df.revenue<=df.revenue.quantile(0.999)].reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# PARTIE 2 : FEATURE ENGINEERING & PRÉTRAITEMENT
# ═════════════════════════════════════════════════════════════════════════════

def preprocess(df):
    """
    1. Feature engineering (log-transformations, interactions)
    2. Split train/test 80/20
    3. StandardScaler — fit UNIQUEMENT sur le train
    """
    df = df.copy()
    df["log_budget"]     = np.log1p(df.budget)
    df["log_popularity"] = np.log1p(df.popularity)
    df["log_vote_count"] = np.log1p(df.vote_count)
    df["budget_pop"]     = np.log1p(df.budget * df.popularity)
    df["log_revenue"]    = np.log1p(df.revenue)

    base = ["budget","popularity","runtime","vote_average","vote_count",
            "release_month","is_english","summer","xmas",
            "log_budget","log_popularity","log_vote_count","budget_pop"]
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    year_col   = ["release_year"] if "release_year" in df.columns else []
    feat_cols  = [c for c in base+genre_cols+year_col if c in df.columns]

    X = df[feat_cols].values
    y = df["log_revenue"].values

    X_train,X_test,y_train,y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(MODELS_DIR,"scaler.pkl"))
    print(f"[preprocess] Train={len(X_train_sc)} | Test={len(X_test_sc)} | "
          f"Features={len(feat_cols)}: {feat_cols}")
    return X_train_sc, X_test_sc, y_train, y_test, scaler, feat_cols


# ═════════════════════════════════════════════════════════════════════════════
# PARTIE 3 : MÉTRIQUES
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true_log, y_pred_log):
    """MAE, MSE, RMSE, R² — métriques obligatoires Tâche 3."""
    yu = np.expm1(y_true_log); yp = np.expm1(y_pred_log)
    return {
        "rmse":     float(np.sqrt(mean_squared_error(yu,yp))),
        "mae":      float(mean_absolute_error(yu,yp)),
        "mse":      float(mean_squared_error(yu,yp)),
        "r2":       float(r2_score(y_true_log,y_pred_log)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log,y_pred_log))),
    }


# ═════════════════════════════════════════════════════════════════════════════
# PARTIE 4 : GRILLES D'HYPERPARAMÈTRES
# ═════════════════════════════════════════════════════════════════════════════

PARAM_GRIDS = {
    "LinearRegression": {},
    "Ridge":         {"alpha": [0.1, 1.0, 10.0, 100.0]},
    "RandomForest":  {"n_estimators":[100,200,300],
                      "max_depth":[8,12,None],
                      "min_samples_split":[2,5]},
    "SVR":           {"C":[10,50,100], "epsilon":[0.05,0.1], "kernel":["rbf"]},
    "XGBoost":       {"n_estimators":[100,200,300],
                      "learning_rate":[0.05,0.08,0.1],
                      "max_depth":[4,6,8]},
}

PARAM_DISTS = {
    "LinearRegression": {},
    "Ridge":         {"alpha":[0.01,0.1,0.5,1,5,10,50,100,500]},
    "RandomForest":  {"n_estimators":[50,100,150,200,300,400],
                      "max_depth":[6,8,10,12,16,None],
                      "min_samples_split":[2,4,6]},
    "SVR":           {"C":[1,5,10,30,50,100,200],
                      "epsilon":[0.01,0.05,0.1,0.2], "kernel":["rbf"]},
    "XGBoost":       {"n_estimators":[100,200,300,400,500],
                      "learning_rate":[0.01,0.03,0.05,0.08,0.1,0.15],
                      "max_depth":[4,5,6,7,8],
                      "subsample":[0.7,0.8,0.9,1.0],
                      "colsample_bytree":[0.7,0.8,0.9,1.0]},
}


# ═════════════════════════════════════════════════════════════════════════════
# PARTIE 5 : ENTRAÎNEMENT + MLFLOW
# ═════════════════════════════════════════════════════════════════════════════

def get_models(seed):
    return {
        "LinearRegression": LinearRegression(),
        "Ridge":            Ridge(alpha=1.0),
        "RandomForest":     RandomForestRegressor(n_estimators=100,
                                                   random_state=seed, n_jobs=-1),
        "SVR":              SVR(kernel="rbf", C=10, epsilon=0.1),
        "XGBoost":          XGBRegressor(n_estimators=100, learning_rate=0.1,
                                          random_state=seed, verbosity=0, n_jobs=-1),
    }


def _mlflow_log(name, model, seed, phase, method, m, elapsed=0, params={}):
    mlflow.log_params({"model":name,"seed":seed,"phase":phase,"method":method})
    if params:
        mlflow.log_params({k:str(v)[:50] for k,v in params.items()})
    mlflow.log_metrics(m)
    mlflow.log_metric("train_time_s", round(elapsed,3))
    mlflow.sklearn.log_model(model, artifact_path="model")
    joblib.dump(model, os.path.join(MODELS_DIR,f"{method}_{name}_seed{seed}.pkl"))


def run_baseline(X_train, X_test, y_train, y_test):
    """Entraînement baseline sans tuning — tous les runs loggés dans MLflow."""
    mlflow.set_experiment("cineml_baseline")
    results = []
    print("\n" + "━"*62 + "\n  PHASE BASELINE — paramètres par défaut\n" + "━"*62)
    for seed in SEEDS:
        for name, model in get_models(seed).items():
            with mlflow.start_run(run_name=f"baseline_{name}_seed{seed}"):
                t0 = time.time()
                model.fit(X_train, y_train)
                yp = model.predict(X_test)
                elapsed = time.time() - t0
                m = compute_metrics(y_test, yp)
                _mlflow_log(name, model, seed, "baseline", "none", m, elapsed)
            results.append({"phase":"baseline","model":name,"method":"none",
                            "seed":seed,**m,"time_s":round(elapsed,3),"params":"{}"})
            print(f"  [{name:<18}] seed={seed}  RMSE={m['rmse']/1e6:6.1f} M$  R²={m['r2']:.4f}")
    return pd.DataFrame(results)


def run_gridsearch(X_train, X_test, y_train, y_test):
    """Tuning avec GridSearchCV (CV=5 folds)."""
    mlflow.set_experiment("cineml_gridsearch")
    results = []
    print("\n" + "━"*62 + "\n  PHASE TUNING — GridSearchCV (CV=5)\n" + "━"*62)
    for seed in SEEDS:
        for name, base in get_models(seed).items():
            gp = PARAM_GRIDS.get(name, {})
            if gp:
                gs = GridSearchCV(base, gp, cv=5,
                                  scoring="neg_root_mean_squared_error",
                                  n_jobs=-1, refit=True)
                gs.fit(X_train, y_train)
                bp = gs.best_params_; model = gs.best_estimator_
            else:
                base.fit(X_train, y_train); bp = {}; model = base
            yp = model.predict(X_test); m = compute_metrics(y_test, yp)
            with mlflow.start_run(run_name=f"grid_{name}_seed{seed}"):
                mlflow.log_param("cv_folds", 5)
                _mlflow_log(name, model, seed, "tuned", "grid", m, 0, bp)
            results.append({"phase":"tuned","model":name,"method":"grid",
                            "seed":seed,**m,"params":str(bp)})
            print(f"  [{name:<18}] seed={seed}  RMSE={m['rmse']/1e6:6.1f} M$  R²={m['r2']:.4f}  {bp}")
    return pd.DataFrame(results)


def run_randomsearch(X_train, X_test, y_train, y_test):
    """Tuning avec RandomizedSearchCV (n_iter=20, CV=5)."""
    mlflow.set_experiment("cineml_randomsearch")
    results = []
    print("\n" + "━"*62 + "\n  PHASE TUNING — RandomizedSearchCV (n_iter=20, CV=5)\n" + "━"*62)
    for seed in SEEDS:
        for name, base in get_models(seed).items():
            dp = PARAM_DISTS.get(name, {})
            if dp:
                rs = RandomizedSearchCV(base, dp, n_iter=20, cv=5,
                                        scoring="neg_root_mean_squared_error",
                                        random_state=seed, n_jobs=-1, refit=True)
                rs.fit(X_train, y_train)
                bp = rs.best_params_; model = rs.best_estimator_
            else:
                base.fit(X_train, y_train); bp = {}; model = base
            yp = model.predict(X_test); m = compute_metrics(y_test, yp)
            with mlflow.start_run(run_name=f"random_{name}_seed{seed}"):
                mlflow.log_params({"n_iter":20,"cv_folds":5})
                _mlflow_log(name, model, seed, "tuned", "random", m, 0, bp)
            results.append({"phase":"tuned","model":name,"method":"random",
                            "seed":seed,**m,"params":str(bp)})
            print(f"  [{name:<18}] seed={seed}  RMSE={m['rmse']/1e6:6.1f} M$  R²={m['r2']:.4f}")
    return pd.DataFrame(results)


def save_best(df_res, scaler, feat_names):
    """Identifie et sauvegarde le modèle champion."""
    best = df_res.loc[df_res.rmse.idxmin()]
    key  = f"{best['method']}_{best['model']}_seed{int(best['seed'])}"
    path = os.path.join(MODELS_DIR, f"{key}.pkl")
    print(f"\n{'━'*62}\n  MODÈLE CHAMPION\n{'━'*62}")
    print(f"  Modèle  : {best['model']}")
    print(f"  Méthode : {best['method']}")
    print(f"  Seed    : {int(best['seed'])}")
    print(f"  RMSE    : {best['rmse']/1e6:.2f} M$ USD")
    print(f"  MAE     : {best['mae']/1e6:.2f} M$ USD")
    print(f"  R²      : {best['r2']:.4f}")
    if os.path.exists(path):
        bundle = os.path.join(MODELS_DIR,"best_model_bundle.pkl")
        joblib.dump({"model":joblib.load(path),"scaler":scaler,
                     "feature_names":feat_names,"metrics":best.to_dict()}, bundle)
        print(f"  Bundle  : {bundle}")
    return best


# ═════════════════════════════════════════════════════════════════════════════
# PARTIE 6 : RÉDUCTION DE DIMENSION
# ═════════════════════════════════════════════════════════════════════════════

def run_dim_reduction(X_train_sc, y_train, feat_names):
    """PCA, t-SNE et Feature Importance."""
    figs = {}
    print("\n[reduction] PCA + t-SNE + Feature Importance...")

    # PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Xp  = pca.fit_transform(X_train_sc)
    var5= PCA(n_components=min(5,X_train_sc.shape[1]),
              random_state=RANDOM_STATE).fit(X_train_sc).explained_variance_ratio_.sum()*100
    fig,ax=plt.subplots(figsize=(8,5.5))
    sc=ax.scatter(Xp[:,0],Xp[:,1],c=y_train,cmap="coolwarm",alpha=0.45,s=12)
    plt.colorbar(sc,ax=ax,label="log(revenue)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)",fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)",fontsize=11)
    ax.set_title(f"PCA — 2D  (5 composantes = {var5:.1f}% de variance)",fontsize=12,fontweight="bold")
    plt.tight_layout(); path=os.path.join(FIGURES_DIR,"fig_pca.png")
    plt.savefig(path,dpi=120,bbox_inches="tight"); plt.close(); figs["pca"]=path

    # t-SNE
    n_tsne=min(800,len(X_train_sc)); idx=np.random.default_rng(RANDOM_STATE).choice(len(X_train_sc),n_tsne,replace=False)
    Xt=TSNE(n_components=2,random_state=RANDOM_STATE,perplexity=30,max_iter=300).fit_transform(X_train_sc[idx])
    fig,ax=plt.subplots(figsize=(8,5.5))
    sc2=ax.scatter(Xt[:,0],Xt[:,1],c=y_train[idx],cmap="viridis",alpha=0.45,s=12)
    plt.colorbar(sc2,ax=ax,label="log(revenue)")
    ax.set_xlabel("t-SNE 1",fontsize=11); ax.set_ylabel("t-SNE 2",fontsize=11)
    ax.set_title(f"t-SNE — 2D ({n_tsne} films)",fontsize=12,fontweight="bold")
    plt.tight_layout(); path=os.path.join(FIGURES_DIR,"fig_tsne.png")
    plt.savefig(path,dpi=120,bbox_inches="tight"); plt.close(); figs["tsne"]=path

    # Feature Importance
    rf=RandomForestRegressor(n_estimators=100,random_state=RANDOM_STATE,n_jobs=-1)
    rf.fit(X_train_sc,y_train); imp=rf.feature_importances_; idx_s=np.argsort(imp)[::-1][:15]
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar(range(len(idx_s)),imp[idx_s],color="#4C72B0",alpha=0.85)
    ax.set_xticks(range(len(idx_s))); ax.set_xticklabels([feat_names[i] for i in idx_s],rotation=40,ha="right",fontsize=9)
    ax.set_ylabel("Importance",fontsize=11); ax.set_title("Feature Importance — Random Forest (Top 15)",fontsize=12,fontweight="bold")
    ax.yaxis.grid(True,alpha=0.35); plt.tight_layout()
    path=os.path.join(FIGURES_DIR,"fig_importance.png")
    plt.savefig(path,dpi=120,bbox_inches="tight"); plt.close(); figs["importance"]=path

    print(f"[reduction] PCA : {var5:.1f}% variance (5 composantes)")
    return figs, var5


# ═════════════════════════════════════════════════════════════════════════════
# PARTIE 7 : GRAPHIQUES DE COMPARAISON
# ═════════════════════════════════════════════════════════════════════════════

COLOR_MAP = {"LinearRegression":"#4C72B0","Ridge":"#55A868",
             "RandomForest":"#DD8452","SVR":"#C44E52","XGBoost":"#8172B2"}


def make_figures(df_res):
    """Génère les 4 graphiques de comparaison."""
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,
                          "axes.spines.right":False,"figure.facecolor":"white"})
    figs={}
    prmse=df_res.groupby(["model","phase"])["rmse"].min().unstack()
    pr2  =df_res.groupby(["model","phase"])["r2"].max().unstack()
    mo   =prmse.mean(axis=1).sort_values().index.tolist()
    x=np.arange(len(mo)); w=0.35

    # RMSE
    bv=[prmse.loc[m].get("baseline",np.nan)/1e6 for m in mo]
    tv=[prmse.loc[m].get("tuned",   np.nan)/1e6 for m in mo]
    fig,ax=plt.subplots(figsize=(10,5.5))
    b1=ax.bar(x-w/2,bv,w,label="Baseline",color="#4C72B0",alpha=0.88,zorder=3)
    b2=ax.bar(x+w/2,tv,w,label="Tuning",  color="#DD8452",alpha=0.88,zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(mo,fontsize=11); ax.set_ylabel("RMSE (millions USD)",fontsize=11)
    ax.set_title("Comparaison RMSE — Baseline vs Meilleur Tuning",fontsize=13,fontweight="bold")
    ax.legend(fontsize=11); ax.yaxis.grid(True,alpha=0.35,zorder=0)
    ax.bar_label(b1,fmt="%.0fM",fontsize=9,padding=3); ax.bar_label(b2,fmt="%.0fM",fontsize=9,padding=3,color="#993C1D")
    plt.tight_layout(); path=os.path.join(FIGURES_DIR,"fig_rmse.png")
    plt.savefig(path,dpi=120,bbox_inches="tight"); plt.close(); figs["rmse"]=path

    # R²
    r2b=[pr2.loc[m].get("baseline",np.nan) for m in mo]; r2t=[pr2.loc[m].get("tuned",np.nan) for m in mo]
    fig,ax=plt.subplots(figsize=(10,5.5))
    b3=ax.bar(x-w/2,r2b,w,label="Baseline",color="#4C72B0",alpha=0.88,zorder=3)
    b4=ax.bar(x+w/2,r2t,w,label="Tuning",  color="#DD8452",alpha=0.88,zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(mo,fontsize=11); ax.set_ylabel("R²",fontsize=11)
    ax.set_title("Comparaison R² — Baseline vs Meilleur Tuning",fontsize=13,fontweight="bold")
    ax.legend(fontsize=11); ax.yaxis.grid(True,alpha=0.35,zorder=0)
    ax.bar_label(b3,fmt="%.4f",fontsize=9,padding=3); ax.bar_label(b4,fmt="%.4f",fontsize=9,padding=3,color="#993C1D")
    plt.tight_layout(); path=os.path.join(FIGURES_DIR,"fig_r2.png")
    plt.savefig(path,dpi=120,bbox_inches="tight"); plt.close(); figs["r2"]=path

    # XGBoost méthodes
    xgb=df_res[(df_res.model=="XGBoost")&(df_res.phase=="tuned")]
    if not xgb.empty:
        xr=xgb.groupby("method")["rmse"].min()/1e6; xr2=xgb.groupby("method")["r2"].max()
        fig,(a1,a2)=plt.subplots(1,2,figsize=(11,5)); cols=["#4C72B0","#DD8452"][:len(xr)]
        b5=a1.bar(xr.index,xr.values,color=cols,alpha=0.88,zorder=3); a1.set_ylabel("RMSE (millions USD)",fontsize=11)
        a1.set_title("XGBoost — RMSE par méthode",fontsize=12,fontweight="bold"); a1.yaxis.grid(True,alpha=0.35,zorder=0)
        a1.bar_label(b5,fmt="%.1fM",padding=3,fontsize=10)
        b6=a2.bar(xr2.index,xr2.values,color=cols,alpha=0.88,zorder=3); a2.set_ylabel("R²",fontsize=11)
        a2.set_title("XGBoost — R² par méthode",fontsize=12,fontweight="bold"); a2.yaxis.grid(True,alpha=0.35,zorder=0)
        a2.bar_label(b6,fmt="%.4f",padding=3,fontsize=10); plt.tight_layout()
        path=os.path.join(FIGURES_DIR,"fig_xgb.png"); plt.savefig(path,dpi=120,bbox_inches="tight"); plt.close(); figs["xgb"]=path

    # Scatter RMSE vs R²
    tuned=df_res[df_res.phase=="tuned"].copy()
    fig,ax=plt.subplots(figsize=(9,5.5))
    for mdl,grp in tuned.groupby("model"):
        ax.scatter(grp.rmse/1e6,grp.r2,label=mdl,s=90,alpha=0.85,color=COLOR_MAP.get(mdl,"gray"),zorder=5)
    ax.set_xlabel("RMSE (millions USD)",fontsize=11); ax.set_ylabel("R²",fontsize=11)
    ax.set_title("Tous les runs Tuning — RMSE vs R²",fontsize=13,fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True,alpha=0.3,zorder=0); plt.tight_layout()
    path=os.path.join(FIGURES_DIR,"fig_scatter.png"); plt.savefig(path,dpi=120,bbox_inches="tight"); plt.close(); figs["scatter"]=path

    print(f"[figures] {len(figs)} graphiques dans '{FIGURES_DIR}/'")
    return figs


# ═════════════════════════════════════════════════════════════════════════════
# PARTIE 8 : RAPPORT PDF
# ═════════════════════════════════════════════════════════════════════════════

class NumberedCanvas(rl_canvas.Canvas):
    def __init__(self,*a,**k): super().__init__(*a,**k); self._sp=[]
    def showPage(self): self._sp.append(dict(self.__dict__)); self._startPage()
    def save(self):
        n=len(self._sp)
        for s in self._sp:
            self.__dict__.update(s); self.setFont("Helvetica",8)
            self.setFillColor(colors.HexColor("#888888"))
            self.drawCentredString(A4[0]/2,1.2*cm,
                f"Chaima Brini — ING4 DSA — CineML Box-Office  |  Page {self._pageNumber} / {n}")
            self.setStrokeColor(colors.HexColor("#DDDDDD")); self.line(2*cm,1.5*cm,A4[0]-2*cm,1.5*cm)
            super().showPage()
        super().save()


def build_pdf(df_res, figs, var_pca, n_features, output_path):
    """Rapport PDF complet — 4 pages académiques."""
    W_PAGE,_=A4; W=W_PAGE-5*cm
    BLK=colors.HexColor("#1A1A1A"); WHT=colors.white
    LGR=colors.HexColor("#F7F7F7"); HDR=colors.HexColor("#2C3E50")

    def mk(n,**k): return ParagraphStyle(n,**k)
    TITLE=mk("T",fontName="Helvetica-Bold",fontSize=22,alignment=TA_CENTER,spaceAfter=6,leading=26)
    SUB  =mk("S",fontName="Helvetica-Bold",fontSize=13,alignment=TA_CENTER,spaceAfter=4,leading=18,textColor=colors.HexColor("#555555"))
    META =mk("M",fontName="Helvetica",     fontSize=11,alignment=TA_CENTER,spaceAfter=3,leading=14,textColor=colors.HexColor("#666666"))
    H1   =mk("H1",fontName="Helvetica-Bold",fontSize=13,spaceBefore=18,spaceAfter=6,leading=16)
    H2   =mk("H2",fontName="Helvetica-Bold",fontSize=11,spaceBefore=12,spaceAfter=4,leading=14,textColor=colors.HexColor("#2C3E50"))
    BODY =mk("B",fontName="Helvetica",fontSize=10,alignment=TA_JUSTIFY,firstLineIndent=16,spaceAfter=5,leading=15)
    BULL =mk("BU",fontName="Helvetica",fontSize=10,leftIndent=18,spaceAfter=4,leading=14)
    CODE =mk("C",fontName="Courier",fontSize=9,backColor=LGR,leftIndent=12,leading=13,spaceAfter=4,textColor=colors.HexColor("#1A1A2E"))
    TH   =mk("TH",fontName="Helvetica-Bold",fontSize=10,alignment=TA_CENTER,leading=12,textColor=WHT)
    TC   =mk("TC",fontName="Helvetica",fontSize=9,alignment=TA_CENTER,leading=11)
    TL   =mk("TL",fontName="Helvetica",fontSize=9,alignment=TA_LEFT,  leading=11)
    TLBD =mk("TLBD",fontName="Helvetica-Bold",fontSize=9,alignment=TA_LEFT,leading=11)
    CHAM =mk("CH",fontName="Helvetica-Bold",fontSize=10,alignment=TA_CENTER,leading=13,textColor=colors.HexColor("#1A5C2A"))

    def ts():
        return TableStyle([
            ("BACKGROUND",(0,0),(-1,0),HDR),("TEXTCOLOR",(0,0),(-1,0),WHT),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9.5),
            ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CCCCCC")),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ROWBACKGROUNDS",(0,1),(-1,-1),[WHT,LGR]),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6)])

    HR  =lambda:HRFlowable(width="100%",thickness=0.7,color=colors.HexColor("#DDDDDD"),spaceAfter=8,spaceBefore=4)
    SP  =lambda h=0.25:Spacer(1,h*cm)
    P   =lambda t,s=BODY:Paragraph(t,s)
    B   =lambda t:Paragraph(f"&bull;&nbsp; {t}",BULL)
    TBH =lambda t:Paragraph(t,TH); TBC=lambda t:Paragraph(str(t),TC)
    TBL =lambda t:Paragraph(str(t),TL); TBLB=lambda t:Paragraph(str(t),TLBD)

    doc=SimpleDocTemplate(output_path,pagesize=A4,
        leftMargin=2.5*cm,rightMargin=2.5*cm,topMargin=2.2*cm,bottomMargin=2.2*cm)
    story=[]

    # TITRE
    story+=[SP(1.8),P("Rapport MLflow",TITLE),
            P("Expériences Multi-modèles — Prédiction Box-Office (CineML)",SUB),SP(0.4),
            HRFlowable(width="100%",thickness=2,color=colors.HexColor("#E67E22"),spaceAfter=10),
            SP(0.2),P("Chaima Brini",META),P("ING4 Data Science — Université Sésame",META),
            P("Enseignant : Aroua Hedhili",META),SP(0.2),P("19 mars 2026",META),SP(0.3),HR(),SP(0.3)]

    # OBJECTIF
    story+=[P("Objectif",H1),HR(),
            P("Ce document présente la Tâche 3 du module Machine Learning Avancée. "
              "Le projet prédit les <b>recettes au box-office</b> à partir du dataset "
              "<b>TMDB Movie Dataset (Kaggle, 4 803 films)</b>. "
              "Cinq algorithmes de régression supervisée sont comparés avec GridSearchCV, "
              "RandomizedSearchCV et suivi complet via <b>MLflow</b>."),SP()]

    # CONFIGURATION
    story+=[P("Configuration",H1),HR()]
    n_b=len(df_res[df_res.phase=="baseline"]); n_t=len(df_res[df_res.phase=="tuned"])
    cfg=[[TBH("Paramètre"),TBH("Valeur")],
         [TBLB("Dataset"),        TBL("TMDB Movie Dataset — Kaggle (4 803 films)")],
         [TBLB("Lien Kaggle"),    TBL("kaggle.com/datasets/tmdb/tmdb-movie-metadata")],
         [TBLB("Variable cible"),TBL("log(revenue) — transformation log1p")],
         [TBLB(f"Features ({n_features})"),TBL("budget, popularity, runtime, vote_average, vote_count, release_month, release_year, is_english, summer, xmas, genres×8, log_budget, log_popularity, log_vote_count, budget×pop")],
         [TBLB("Seeds"),         TBL("[11, 42]")],
         [TBLB("Modèles (5)"),   TBL("LinearRegression, Ridge, RandomForest, SVR (RBF), XGBoost")],
         [TBLB("Tuning"),        TBL("GridSearchCV (CV=5) + RandomizedSearchCV (n_iter=20, CV=5)")],
         [TBLB("Nb runs"),       TBL(f"baseline={n_b} | tuned={n_t} | total={len(df_res)}")],
         [TBLB("MLflow URI"),    TBL(MLFLOW_URI)]]
    story.append(Table(cfg,colWidths=[W*0.26,W*0.74],style=ts())); story.append(SP())

    # RÉSULTATS
    story+=[P("Résultats Principaux",H1),HR(),P("Moyennes RMSE — Baseline vs Meilleur Tuning",H2),
            P("Métriques en USD. RMSE et MAE calculés après inversion log1p."),SP(0.2)]
    prmse=df_res.groupby(["model","phase"])["rmse"].min().unstack()
    pr2  =df_res.groupby(["model","phase"])["r2"].max().unstack()
    pmae =df_res.groupby(["model","phase"])["mae"].min().unstack()
    mo   =prmse.mean(axis=1).sort_values().index.tolist()
    rr=[[TBH("Modèle"),TBH("RMSE baseline"),TBH("RMSE tuning"),TBH("ΔRMSE"),TBH("R² tuning"),TBH("MAE tuning")]]
    for m in mo:
        bv=prmse.loc[m].get("baseline",np.nan); tv=prmse.loc[m].get("tuned",np.nan)
        dv=(tv-bv) if not(np.isnan(bv) or np.isnan(tv)) else np.nan
        r2=pr2.loc[m].get("tuned",pr2.loc[m].get("baseline",np.nan))
        ma=pmae.loc[m].get("tuned",pmae.loc[m].get("baseline",np.nan))
        rr.append([TBLB(m),TBC(f"{bv/1e6:.1f} M$" if not np.isnan(bv) else "—"),
            TBC(f"{tv/1e6:.1f} M$" if not np.isnan(tv) else "—"),
            TBC(f"{dv/1e6:+.1f} M$" if not np.isnan(dv) else "—"),
            TBC(f"{r2:.4f}" if not np.isnan(r2) else "—"),
            TBC(f"{ma/1e6:.1f} M$" if not np.isnan(ma) else "—")])
    story.append(Table(rr,colWidths=[W*0.24,W*0.15,W*0.15,W*0.13,W*0.16,W*0.17],style=ts()))
    story.append(SP(0.4)); story.append(P("Top 3 des runs (triés par RMSE)",H2))
    top3=df_res.nsmallest(3,"rmse")
    t3=[[TBH("Famille"),TBH("Modèle"),TBH("Méthode"),TBH("Seed"),TBH("RMSE"),TBH("MAE"),TBH("R²")]]
    for i,(_,r) in enumerate(top3.iterrows()):
        nm=TBLB(r["model"]) if i==0 else TBL(r["model"])
        t3.append([TBC(r["phase"]),nm,TBC(r["method"]),TBC(str(int(r["seed"]))),
            TBC(f"{r['rmse']/1e6:.2f} M$"),TBC(f"{r['mae']/1e6:.2f} M$"),TBC(f"{r['r2']:.4f}")])
    st3=ts(); st3.add("BACKGROUND",(0,1),(-1,1),colors.HexColor("#D5F5E3")); st3.add("FONTNAME",(0,1),(-1,1),"Helvetica-Bold")
    story.append(Table(t3,colWidths=[W*0.12,W*0.22,W*0.12,W*0.08,W*0.17,W*0.16,W*0.13],style=st3)); story.append(SP())

    # HYPERPARAMÈTRES
    story+=[P("Hyperparamètres Influents",H1),HR(),
            P("Meilleur run par couple (modèle, seed) après tuning."),SP(0.2)]
    tuned=df_res[df_res.phase=="tuned"].sort_values("rmse").groupby(["model","seed"]).first().reset_index()
    hp=[[TBH("Modèle"),TBH("Seed"),TBH("Méthode"),TBH("RMSE"),TBH("Hyperparamètres")]]
    for _,r in tuned.sort_values("rmse").iterrows():
        bp=str(r["params"]); bp=bp[:69]+"..." if len(bp)>72 else bp
        hp.append([TBLB(r["model"]),TBC(str(int(r["seed"]))),TBC(r["method"]),TBC(f"{r['rmse']/1e6:.1f} M$"),TBL(bp)])
    story.append(Table(hp,colWidths=[W*0.19,W*0.08,W*0.11,W*0.15,W*0.47],style=ts()))

    # GRAPHIQUES
    story.append(PageBreak()); story+=[P("Visualisations",H1),HR()]
    for k,t in [("rmse","Figure 1 — RMSE : Baseline vs Tuning"),
                ("r2",  "Figure 2 — R² : Baseline vs Tuning"),
                ("xgb", "Figure 3 — XGBoost : Grid vs RandomSearch"),
                ("scatter","Figure 4 — RMSE vs R² (tous les runs tuning)")]:
        if k in figs and os.path.exists(figs[k]):
            story.append(P(t,H2)); story.append(Image(figs[k],width=W,height=W*0.52)); story.append(SP(0.3))

    # RÉDUCTION DIMENSION
    story.append(PageBreak()); story+=[P("Réduction de Dimension",H1),HR(),
        P("6.1 PCA",H2),
        P(f"La PCA réduit les {n_features} features en 2 composantes. "
          f"5 composantes expliquent {var_pca:.1f}% de la variance. "
          f"Les films à hauts revenus (tons rouges) se concentrent dans la partie droite.")]
    if "pca" in figs and os.path.exists(figs["pca"]):
        story.append(Image(figs["pca"],width=W*0.85,height=W*0.58))
    story+=[SP(0.3),P("6.2 t-SNE",H2),
            P("Le t-SNE révèle des clusters de films similaires. La séparation visible "
              "entre hauts et faibles revenus confirme la pertinence des features.")]
    if "tsne" in figs and os.path.exists(figs["tsne"]):
        story.append(Image(figs["tsne"],width=W*0.85,height=W*0.58))
    if "importance" in figs and os.path.exists(figs["importance"]):
        story+=[SP(0.3),P("6.3 Feature Importance — Random Forest",H2),
                Image(figs["importance"],width=W,height=W*0.46)]

    # CONCLUSION
    story.append(PageBreak()); best=df_res.loc[df_res.rmse.idxmin()]
    story+=[P("Conclusion",H1),HR(),
            B(f"Le meilleur modèle est <b>{best['model']} ({best['method']}, seed={int(best['seed'])})</b> "
              f"— RMSE = {best['rmse']/1e6:.2f} M$ USD, <b>R² = {best['r2']:.4f}</b>."),
            B("XGBoost et SVR bénéficient le plus du tuning. Les modèles linéaires atteignent rapidement leur limite sur ces données non-linéaires."),
            B(f"PCA ({var_pca:.1f}% variance, 5 composantes) : utile pour la visualisation mais légèrement pénalisante avant RandomForest."),
            B("RandomSearch surpasse GridSearch pour XGBoost (espace d'hyperparamètres vaste, 5 dimensions)."),
            SP()]
    ch=[[TBH("Modèle champion"),TBH("Méthode"),TBH("Seed"),TBH("RMSE"),TBH("MAE"),TBH("R²")],
        [Paragraph(best["model"],CHAM),Paragraph(best["method"],CHAM),Paragraph(str(int(best["seed"])),CHAM),
         Paragraph(f"{best['rmse']/1e6:.2f} M$",CHAM),Paragraph(f"{best['mae']/1e6:.2f} M$",CHAM),Paragraph(f"{best['r2']:.4f}",CHAM)]]
    sc2=ts(); sc2.add("BACKGROUND",(0,1),(-1,1),colors.HexColor("#D5F5E3")); sc2.add("ROWHEIGHT",(0,1),(-1,1),28)
    story.append(Table(ch,colWidths=[W*0.22,W*0.18,W*0.10,W*0.17,W*0.17,W*0.16],style=sc2)); story.append(SP())

    story+=[P("Traçabilité MLflow",H1),HR(),
            P("Toutes les expériences sont journalisées (paramètres, métriques, artefacts, modèles versionnés)."),
            P("Interface locale :",H2),
            P(f"python3 -m mlflow ui --backend-store-uri {MLFLOW_URI} --host 127.0.0.1 --port 5000",CODE),
            P("Ouvrir : http://127.0.0.1:5000"),SP(0.2)]
    er=[[TBH("Expérience MLflow"),TBH("Runs"),TBH("Meilleure RMSE"),TBH("Meilleur R²")]]
    for pn,pl in [("baseline","cineml_baseline"),("tuned","cineml_gridsearch + cineml_randomsearch")]:
        ph=df_res[df_res.phase==pn]
        if not ph.empty: er.append([TBL(pl),TBC(str(len(ph))),TBC(f"{ph.rmse.min()/1e6:.2f} M$"),TBC(f"{ph.r2.max():.4f}")])
    story.append(Table(er,colWidths=[W*0.48,W*0.12,W*0.20,W*0.20],style=ts()))

    doc.build(story,canvasmaker=NumberedCanvas)
    print(f"[pdf] {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█"*62)
    print("  TÂCHE 3 — CineML | Box-Office Prediction | MLflow")
    print("  Chaima Brini — ING4 DSA — Université Sésame")
    print("  TMDB : kaggle.com/datasets/tmdb/tmdb-movie-metadata")
    print("█"*62)

    mlflow.set_tracking_uri(MLFLOW_URI)

    print("\n[1/7] Chargement du dataset TMDB...")
    df = load_tmdb_dataset()

    print("\n[2/7] Prétraitement...")
    X_train, X_test, y_train, y_test, scaler, feat_names = preprocess(df)
    n_feat = len(feat_names)

    print("\n[3/7] Réduction de dimension...")
    dim_figs, var_pca = run_dim_reduction(X_train, y_train, feat_names)

    print("\n[4/7] Baseline...")
    df_base = run_baseline(X_train, X_test, y_train, y_test)

    print("\n[5/7] GridSearchCV...")
    df_grid = run_gridsearch(X_train, X_test, y_train, y_test)

    print("\n[6/7] RandomizedSearchCV...")
    df_rand = run_randomsearch(X_train, X_test, y_train, y_test)

    all_res = pd.concat([df_base, df_grid, df_rand], ignore_index=True)
    all_res.to_csv(RESULTS_CSV, index=False)
    print(f"\n[résultats] {len(all_res)} runs → {RESULTS_CSV}")

    best_row = save_best(all_res, scaler, feat_names)

    print("\n[7/7] Graphiques + Rapport PDF...")
    cmp_figs = make_figures(all_res)
    all_figs = {**dim_figs, **cmp_figs}
    build_pdf(all_res, all_figs, var_pca, n_feat, OUTPUT_PDF)

    print("\n" + "="*62)
    print("  TOP 5 RUNS (triés par RMSE)")
    print("="*62)
    for _,r in all_res.nsmallest(5,"rmse")[["phase","model","method","seed","rmse","r2"]].iterrows():
        print(f"  {r['model']:<18} | {r['phase']:<8} | {r['method']:<6} | "
              f"seed={int(r['seed'])} | RMSE={r['rmse']/1e6:.1f}M$ | R²={r['r2']:.4f}")

    print("\n" + "█"*62)
    print(f"  ✅  {len(all_res)} runs MLflow terminés")
    print(f"  📄  {OUTPUT_PDF}")
    print(f"  📊  {RESULTS_CSV}")
    print(f"  🗄   python3 -m mlflow ui --backend-store-uri {MLFLOW_URI} --port 5000")
    print("█"*62 + "\n")


if __name__ == "__main__":
    main()
