"""Train, compare, tune and save the crop model.   Run:  python train_model.py
Uses data/Crop_recommendation.csv (Kaggle). If it is missing, a SYNTHETIC look-alike dataset is
generated so the app still runs - replace it with the real file and retrain for real results."""
import json, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from predict import FEATURES, RANGES
warnings.filterwarnings("ignore")

REAL, SYN = Path("data/Crop_recommendation.csv"), Path("data/synthetic_crop_data.csv")
# Approximate per-crop means (N, P, K, temp, humidity, pH, rainfall) used ONLY for the synthetic fallback
MEANS = {"rice": (80,48,40,23.7,82,6.4,236), "maize": (77,48,20,22.4,65,6.2,85), "chickpea": (40,68,80,18.9,17,7.3,80),
 "kidneybeans": (21,68,20,20.1,22,5.75,106), "pigeonpeas": (21,68,20,27.7,48,5.8,150), "mothbeans": (21,48,20,28.2,53,6.8,51),
 "mungbean": (21,48,20,28.5,85,6.7,48), "blackgram": (40,68,19,30,65,7.1,68), "lentil": (19,68,19,24.5,65,6.9,46),
 "pomegranate": (19,19,40,21.8,90,6.4,108), "banana": (100,82,50,27.4,80,6.0,105), "mango": (20,27,30,31.2,50,5.8,95),
 "grapes": (23,133,200,23.8,82,6.0,70), "watermelon": (99,17,50,25.6,85,6.5,51), "muskmelon": (100,17,50,28.7,92,6.4,25),
 "apple": (21,134,200,22.6,92,5.9,113), "orange": (20,10,10,22.8,92,7.0,110), "papaya": (50,59,50,33.7,92,6.7,143),
 "coconut": (22,17,31,27.4,95,6.0,176), "cotton": (118,46,20,24,80,6.9,80), "jute": (78,47,40,25,80,6.7,175),
 "coffee": (101,29,30,25.5,59,6.8,158)}

def make_synthetic(n=100, seed=42):
    rng, rows = np.random.default_rng(seed), []
    for crop, m in MEANS.items():
        sd = [max(.12*m[0], 3), max(.12*m[1], 3), max(.12*m[2], 3), 2.5, 5, .4, .15*m[6] + 5]
        for _ in range(n):
            v = [rng.normal(m[i], sd[i]) for i in range(7)]
            v = [float(np.clip(v[i], *RANGES[FEATURES[i]])) for i in range(7)]
            rows.append([round(x, 2) for x in v] + [crop])
    return pd.DataFrame(rows, columns=FEATURES + ["label"])

def make(clf): return Pipeline([("scaler", StandardScaler()), ("clf", clf)])  # scaler fitted inside CV folds
def models():
    return {"Logistic Regression": LogisticRegression(max_iter=2000), "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42), "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(), "SVM": SVC(probability=True, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)}
GRIDS = {"Logistic Regression": {"clf__C": [1, 10, 100]}, "Decision Tree": {"clf__max_depth": [None, 10, 20]},
         "Random Forest": {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 20]},
         "KNN": {"clf__n_neighbors": [3, 5, 7, 11]}, "Naive Bayes": {"clf__var_smoothing": [1e-9, 1e-8, 1e-7]},
         "SVM": {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.1]},
         "Gradient Boosting": {"clf__n_estimators": [100], "clf__learning_rate": [0.05, 0.1]}}

def main():
    for d in ("data", "models", "figures"): Path(d).mkdir(exist_ok=True)
    if REAL.exists(): df, src = pd.read_csv(REAL), "Kaggle Crop Recommendation dataset"
    else:
        df, src = make_synthetic(), "SYNTHETIC data (put Crop_recommendation.csv in data/ and retrain)"; df.to_csv(SYN, index=False)
    print("Data source:", src, df.shape)
    # --- Cleaning: missing values, duplicates, outliers (IQR rule; reported, kept: extremes are valid for some crops)
    print("Missing values:", int(df.isna().sum().sum()), "| Duplicates:", int(df.duplicated().sum()))
    df = df.dropna().drop_duplicates()
    q1, q3 = df[FEATURES].quantile(.25), df[FEATURES].quantile(.75); iqr = q3 - q1
    print("IQR outliers per feature:\n", ((df[FEATURES] < q1 - 1.5*iqr) | (df[FEATURES] > q3 + 1.5*iqr)).sum().to_string())
    X, y = df[FEATURES], df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    cv, rows = StratifiedKFold(5, shuffle=True, random_state=42), []
    for name, clf in models().items():
        pipe = make(clf); cvs = cross_val_score(pipe, Xtr, ytr, cv=cv, n_jobs=-1)
        pred = pipe.fit(Xtr, ytr).predict(Xte)
        p, r, f, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
        rows.append({"model": name, "cv_accuracy": cvs.mean(), "cv_std": cvs.std(), "test_accuracy": accuracy_score(yte, pred),
                     "macro_precision": p, "macro_recall": r, "macro_f1": f})
    res = pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False).round(4)
    res.to_csv("models/model_comparison.csv", index=False); print(res.to_string(index=False))
    # --- Tune the best model (by CV accuracy)
    best = res.iloc[0]["model"]; print("Tuning:", best)
    gs = GridSearchCV(make(models()[best]), GRIDS[best], cv=cv, n_jobs=-1).fit(Xtr, ytr)
    model, pred = gs.best_estimator_, gs.best_estimator_.predict(Xte)
    print("Best params:", gs.best_params_, "| test accuracy:", round(accuracy_score(yte, pred), 4))
    open("models/classification_report.txt", "w").write(classification_report(yte, pred))
    fig, ax = plt.subplots(figsize=(13, 13))
    ConfusionMatrixDisplay(confusion_matrix(yte, pred, labels=model.classes_), display_labels=model.classes_).plot(
        ax=ax, xticks_rotation=90, colorbar=False, cmap="Greens"); ax.set_title(f"Confusion matrix - {best}")
    fig.tight_layout(); fig.savefig("figures/confusion_matrix.png", dpi=100); plt.close(fig)
    imp = permutation_importance(model, Xte, yte, n_repeats=10, random_state=42, n_jobs=-1).importances_mean
    fig, ax = plt.subplots(figsize=(6, 4)); order = np.argsort(imp)
    ax.barh(np.array(FEATURES)[order], imp[order], color="#2e7d32"); ax.set_xlabel("Drop in accuracy when shuffled")
    ax.set_title("Feature importance"); fig.tight_layout(); fig.savefig("figures/feature_importance.png", dpi=100); plt.close(fig)
    # --- Per-crop ideal profile (mean + 10th/90th percentile) for the chart and advice rules
    prof = {c: {f: {"mean": float(g[f].mean()), "low": float(g[f].quantile(.1)), "high": float(g[f].quantile(.9))}
                for f in FEATURES} for c, g in df.groupby("label")}
    json.dump(prof, open("models/crop_profiles.json", "w"), indent=1)
    json.dump({"data_source": src, "best_model": best, "best_params": {k: str(v) for k, v in gs.best_params_.items()},
               "test_accuracy": float(accuracy_score(yte, pred))}, open("models/metrics.json", "w"), indent=1)
    joblib.dump(model, "models/crop_model.joblib"); print("Saved models/crop_model.joblib")

if __name__ == "__main__": main()
