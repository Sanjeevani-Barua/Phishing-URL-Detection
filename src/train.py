"""Train and evaluate the phishing URL classifier.

* The train/test split is GROUPED BY REGISTERED DOMAIN, so no website appears
  in both sets. A random split lets models memorise domains and inflates the
  score; both are reported to show the difference.
* Compares baselines with the final hybrid model:
    XGBoost (30 lexical features)  +  Logistic Regression (char 3-5-grams)

Usage:  python -m src.train
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.features import FEATURE_LABELS, FEATURE_NAMES, features_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "urls.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SEED = 42


def scores(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    return {
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
    }


def new_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.08, subsample=0.9,
        colsample_bytree=0.9, eval_metric="logloss", n_jobs=-1, random_state=SEED,
    )


def new_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3,
        max_features=100_000, sublinear_tf=True, dtype=np.float32,
    )


def fit_hybrid(urls_tr, X_tr, y_tr):
    xgb_model = new_xgb().fit(X_tr, y_tr)
    vec = new_vectorizer()
    lr = LogisticRegression(C=4, max_iter=3000).fit(vec.fit_transform(urls_tr), y_tr)
    return xgb_model, vec, lr


def predict_hybrid(models, urls, X):
    xgb_model, vec, lr = models
    p_xgb = xgb_model.predict_proba(X)[:, 1]
    p_ng = lr.predict_proba(vec.transform(urls))[:, 1]
    return p_xgb, p_ng, (p_xgb + p_ng) / 2


def main() -> None:
    df = pd.read_csv(DATA, keep_default_na=False)
    urls = df["url"].to_numpy()
    y = df["label"].to_numpy()
    print(f"Loaded {len(df):,} URLs")
    t0 = time.time()
    X = features_frame(urls)
    print(f"Extracted {X.shape[1]} features in {time.time() - t0:.1f}s")

    # --- grouped split: no domain in both train and test -------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(X, y, groups=df["domain"]))
    assert not set(df["domain"].iloc[tr]) & set(df["domain"].iloc[te])

    results, probas = {}, {}
    baselines = {
        "Logistic Regression (features)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random Forest (features)": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED
        ),
    }
    for name, model in baselines.items():
        model.fit(X.iloc[tr], y[tr])
        probas[name] = model.predict_proba(X.iloc[te])[:, 1]

    t0 = time.time()
    hybrid = fit_hybrid(urls[tr], X.iloc[tr], y[tr])
    p_xgb, p_ng, p_ens = predict_hybrid(hybrid, urls[te], X.iloc[te])
    print(f"Hybrid trained in {time.time() - t0:.0f}s")
    probas["XGBoost (features)"] = p_xgb
    probas["LogReg (char n-grams)"] = p_ng
    probas["Hybrid ensemble (final)"] = p_ens

    for name, p in probas.items():
        results[name] = scores(y[te], p)
        print(f"{name:32s} {results[name]}")

    # --- random split, for comparison only ---------------------------------
    r_tr, r_te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)
    leaky = fit_hybrid(urls[r_tr], X.iloc[r_tr], y[r_tr])
    random_split = scores(y[r_te], predict_hybrid(leaky, urls[r_te], X.iloc[r_te])[2])
    print(f"Hybrid (random split, for comparison) {random_split}")

    # --- refit final model on ALL data and save ----------------------------
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    xgb_full, vec_full, lr_full = fit_hybrid(urls, X, y)
    xgb_full.get_booster().save_model(MODEL_DIR / "xgb_model.json")
    lr_full.coef_ = lr_full.coef_.astype(np.float32)
    joblib.dump({"vectorizer": vec_full, "model": lr_full}, MODEL_DIR / "ngram_model.joblib", compress=3)

    metrics = {
        "dataset": {
            "total_urls": int(len(df)),
            "phishing_or_malicious": int(y.sum()),
            "benign": int(len(y) - y.sum()),
            "unique_domains": int(df["domain"].nunique()),
            "train_urls": int(len(tr)),
            "test_urls": int(len(te)),
        },
        "split": "GroupShuffleSplit by registered domain (80/20) - test domains never seen in training",
        "test_metrics": results,
        "hybrid_random_split_for_comparison": random_split,
        "features": FEATURE_NAMES,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- plots (evaluated on the held-out domains) --------------------------
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y[te], (p_ens >= 0.5).astype(int), display_labels=["Benign", "Phishing"],
        cmap="Blues", ax=ax, colorbar=False, values_format=",",
    )
    ax.set_title("Hybrid model – unseen domains")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y[te], p)
        ax.plot(fpr, tpr, lw=2 if "Hybrid" in name else 1.2,
                label=f"{name} (AUC {results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves – unseen domains")
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "roc_curves.png")
    plt.close(fig)

    gain = xgb_full.get_booster().get_score(importance_type="gain")
    imp = pd.Series({f: gain.get(f, 0.0) for f in FEATURE_NAMES})
    imp = (imp / imp.sum()).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6.6, 5))
    ax.barh([FEATURE_LABELS[f] for f in imp.index], imp.values, color="#3b6fb6")
    ax.set_xlabel("Relative importance (gain)")
    ax.set_title("Top 15 lexical features – XGBoost")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "feature_importance.png")
    plt.close(fig)

    print("Saved models/, metrics.json and reports/*.png")


if __name__ == "__main__":
    main()
