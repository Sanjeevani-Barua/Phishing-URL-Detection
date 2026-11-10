"""Hybrid phishing detector used by the app (and tests).

Two complementary models, averaged:
  1. XGBoost on 30 hand-crafted lexical features  -> explainable signals
     (per-feature contributions via XGBoost's built-in TreeSHAP).
  2. Logistic Regression on character 3-5-grams    -> learns suspicious
     character patterns ("paypa1", "-secure-login", "/wp-admin/...").

Plus a reputation layer: URLs on top-ranked domains (excluding shorteners,
free hosting and user-content hosts like sites.google.com) are marked as
trusted. A lexical model alone flags real login pages such as
accounts.google.com/signin, because they *look* like phishing pages - real
email/web gateways combine ML with reputation for exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

from src.features import (
    FEATURE_LABELS,
    FEATURE_NAMES,
    USER_CONTENT_HOSTS,
    extract_features,
    features_frame,
    normalize_url,
    split_url,
)

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


@dataclass
class Prediction:
    url: str
    probability: float          # ensemble probability of phishing (0-1)
    xgb_probability: float
    ngram_probability: float
    verdict: str                # "Phishing", "Suspicious" or "Likely safe"
    trusted_domain: bool        # registered domain is on the trusted list
    registered_domain: str
    reasons: list               # [(label, value, contribution)] pushing towards phishing
    safe_signals: list          # [(label, value, contribution)] pushing towards benign
    fragments: list             # [(ngram, weight)] most suspicious character patterns


class PhishingDetector:
    def __init__(self, model_dir: Path = MODEL_DIR, threshold: float = 0.5):
        self.booster = xgb.Booster()
        self.booster.load_model(str(model_dir / "xgb_model.json"))
        bundle = joblib.load(model_dir / "ngram_model.joblib")
        self.vectorizer, self.ngram_lr = bundle["vectorizer"], bundle["model"]
        self.vocab = np.array(self.vectorizer.get_feature_names_out())
        self.threshold = threshold
        trusted_file = model_dir / "trusted_domains.txt"
        self.trusted = set(trusted_file.read_text().split()) if trusted_file.exists() else set()

    def is_trusted(self, url: str) -> bool:
        parts = split_url(url)
        return parts["registered_domain"] in self.trusted and parts["host"] not in USER_CONTENT_HOSTS

    # ---- batch scoring (used for evaluation) --------------------------------
    def predict_proba(self, urls) -> np.ndarray:
        urls = [normalize_url(u) for u in urls]
        p_xgb = self.booster.predict(xgb.DMatrix(features_frame(urls)))
        p_ng = self.ngram_lr.predict_proba(self.vectorizer.transform(urls))[:, 1]
        return (p_xgb + p_ng) / 2

    def _suspicious_fragments(self, norm: str, vec, top_k: int = 15, max_out: int = 5) -> list:
        """Map the most phishing-weighted n-grams back onto the URL and merge
        overlapping ones into whole segments, e.g. 'pay'+'ayp'+'pa1' -> 'paypa1'."""
        idx = vec.nonzero()[1]
        weights = vec.data * self.ngram_lr.coef_[0][idx]
        text = norm.lower()
        score = np.zeros(len(text))
        for j in np.argsort(weights)[::-1][:top_k]:
            if weights[j] <= 0:
                break
            gram = self.vocab[idx[j]].strip()
            start = text.find(gram)
            while gram and start != -1:
                score[start:start + len(gram)] += weights[j]
                start = text.find(gram, start + 1)
        segments, i = [], 0
        while i < len(text):
            if score[i] > 0:
                k = i
                while k < len(text) and score[k] > 0:
                    k += 1
                segments.append((norm[i:k], float(score[i:k].sum())))
                i = k
            else:
                i += 1
        best: dict = {}
        for seg, s in segments:
            best[seg] = max(s, best.get(seg, 0.0))
        return sorted(best.items(), key=lambda s: -s[1])[:max_out]

    # ---- single URL with explanation ----------------------------------------
    def analyse(self, url: str) -> Prediction:
        norm = normalize_url(url)
        feats = extract_features(norm)
        X = features_frame([norm])
        dm = xgb.DMatrix(X)
        p_xgb = float(self.booster.predict(dm)[0])
        contribs = self.booster.predict(dm, pred_contribs=True)[0][:-1]  # drop bias term

        vec = self.vectorizer.transform([norm])
        p_ng = float(self.ngram_lr.predict_proba(vec)[0, 1])
        prob = (p_xgb + p_ng) / 2

        order = np.argsort(contribs)
        reasons = [
            (FEATURE_LABELS[FEATURE_NAMES[i]], feats[FEATURE_NAMES[i]], float(contribs[i]))
            for i in order[::-1] if contribs[i] > 0.05
        ][:6]
        safe = [
            (FEATURE_LABELS[FEATURE_NAMES[i]], feats[FEATURE_NAMES[i]], float(contribs[i]))
            for i in order if contribs[i] < -0.05
        ][:4]

        fragments = self._suspicious_fragments(norm, vec)

        trusted = self.is_trusted(norm)
        if trusted:
            verdict = "Likely safe"
        elif prob >= 0.8:
            verdict = "Phishing"
        elif prob >= self.threshold:
            verdict = "Suspicious"
        else:
            verdict = "Likely safe"
        return Prediction(
            norm, prob, p_xgb, p_ng, verdict, trusted, split_url(norm)["registered_domain"],
            reasons, safe, fragments,
        )
