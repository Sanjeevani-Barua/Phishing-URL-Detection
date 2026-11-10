"""Streamlit front-end for the phishing URL detector.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.model import PhishingDetector

ROOT = Path(__file__).resolve().parent
MAX_BATCH = 1000

st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_detector() -> PhishingDetector:
    return PhishingDetector()


@st.cache_data
def load_metrics() -> dict:
    return json.loads((ROOT / "models" / "metrics.json").read_text())


def signal_table(signals) -> pd.DataFrame:
    rows = [(label, round(v, 2) if isinstance(v, float) else v, round(c, 2)) for label, v, c in signals]
    return pd.DataFrame(rows, columns=["Signal", "Value", "Impact"])


detector = load_detector()
metrics = load_metrics()
final = metrics["test_metrics"]["Hybrid ensemble (final)"]

EXAMPLES = {
    "Fake PayPal login": "http://paypal-secure-login.verify-account.tk/webscr?cmd=login",
    "Parcel-fee scam": "http://anpost-parcel-redelivery-fee.com/pay",
    "Look-alike Apple ID": "https://appleid.apple.com.security-check.xyz/verify",
    "Real Wikipedia page": "https://en.wikipedia.org/wiki/Phishing",
}

st.title("🛡️ Phishing URL Detector")
st.caption(
    f"Hybrid ML model (XGBoost + character n-grams) · {final['accuracy']:.1%} accuracy and "
    f"{final['false_positive_rate']:.1%} false-positive rate on websites it never saw during training. "
    "The URL is analysed as text only and is **never opened**."
)

tab_single, tab_batch, tab_about = st.tabs(["Check a URL", "Batch check", "How it works"])

# --------------------------------------------------------------------------- single
with tab_single:
    if "url" not in st.session_state:
        st.session_state.url = ""

    st.write("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, (label, example) in zip(cols, EXAMPLES.items()):
        if col.button(label, use_container_width=True):
            st.session_state.url = example

    url = st.text_input("URL to check", key="url", placeholder="e.g. http://secure-login-paypal.com/verify")

    if url.strip():
        r = detector.analyse(url.strip())
        colour = {"Phishing": "red", "Suspicious": "orange", "Likely safe": "green"}[r.verdict]
        icon = {"Phishing": "🚨", "Suspicious": "⚠️", "Likely safe": "✅"}[r.verdict]

        st.markdown(f"### {icon} :{colour}[{r.verdict}]")
        st.progress(r.probability, text=f"Phishing probability (ML model): **{r.probability:.0%}**")

        if r.trusted_domain:
            st.info(
                f"**{r.registered_domain}** is on the trusted-domain list (top-ranked websites), "
                "so it is marked safe even if the URL pattern looks risky. Real login pages such as "
                "accounts.google.com look a lot like phishing pages to a text-only model."
            )

        c1, c2 = st.columns(2)
        c1.metric("Lexical-feature model (XGBoost)", f"{r.xgb_probability:.0%}")
        c2.metric("Character-pattern model (n-grams)", f"{r.ngram_probability:.0%}")

        if r.reasons:
            st.markdown("#### Why it looks suspicious")
            st.dataframe(signal_table(r.reasons), hide_index=True, use_container_width=True)
        if r.fragments:
            st.markdown("#### Suspicious text patterns")
            st.write(" ".join(f"`{g}`" for g, _ in r.fragments))
        if r.safe_signals:
            with st.expander("Signals pointing towards safe"):
                st.dataframe(signal_table(r.safe_signals), hide_index=True, use_container_width=True)
        st.caption(
            "Impact = contribution to the XGBoost score (log-odds, TreeSHAP). "
            "This tool is a learning project — always verify before trusting or reporting a link."
        )

# --------------------------------------------------------------------------- batch
with tab_batch:
    st.write(f"Paste up to {MAX_BATCH} URLs (one per line) or upload a CSV with a `url` column.")
    text = st.text_area("URLs", height=150, placeholder="example.com/login\nhttp://bit.ly/abc123")
    upload = st.file_uploader("…or upload CSV", type=["csv"])

    urls: list[str] = []
    if upload is not None:
        df_up = pd.read_csv(upload)
        if "url" not in df_up.columns:
            st.error("CSV needs a column named `url`.")
        else:
            urls = df_up["url"].dropna().astype(str).tolist()
    elif text.strip():
        urls = [u.strip() for u in text.splitlines() if u.strip()]

    if urls:
        if len(urls) > MAX_BATCH:
            st.warning(f"Only the first {MAX_BATCH} URLs are checked.")
            urls = urls[:MAX_BATCH]
        probs = detector.predict_proba(urls)
        rows = []
        for u, p in zip(urls, probs):
            trusted = detector.is_trusted(u)
            verdict = "Likely safe" if trusted or p < 0.5 else ("Phishing" if p >= 0.8 else "Suspicious")
            rows.append({"url": u, "phishing_probability": round(float(p), 3),
                         "trusted_domain": trusted, "verdict": verdict})
        out = pd.DataFrame(rows).sort_values("phishing_probability", ascending=False)
        c1, c2, c3 = st.columns(3)
        c1.metric("Checked", len(out))
        c2.metric("Phishing", int((out.verdict == "Phishing").sum()))
        c3.metric("Suspicious", int((out.verdict == "Suspicious").sum()))
        st.dataframe(out, hide_index=True, use_container_width=True)
        st.download_button("Download results (CSV)", out.to_csv(index=False), "phishing_results.csv", "text/csv")

# --------------------------------------------------------------------------- about
with tab_about:
    ds = metrics["dataset"]
    st.markdown(
        f"""
**Data:** {ds['total_urls']:,} URLs ({ds['phishing_or_malicious']:,} phishing/malicious,
{ds['benign']:,} benign) from {ds['unique_domains']:,} different domains — a public malicious-URL
research dataset, a PhishTank verified-phishing snapshot and top-ranked domain homepages.

**Honest evaluation:** the test set contains only websites (registered domains) that never
appear in training, so the model can't score well by memorising domains.

**Model:** average of two models:
1. **XGBoost** on 30 hand-crafted lexical features (length, entropy, IP host, risky TLD,
   brand name outside the real domain, phishing keywords…), explainable per URL.
2. **Logistic regression on character 3–5-grams**, which picks up patterns like `paypa1`, `-secure-`, `.tk/`.

**Reputation layer:** top-ranked domains (minus URL shorteners, free hosting and
user-content hosts like `sites.google.com`) are treated as trusted.
"""
    )
    table = pd.DataFrame(metrics["test_metrics"]).T[
        ["accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate"]
    ]
    st.markdown("**Results on unseen domains**")
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)
    for img, cap in [("roc_curves.png", "ROC curves"), ("confusion_matrix.png", "Confusion matrix"),
                     ("feature_importance.png", "Most important lexical features")]:
        path = ROOT / "reports" / img
        if path.exists():
            st.image(str(path), caption=cap)
