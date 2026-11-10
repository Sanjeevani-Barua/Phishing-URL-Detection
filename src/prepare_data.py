"""Download and clean the raw URL datasets into data/urls.csv.

Sources (public research datasets):
  * Malicious-URL dataset (faizann24): ~420k URLs labelled good/bad.
  * PhishTank verified phishing feed snapshot (via shreyagopal): ~15k URLs.
  * Top-10k ranked domains (Alexa snapshot, zer0h): added as benign homepages,
    because the URL dataset has few bare homepages - without them the model
    learned "short URL with no path = phishing". The same list (minus
    shorteners / free hosting) is used as a trusted-domain allowlist.

Cleaning steps:
  1. Normalise every URL (strip scheme and leading 'www.').
  2. Drop empty rows, exact duplicates and URLs that appear with BOTH labels.
  3. Cap the number of URLs kept per registered domain so a handful of big
     sites (wikipedia, youtube...) can't dominate training.
  4. Balance the classes by down-sampling the majority (benign) class.

Usage:  python -m src.prepare_data
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

from src.features import FREE_HOSTING, SHORTENERS, normalize_url, registered_domain

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "data" / "urls.csv"
TRUSTED_OUT = ROOT / "models" / "trusted_domains.txt"

SOURCES = {
    "malicious_urls.csv": "https://raw.githubusercontent.com/faizann24/"
    "Using-machine-learning-to-detect-malicious-URLs/master/data/data.csv",
    "top_domains.txt": "https://raw.githubusercontent.com/zer0h/top-1000000-domains/master/top-10000-domains",
    "phishtank.csv": "https://raw.githubusercontent.com/shreyagopal/"
    "Phishing-Website-Detection-by-Machine-Learning-Techniques/master/DataFiles/2.online-valid.csv",
}

MAX_PER_DOMAIN = 25
SEED = 42


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Only fixed https:// URLs from SOURCES are opened (hence the nosec below).
    for name, url in SOURCES.items():
        dest = RAW_DIR / name
        if dest.exists():
            print(f"[skip] {name} already downloaded")
            continue
        print(f"[get ] {url}")
        with urllib.request.urlopen(url, timeout=120) as resp:  # nosec B310
            dest.write_bytes(resp.read())


def load() -> pd.DataFrame:
    mal = pd.read_csv(RAW_DIR / "malicious_urls.csv")
    mal["label"] = (mal["label"].str.strip() == "bad").astype(int)
    mal["source"] = "malicious_urls"

    pt = pd.read_csv(io.BytesIO((RAW_DIR / "phishtank.csv").read_bytes()), usecols=["url"])
    pt["label"] = 1
    pt["source"] = "phishtank"

    top = pd.read_csv(RAW_DIR / "top_domains.txt", header=None, names=["url"])
    top["label"] = 0
    top["source"] = "top_domains"

    return pd.concat([mal[["url", "label", "source"]], pt, top], ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["url"]).copy()
    df["url"] = df["url"].astype(str).map(normalize_url)
    df = df[df["url"].str.len() > 3]

    # URLs labelled both ways are ambiguous -> drop them entirely.
    conflicts = df.groupby("url")["label"].nunique()
    df = df[~df["url"].isin(conflicts[conflicts > 1].index)]
    df = df.drop_duplicates(subset="url")

    df["domain"] = df["url"].map(registered_domain)
    df = (
        df.sample(frac=1, random_state=SEED)
        .groupby(["domain", "label"], group_keys=False)
        .head(MAX_PER_DOMAIN)
    )

    phish, benign = df[df.label == 1], df[df.label == 0]
    n = min(len(phish), len(benign))
    df = pd.concat([phish.sample(n, random_state=SEED), benign.sample(n, random_state=SEED)])
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def write_trusted_list() -> None:
    """Top-ranked domains minus shorteners / free hosting (anyone can publish there)."""
    top = (RAW_DIR / "top_domains.txt").read_text().split()
    keep = sorted({d.strip().lower() for d in top} - SHORTENERS - FREE_HOSTING)
    TRUSTED_OUT.parent.mkdir(parents=True, exist_ok=True)
    TRUSTED_OUT.write_text("\n".join(keep) + "\n")
    print(f"[done] {len(keep):,} trusted domains -> {TRUSTED_OUT.relative_to(ROOT)}")


def main() -> None:
    download()
    write_trusted_list()
    df = clean(load())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[done] {len(df):,} URLs -> {OUT.relative_to(ROOT)}")
    print(df["label"].value_counts().rename({0: "benign", 1: "phishing/malicious"}).to_string())
    print(f"unique registered domains: {df['domain'].nunique():,}")


if __name__ == "__main__":
    main()
