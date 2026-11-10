"""Lexical feature extraction for URLs.

Every feature is computed from the URL string alone (no network calls), so a
prediction is instant, works offline and never visits a potentially malicious
page.

URLs are normalised first (scheme and leading "www." removed). This stops the
model from learning dataset artefacts such as "benign URLs in this dataset
were stored without https" instead of real phishing signals.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import unquote, urlsplit

import tldextract

# Offline extractor: uses the bundled Public Suffix List snapshot (no HTTP).
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

SUSPICIOUS_WORDS = [
    "login", "log-in", "signin", "sign-in", "verify", "verification", "account",
    "update", "secure", "security", "banking", "confirm", "password", "passwd",
    "webscr", "billing", "suspend", "unlock", "wallet", "recover", "invoice",
    "authenticate", "validation", "support", "customer", "auth", "session",
]

# Brands commonly impersonated in phishing campaigns.
BRANDS = [
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "onedrive",
    "google", "gmail", "amazon", "netflix", "facebook", "instagram", "whatsapp",
    "linkedin", "dropbox", "docusign", "adobe", "chase", "wellsfargo",
    "bankofamerica", "hsbc", "barclays", "aib", "bankofireland", "revolut",
    "dhl", "fedex", "ups", "usps", "anpost", "ebay", "coinbase", "binance",
    "metamask", "steam", "yahoo", "aol",
]

# TLDs over-represented in abuse reports (Spamhaus / Interisle style lists).
RISKY_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "zip", "mov", "click", "link",
    "work", "support", "rest", "fit", "cn", "ru", "pw", "cc", "su", "buzz",
    "icu", "cyou", "monster", "live", "online", "site", "info", "biz", "ws",
}

SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "rebrand.ly", "shorturl.at", "tiny.cc",
    "rb.gy", "s.id", "t.ly",
}

# Free hosting / site builders frequently abused to host phishing kits.
FREE_HOSTING = {
    "000webhostapp.com", "weebly.com", "wixsite.com", "firebaseapp.com",
    "web.app", "blogspot.com", "github.io", "netlify.app", "vercel.app",
    "herokuapp.com", "glitch.me", "repl.co", "godaddysites.com", "square.site",
    "sites.google.com", "pages.dev", "webflow.io", "ipfs.io", "r2.dev",
    "workers.dev", "azurewebsites.net", "appspot.com", "sharepoint.com",
}

# Hosts on otherwise-trusted domains where anyone can publish content.
USER_CONTENT_HOSTS = {
    "sites.google.com", "docs.google.com", "drive.google.com", "forms.gle",
    "storage.googleapis.com", "script.google.com", "dl.dropboxusercontent.com",
    "onedrive.live.com", "1drv.ms", "forms.office.com", "notion.site",
}

RISKY_EXTENSIONS = (".php", ".exe", ".zip", ".rar", ".scr", ".js", ".apk", ".bin", ".asp", ".cgi")

FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "query_length",
    "num_dots_host", "num_subdomains", "num_hyphens_host", "num_hyphens_url",
    "num_digits", "digit_ratio", "num_special_chars", "num_path_segments",
    "num_query_params", "has_at_symbol", "has_ip_host", "has_port",
    "has_double_slash_path", "has_punycode", "has_percent_encoding",
    "host_entropy", "url_entropy", "longest_token", "suspicious_word_count",
    "brand_outside_domain", "risky_tld", "is_shortener", "is_free_hosting",
    "risky_extension", "domain_vowel_ratio", "domain_digit_count",
]

# Human-readable descriptions used by the app's explanation panel.
FEATURE_LABELS = {
    "url_length": "URL length",
    "host_length": "Hostname length",
    "path_length": "Path length",
    "query_length": "Query string length",
    "num_dots_host": "Dots in hostname",
    "num_subdomains": "Number of subdomains",
    "num_hyphens_host": "Hyphens in hostname",
    "num_hyphens_url": "Hyphens in URL",
    "num_digits": "Digits in URL",
    "digit_ratio": "Share of digits",
    "num_special_chars": "Special characters (@ ~ _ % = & ! *)",
    "num_path_segments": "Path depth",
    "num_query_params": "Query parameters",
    "has_at_symbol": "Contains '@' (can hide the real host)",
    "has_ip_host": "Raw IP address instead of a domain",
    "has_port": "Explicit port number",
    "has_double_slash_path": "'//' inside the path (redirect trick)",
    "has_punycode": "Punycode / look-alike characters (xn--)",
    "has_percent_encoding": "Percent-encoded characters",
    "host_entropy": "Hostname randomness (entropy)",
    "url_entropy": "URL randomness (entropy)",
    "longest_token": "Longest word/token",
    "suspicious_word_count": "Phishing keywords (login, verify, account...)",
    "brand_outside_domain": "Brand name used outside the real domain",
    "risky_tld": "Frequently-abused top-level domain",
    "is_shortener": "URL shortener",
    "is_free_hosting": "Free hosting / site-builder platform",
    "risky_extension": "Risky file extension (.php, .exe, .zip...)",
    "domain_vowel_ratio": "Vowel ratio of domain name",
    "domain_digit_count": "Digits in domain name",
}


def normalize_url(url: str) -> str:
    """Lower-case the host, drop the scheme and a leading 'www.'."""
    url = str(url).strip()
    url = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", url)
    if url.lower().startswith("www."):
        url = url[4:]
    host, sep, rest = url.partition("/")
    # Drop trailing slashes: "site.com/" and "site.com" are the same page,
    # and the raw datasets store them inconsistently per class.
    return (host.lower() + sep + rest).rstrip("/")


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def _is_ip(host: str) -> bool:
    host = host.strip("[]")
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        # Hex / integer encoded IPs such as 0x7f000001 or 3232235777
        return bool(re.fullmatch(r"(0x[0-9a-f]+|\d{8,10})", host))


def split_url(url: str) -> dict:
    """Return the parts of a normalised URL."""
    norm = normalize_url(url)
    try:
        parts = urlsplit("http://" + norm)
        netloc, path, query = parts.netloc, parts.path, parts.query
        host = parts.hostname or ""
    except ValueError:
        # Malformed URLs (e.g. stray '[') are common in phishing feeds;
        # fall back to a simple manual split instead of crashing.
        netloc = re.split(r"[/?#]", norm, maxsplit=1)[0]
        rest = norm[len(netloc):]
        path, _, query = rest.partition("?")
        host = netloc.rsplit("@", 1)[-1].split(":")[0].strip("[]").lower()
    ext = _EXTRACT(host)
    registered = ext.top_domain_under_public_suffix or host
    return {
        "normalized": norm,
        "netloc": netloc,
        "host": host,
        "path": path,
        "query": query,
        "domain": ext.domain,
        "suffix": ext.suffix,
        "subdomain": ext.subdomain,
        "registered_domain": registered,
    }


def registered_domain(url: str) -> str:
    return split_url(url)["registered_domain"]


def extract_features(url: str) -> dict:
    p = split_url(url)
    norm, host, path, query = p["normalized"], p["host"], p["path"], p["query"]
    lower = unquote(norm).lower()
    domain = p["domain"].lower()
    registered = p["registered_domain"].lower()
    last_tld = p["suffix"].split(".")[-1] if p["suffix"] else ""

    digits = sum(ch.isdigit() for ch in norm)
    tokens = [t for t in re.split(r"[/\.\?=&\-_:@~%+]", lower) if t]
    outside_domain = lower.replace(registered, " ", 1) if registered else lower
    vowels = sum(ch in "aeiou" for ch in domain)

    feats = {
        "url_length": len(norm),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots_host": host.count("."),
        "num_subdomains": len([s for s in p["subdomain"].split(".") if s]),
        "num_hyphens_host": host.count("-"),
        "num_hyphens_url": norm.count("-"),
        "num_digits": digits,
        "digit_ratio": digits / max(len(norm), 1),
        "num_special_chars": sum(norm.count(c) for c in "@~_%=&!*"),
        "num_path_segments": len([s for s in path.split("/") if s]),
        "num_query_params": len([q for q in query.split("&") if q]),
        "has_at_symbol": int("@" in norm),
        "has_ip_host": int(_is_ip(host)),
        "has_port": int(bool(re.search(r":\d+$", p["netloc"]))),
        "has_double_slash_path": int("//" in path),
        "has_punycode": int("xn--" in host),
        "has_percent_encoding": int("%" in norm),
        "host_entropy": _entropy(host),
        "url_entropy": _entropy(norm),
        "longest_token": max((len(t) for t in tokens), default=0),
        "suspicious_word_count": sum(lower.count(w) for w in SUSPICIOUS_WORDS),
        "brand_outside_domain": int(any(b in outside_domain for b in BRANDS) and domain not in BRANDS),
        "risky_tld": int(last_tld in RISKY_TLDS),
        "is_shortener": int(registered in SHORTENERS or host in SHORTENERS),
        "is_free_hosting": int(registered in FREE_HOSTING or any(host.endswith("." + f) for f in FREE_HOSTING)),
        "risky_extension": int(path.lower().endswith(RISKY_EXTENSIONS)),
        "domain_vowel_ratio": vowels / max(len(domain), 1),
        "domain_digit_count": sum(ch.isdigit() for ch in domain),
    }
    return feats


def features_frame(urls):
    """Vectorise an iterable of URLs into a DataFrame with FEATURE_NAMES columns."""
    import pandas as pd

    return pd.DataFrame([extract_features(u) for u in urls], columns=FEATURE_NAMES)
