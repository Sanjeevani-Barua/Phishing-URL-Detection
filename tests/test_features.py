from src.features import FEATURE_NAMES, extract_features, normalize_url, split_url
from src.model import PhishingDetector


def test_normalize_strips_scheme_www_and_trailing_slash():
    assert normalize_url("HTTPS://WWW.Example.COM/Path/") == "example.com/Path"
    assert normalize_url("example.com") == "example.com"


def test_all_features_present_and_numeric():
    feats = extract_features("http://paypal-login.verify.tk/webscr?cmd=1")
    assert list(feats) == FEATURE_NAMES
    assert all(isinstance(v, (int, float)) for v in feats.values())


def test_ip_host_detected():
    assert extract_features("http://192.168.1.10/login.php")["has_ip_host"] == 1
    assert extract_features("https://google.com")["has_ip_host"] == 0


def test_brand_outside_domain():
    assert extract_features("http://paypal.secure-check.com/login")["brand_outside_domain"] == 1
    assert extract_features("https://paypal.com/signin")["brand_outside_domain"] == 0


def test_malformed_url_does_not_crash():
    parts = split_url("http://[broken.example.com/login")
    assert parts["host"]
    extract_features("http://[broken.example.com/login")


def test_detector_flags_obvious_phishing_and_trusts_known_domains():
    d = PhishingDetector()
    assert d.analyse("http://paypal-secure-login.verify-account.tk/webscr?cmd=login").verdict == "Phishing"
    safe = d.analyse("https://accounts.google.com/signin")
    assert safe.trusted_domain and safe.verdict == "Likely safe"
    # user-content host on a trusted domain must NOT be auto-trusted
    assert not d.is_trusted("https://sites.google.com/view/paypal-verify")
