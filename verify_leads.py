"""
Standalone verification script for leads (safe, run offline).

Usage:
  python verify_leads.py --input scraped_leads.csv --output verified_leads.csv

What it does (free-only approach):
 - Reads input CSV (expects columns like agency_name, location, website, phone, contact_email).
 - For each lead, queries OpenStreetMap Nominatim for matches (name + location).
 - If a website is available, checks that the site is reachable and looks for keywords.
 - Combines signals into a verification_score (0-100) and sets verification_status:
     - Verified (score >= 70)
     - Needs Review (40 <= score < 70)
     - Unverified (score < 40)
 - Writes output CSV with new columns: verification_status, verification_score, verified_sources, verified_at

Notes and safety:
 - Uses Nominatim (free) and respects a 1-second delay between requests. Do not hammer the service.
 - No paid third-party APIs required.
 - This is a standalone tool and doesn't modify your Streamlit app; you can run it locally or in a CI job.
 - Optional: add Justdial/Sulekha scraping later (requires adding BeautifulSoup) but be mindful of terms of service.

"""

import argparse
import pandas as pd
import requests
import time
import datetime
import re

# NOTE: This file is intentionally standalone and does not modify the app runtime.
# It reads input CSVs and writes a verification report.

# Config
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "SkillUpLeadVerifier/1.0 (your-email@example.com)"  # replace email if you like
NOMINATIM_DELAY = 1.1  # seconds between requests (respect the service)
KEYWORDS = [
    "detective",
    "investigation",
    "investigators",
    "private investigator",
    "private detective",
    "investigation services",
    "security",
    "surveillance",
    "inquiry",
    "advocate",
    "lawyer",
    "criminal lawyer",
    "police",
    "detectives",
]


def normalize_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def detect_profession(name, website, page_text):
    """Classify lead as Police / Private Detective / Lawyer / Security / Unknown."""
    combined = " ".join(filter(None, [normalize_text(name), normalize_text(website), normalize_text(page_text)]))
    if not combined:
        return "Unknown"

    police_keywords = ["police", "police station", "district police", "police department"]
    if any(k in combined for k in police_keywords):
        return "Police"

    lawyer_keywords = ["advocate", "lawyer", "attorney", "legal services", "criminal lawyer", "barrister", "law chamber"]
    if any(k in combined for k in lawyer_keywords):
        return "Lawyer"

    detective_keywords = [
        "private detective",
        "private investigator",
        "detective agency",
        "investigation agency",
        "investigation services",
        "surveillance",
        "inquiry bureau",
        "detectives",
        "investigators",
    ]
    if any(k in combined for k in detective_keywords):
        return "Private Detective"

    security_keywords = ["security services", "security agency", "security solutions", "security guards", "security experts"]
    if any(k in combined for k in security_keywords):
        return "Security"

    return "Unknown"


def fetch_website_text(url):
    """Fetch website page text for classification. Returns text string or empty string."""
    if not url or not isinstance(url, str):
        return ""
    cleaned = url.strip()
    if not cleaned.startswith(("http://", "https://")):
        cleaned = "http://" + cleaned
    try:
        resp = requests.get(cleaned, headers={"User-Agent": USER_AGENT}, timeout=8)
        resp.raise_for_status()
        text = resp.text or ""
        # strip HTML tags roughly
        text = re.sub(r"<script.*?</script>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.lower()
    except Exception:
        return ""


def search_nominatim(name, location):
    """Search OSM Nominatim for a place matching name + location. Returns dict or None."""
    q = f"{name}, {location}" if location else name
    params = {
        "q": q,
        "format": "json",
        "addressdetails": 1,
        "limit": 3,
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        results = resp.json()
        if not results:
            return None
        # return the best match (first)
        return results[0]
    except Exception:
        return None


def check_website_for_keywords(url, name=""):
    """Try to GET the site and search for keywords. Returns (boolean_found, list_of_keywords_found, profession, page_text)."""
    if not url or not isinstance(url, str) or url.strip() == "":
        return False, [], "Unknown", ""
    page_text = fetch_website_text(url)
    found = []
    for k in KEYWORDS:
        if k in page_text:
            found.append(k)
    profession = detect_profession(name, url, page_text)
    return (len(found) > 0), found, profession, page_text


def name_similarity(name_a, name_b):
    """Very small heuristic: compare normalized tokens overlap."""
    if not name_a or not name_b:
        return 0.0
    def toks(s):
        return re.findall(r"[a-z0-9]+", s.lower())
    ta = set(toks(name_a))
    tb = set(toks(name_b))
    if not ta or not tb:
        return 0.0
    inter = ta.intersection(tb)
    score = len(inter) / max(len(ta), len(tb))
    return score


def score_lead(row):
    """Given a pandas Series row with fields, return (score:int, sources:list, profession:str)."""
    sources = []
    score = 0

    name = str(row.get("agency_name", "") or "")
    location = str(row.get("location", "") or "")
    website = str(row.get("website", "") or "")
    phone = str(row.get("phone", "") or "")

    profession = "Unknown"

    # 1) Nominatim lookup
    nom = search_nominatim(name, location)
    time.sleep(NOMINATIM_DELAY)
    if nom:
        display = nom.get("display_name", "").lower()
        sim = name_similarity(name, nom.get("display_name", ""))
        if sim >= 0.35:
            score += 50
            sources.append("nominatim_name_match")
        elif any(k in display for k in KEYWORDS):
            score += 35
            sources.append("nominatim_keyword_match")
        else:
            score += 10
            sources.append("nominatim_found")

    # 2) Website presence & keyword scan + profession detection
    found_site, found_kw, profession, page_text = check_website_for_keywords(website, name)
    if found_site:
        score += 30
        sources.append("website_keyword:" + ",".join(found_kw))
        if profession != "Unknown":
            sources.append(f"profession:{profession}")
    elif website:
        try:
            h = requests.head(website if website.startswith('http') else ('http://' + website), timeout=6, headers={"User-Agent": USER_AGENT})
            if h.status_code < 400:
                score += 10
                sources.append("website_exists")
        except Exception:
            pass

    # 3) phone number presence
    if phone and len(re.sub(r"\D", "", phone)) >= 7:
        score += 10
        sources.append("phone_present")

    # If profession is clearly not private detective but name or website suggests police/lawyer, keep but lower confidence slightly
    if profession == "Police":
        score = max(0, score - 10)
    if profession == "Lawyer":
        score = max(0, score - 5)

    # normalize score to 0..100
    score = max(0, min(100, score))
    return int(score), sources, profession


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="scraped_leads.csv", help="Input CSV path")
    parser.add_argument("--output", default="verified_leads.csv", help="Output CSV path")
    parser.add_argument("--sample", type=int, default=0, help="Only process N rows for testing")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input)
    if args.sample and args.sample > 0:
        df = df.head(args.sample).copy()

    # ensure new columns
    for col in ["verification_status", "verification_score", "verified_sources", "verified_at", "profession"]:
        if col not in df.columns:
            df[col] = ""

    results = []
    total = len(df)
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        print(f"Processing {idx+1}/{total}: {row_dict.get('agency_name', '')} - {row_dict.get('location', '')}")
        score, sources, profession = score_lead(row_dict)
        if score >= 70:
            status = "Verified"
        elif score >= 40:
            status = "Needs Review"
        else:
            status = "Unverified"
        verified_at = datetime.datetime.utcnow().isoformat() + "Z"
        results.append({
            "verification_status": status,
            "verification_score": score,
            "verified_sources": ";".join(sources),
            "verified_at": verified_at,
            "profession": profession,
        })

    res_df = pd.DataFrame(results)
    out = pd.concat([df.reset_index(drop=True), res_df.reset_index(drop=True)], axis=1)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == '__main__':
    main()
