"""
clinical_references.py
----------------------
Lightweight educational reference links keyed by condition keywords.
These are public guideline / overview pages, not live literature search.
"""

import json


GUIDELINES = [
    {
        "keywords": ["pneumonia", "cap", "consolidation", "infiltrate"],
        "title": "WHO — Community-acquired pneumonia overview",
        "url": "https://www.who.int/news-room/fact-sheets/detail/pneumonia",
    },
    {
        "keywords": ["heart failure", "cardiomegaly", "dyspnoea", "dyspnea"],
        "title": "WHO — Cardiovascular diseases fact sheet",
        "url": "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
    },
    {
        "keywords": ["melanoma", "pigmented", "abcde", "dermatology"],
        "title": "NIH NCI — Melanoma patient information",
        "url": "https://www.cancer.gov/types/skin/patient/melanoma-treatment-pdq",
    },
    {
        "keywords": ["stroke", "hemiparesis", "aphasia", "brain", "ischemia", "hemorrhage"],
        "title": "WHO — Stroke fact sheet",
        "url": "https://www.who.int/news-room/fact-sheets/detail/stroke",
    },
    {
        "keywords": ["retinopathy", "fundus", "diabetes", "ophthalmology"],
        "title": "NIH NEI — Diabetic retinopathy",
        "url": "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy",
    },
    {
        "keywords": ["fracture", "wrist", "bone", "orthopedic", "scaphoid"],
        "title": "NIH MedlinePlus — Fractures overview",
        "url": "https://medlineplus.gov/fractures.html",
    },
]


def attach_references(record):
    """Return up to five guideline links relevant to a case record."""
    blob = json.dumps(record).lower()
    refs, seen = [], set()
    for item in GUIDELINES:
        if not any(kw in blob for kw in item["keywords"]) or item["url"] in seen:
            continue
        refs.append({"title": item["title"], "url": item["url"], "type": "clinical guideline"})
        seen.add(item["url"])
    return refs[:5]
