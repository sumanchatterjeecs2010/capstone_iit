"""Normalize Llama entity output with a small local clinical dictionary."""

import re


SEX_MAP = {
    "m": "male",
    "male": "male",
    "man": "male",
    "f": "female",
    "female": "female",
    "woman": "female",
}

CONDITION_MAP = {
    "pneumonia": ("community-acquired pneumonia", "COND-PNEUMONIA"),
    "community-acquired pneumonia": ("community-acquired pneumonia", "COND-PNEUMONIA"),
    "consolidation": ("pulmonary consolidation", "COND-CONSOLIDATION"),
    "infiltrate": ("pulmonary infiltrate", "COND-INFILTRATE"),
    "heart failure": ("heart failure", "COND-HF"),
    "pulmonary edema": ("pulmonary edema", "COND-EDEMA"),
    "melanoma": ("cutaneous melanoma", "COND-MELANOMA"),
    "nevus": ("melanocytic nevus", "COND-NEVUS"),
    "stroke": ("acute stroke", "COND-STROKE"),
    "haemorrhage": ("intracranial hemorrhage", "COND-ICH"),
    "hemorrhage": ("intracranial hemorrhage", "COND-ICH"),
    "ischemia": ("cerebral ischemia", "COND-ISCHEMIA"),
    "ischaemia": ("cerebral ischemia", "COND-ISCHEMIA"),
    "retinopathy": ("diabetic retinopathy", "COND-DR"),
    "diabetic retinopathy": ("diabetic retinopathy", "COND-DR"),
    "fracture": ("bone fracture", "COND-FRACTURE"),
    "colles": ("distal radius fracture", "COND-DISTAL-RADIUS"),
}

MEDICATION_MAP = {
    "metformin": ("Metformin", "MED-METFORMIN"),
    "insulin": ("Insulin", "MED-INSULIN"),
    "ramipril": ("Ramipril", "MED-RAMIPRIL"),
    "atorvastatin": ("Atorvastatin", "MED-ATORVASTATIN"),
    "amlodipine": ("Amlodipine", "MED-AMLODIPINE"),
}

STUDY_MAP = {
    "chest radiograph": ("chest_xray", "IMG-CXR", "radiology"),
    "chest x-ray": ("chest_xray", "IMG-CXR", "radiology"),
    "chest xray": ("chest_xray", "IMG-CXR", "radiology"),
    "ct brain": ("brain_ct", "IMG-CT-BRAIN", "radiology"),
    "brain ct": ("brain_ct", "IMG-CT-BRAIN", "radiology"),
    "non-contrast ct": ("brain_ct", "IMG-CT-BRAIN", "radiology"),
    "fundus": ("fundus", "IMG-FUNDUS", "ophthalmology"),
    "wrist": ("bone_xray", "IMG-XR-EXTREMITY", "radiology"),
    "radiograph": ("radiograph", "IMG-XR", "radiology"),
    "histopath": ("pathology_slide", "IMG-PATH", "pathology"),
    "pathology": ("pathology_slide", "IMG-PATH", "pathology"),
    "biopsy": ("pathology_slide", "IMG-PATH", "pathology"),
    "dermatolog": ("dermatology_photo", "IMG-DERM", "dermatology"),
    "clinical photograph": ("dermatology_photo", "IMG-DERM", "dermatology"),
}

ENTITY_KEYS = (
    "age_sex",
    "chief_complaint",
    "history",
    "vitals",
    "exam",
    "medications",
    "requested_study",
    "clinical_question",
)


def _first_int(pattern, text):
    match = re.search(pattern, text, flags=re.I)
    return int(match.group(1)) if match else None


def parse_demographics(blob):
    age = _first_int(r"\b(\d{1,3})\s*[-\s]?year", blob)
    sex = None
    lower = blob.lower()
    for key, value in SEX_MAP.items():
        if re.search(r"\b{}\b".format(key), lower):
            sex = value
            break
    return {"age_years": age, "sex": sex}


def parse_vitals(blob):
    bp = re.search(r"\bBP\s*[:-]?\s*(\d{2,3})\s*/\s*(\d{2,3})", blob, flags=re.I)
    return {
        "heart_rate_bpm": _first_int(r"\bHR\s*[:-]?\s*(\d{2,3})\b", blob),
        "respiratory_rate": _first_int(r"\bRR\s*[:-]?\s*(\d{1,2})\b", blob),
        "spo2_percent": _first_int(r"\bSpO2\s*[:-]?\s*(\d{2,3})", blob),
        "systolic_bp_mmhg": int(bp.group(1)) if bp else None,
        "diastolic_bp_mmhg": int(bp.group(2)) if bp else None,
        "pain_score": _first_int(r"\bPain(?: score)?\s*[:-]?\s*(\d{1,2})", blob),
    }


def _lookup_terms(blob, table):
    found, seen = [], set()
    lower = blob.lower()
    for key, value in sorted(table.items(), key=lambda item: len(item[0]), reverse=True):
        if key in lower and value[1] not in seen:
            seen.add(value[1])
            row = {"name": value[0], "code": value[1]}
            if len(value) > 2:
                row["domain"] = value[2]
            found.append(row)
    return found


def normalize_clinical_entities(extracted, note):
    """Structure Llama entities and map terms to local educational codes."""
    extracted = extracted or {}
    blob = " ".join(str(extracted.get(key) or "") for key in ENTITY_KEYS) + "\n" + (note or "")
    studies = _lookup_terms(blob, STUDY_MAP)
    return {
        "demographics": parse_demographics(blob),
        "vitals": parse_vitals(blob),
        "conditions": _lookup_terms(blob, CONDITION_MAP),
        "medications": _lookup_terms(blob, MEDICATION_MAP),
        "requested_study": studies[0]
        if studies
        else {"name": "unknown", "code": "IMG-UNKNOWN", "domain": "unknown"},
    }
