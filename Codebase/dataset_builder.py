"""
dataset_builder.py
------------------
Five public teaching images plus fictional, de-identified notes.

Images are downloaded from Wikimedia Commons / NIH public collections.
They are real clinical photographs and radiographs published for education.
The accompanying notes are invented teaching cases (not real patient records).

No OpenCV or Pillow is used.
"""

import os

import requests


ROOT = os.path.dirname(os.path.abspath(__file__))
NUM_CASES = 5

USER_AGENT = (
    "HPPCS04-MultimodalAssistant/1.0 "
    "(educational capstone; https://github.com/sumanchatterjeecs2010/capstone_iit)"
)

# Wikimedia Special:FilePath follows the current stored file.
CASES = {
    1: {
        "filename": "X-ray_of_lobar_pneumonia.jpg",
        "dest": "patient_01.jpg",
        "license": "CC BY-SA 4.0",
        "credit": "Mikael Haggstrom, M.D., Wikimedia Commons, File:X-ray of lobar pneumonia.jpg",
    },
    2: {
        "filename": "Melanoma.jpg",
        "dest": "patient_02.jpg",
        "license": "Public domain (NCI)",
        "credit": "National Cancer Institute via Wikimedia Commons, File:Melanoma.jpg",
    },
    3: {
        "filename": "Intracerebral.jpg",
        "dest": "patient_03.jpg",
        "license": "CC BY-SA 3.0",
        "credit": "Lucien Monfils, Wikimedia Commons, File:Intracerebral.jpg",
    },
    4: {
        "filename": "Fundus_retinopathy_EDA03.JPG",
        "dest": "patient_04.jpg",
        "license": "Public domain (NIH/NEI)",
        "credit": "National Eye Institute, NIH, Wikimedia Commons, File:Fundus retinopathy EDA03.JPG",
    },
    5: {
        "filename": "Collesfracture.jpg",
        "dest": "patient_05.jpg",
        "license": "CC BY-SA 3.0",
        "credit": "Wikimedia Commons, File:Collesfracture.jpg",
    },
}

GROUND_TRUTH = {
    1: {
        "modality": "chest_xray",
        "condition_keywords": [
            "pneumonia",
            "infiltrate",
            "opacity",
            "consolidation",
            "respiratory",
            "chest",
        ],
        "triage": "urgent",
    },
    2: {
        "modality": "dermatology_photo",
        "condition_keywords": [
            "melanoma",
            "lesion",
            "pigmented",
            "abcde",
            "dermatology",
            "biopsy",
        ],
        "triage": "soon",
    },
    3: {
        "modality": "brain_ct",
        "condition_keywords": [
            "stroke",
            "neurologic",
            "hemorrhage",
            "ischemia",
            "brain",
            "emergency",
        ],
        "triage": "emergency",
    },
    4: {
        "modality": "fundus",
        "condition_keywords": [
            "retinopathy",
            "diabetes",
            "fundus",
            "hemorrhage",
            "ophthalmology",
            "retina",
        ],
        "triage": "soon",
    },
    5: {
        "modality": "bone_xray",
        "condition_keywords": [
            "fracture",
            "wrist",
            "fall",
            "orthopedic",
            "immobil",
            "bone",
        ],
        "triage": "urgent",
    },
}


def _case_paths(case_id):
    """Return image and note paths for a case, finding the downloaded image file."""
    text_path = os.path.join(ROOT, "patient_{:02d}.txt".format(case_id))
    planned = os.path.join(ROOT, CASES[case_id]["dest"])
    if os.path.exists(planned):
        return planned, text_path
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = os.path.join(ROOT, "patient_{:02d}{}".format(case_id, ext))
        if os.path.exists(candidate):
            return candidate, text_path
    return planned, text_path


def _download_image(case_id, overwrite=False):
    """Download one Wikimedia Commons file into Codebase/."""
    meta = CASES[case_id]
    dest = os.path.join(ROOT, meta["dest"])
    if os.path.exists(dest) and not overwrite:
        return dest
    url = "https://commons.wikimedia.org/wiki/Special:FilePath/{}?width=768".format(
        meta["filename"]
    )
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=90,
        allow_redirects=True,
    )
    response.raise_for_status()
    content_type = (response.headers.get("Content-Type") or "").lower()
    if "html" in content_type or len(response.content) < 2000:
        raise RuntimeError("Download did not return an image for {}".format(meta["filename"]))
    with open(dest, "wb") as handle:
        handle.write(response.content)
    return dest


def _write_sources():
    """Write license credits next to the images."""
    path = os.path.join(ROOT, "IMAGE_SOURCES.txt")
    lines = [
        "Public teaching images used by this project.",
        "Notes in patient_XX.txt are fictional. Do not treat them as the original patients.",
        "",
    ]
    for case_id in range(1, NUM_CASES + 1):
        meta = CASES[case_id]
        lines.append(
            "patient_{:02d}: {} | {} | {}".format(
                case_id, meta["filename"], meta["license"], meta["credit"]
            )
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def prescription_text(case_id):
    """Return a fictional teaching note matched to the image modality."""
    notes = {
        1: (
            "PATIENT DETAILS / PRESCRIPTION (fictional teaching case)\n"
            "Case ID: patient_01\n"
            "Age/Sex: 62-year-old male\n"
            "Chief complaint: Productive cough, fever 38.6 C, and dyspnoea for 4 days.\n"
            "History: Type 2 diabetes mellitus, former smoker (20 pack-years).\n"
            "Vitals: HR 108, RR 24, SpO2 91% on room air, BP 138/84 mmHg.\n"
            "Exam: Right basal crepitations, no wheeze.\n"
            "Current medicines: Metformin 500 mg twice daily.\n"
            "Requested study: Chest radiograph (PA).\n"
            "Clinical question: Community-acquired pneumonia versus heart failure.\n"
            "Plan requested: Correlate image with symptoms and advise triage / next steps.\n"
            "Image source: public teaching radiograph (see IMAGE_SOURCES.txt)."
        ),
        2: (
            "PATIENT DETAILS / PRESCRIPTION (fictional teaching case)\n"
            "Case ID: patient_02\n"
            "Age/Sex: 47-year-old female\n"
            "Chief complaint: Changing dark mole on the left forearm for 3 months.\n"
            "History: Fair skin, several sunburns in childhood, no prior skin cancer.\n"
            "Exam: Asymmetric pigmented lesion ~9 mm, irregular border, colour variation.\n"
            "ABCDE: Asymmetry yes; Border irregular; Colour mixed; Diameter >6 mm; Evolving yes.\n"
            "Current medicines: None.\n"
            "Requested study: Clinical photograph of the lesion.\n"
            "Clinical question: Suspicious pigmented lesion. Need risk stratification.\n"
            "Plan requested: Correlate image with history and recommend dermatology pathway.\n"
            "Image source: public teaching photograph (see IMAGE_SOURCES.txt)."
        ),
        3: (
            "PATIENT DETAILS / PRESCRIPTION (fictional teaching case)\n"
            "Case ID: patient_03\n"
            "Age/Sex: 71-year-old male\n"
            "Chief complaint: Sudden right-sided weakness and speech difficulty for 40 minutes.\n"
            "History: Hypertension, atrial fibrillation (not on anticoagulation).\n"
            "Vitals: BP 188/102, HR 96 irregular, SpO2 97%, NIHSS-like deficits present.\n"
            "Exam: Right hemiparesis, facial droop, aphasia. Symptom onset witnessed.\n"
            "Current medicines: Amlodipine 5 mg daily.\n"
            "Requested study: Non-contrast CT brain.\n"
            "Clinical question: Acute stroke. Haemorrhage versus ischaemia.\n"
            "Plan requested: Urgent image-text correlation and emergency triage.\n"
            "Image source: public teaching CT (see IMAGE_SOURCES.txt)."
        ),
        4: (
            "PATIENT DETAILS / PRESCRIPTION (fictional teaching case)\n"
            "Case ID: patient_04\n"
            "Age/Sex: 58-year-old female\n"
            "Chief complaint: Gradual blurring of vision in both eyes for 6 months.\n"
            "History: Type 2 diabetes mellitus for 14 years, HbA1c 9.2%, hypertension.\n"
            "Vitals: BP 152/90, BMI 31.\n"
            "Exam: Reduced visual acuity 6/18 both eyes. No pain or flashing lights.\n"
            "Current medicines: Insulin, ramipril, atorvastatin.\n"
            "Requested study: Fundus photograph (left eye).\n"
            "Clinical question: Diabetic retinopathy screening / grading support.\n"
            "Plan requested: Correlate fundus appearance with diabetic history.\n"
            "Image source: public teaching fundus photograph (see IMAGE_SOURCES.txt)."
        ),
        5: (
            "PATIENT DETAILS / PRESCRIPTION (fictional teaching case)\n"
            "Case ID: patient_05\n"
            "Age/Sex: 29-year-old male\n"
            "Chief complaint: Pain and swelling of the left wrist after a FOOSH fall.\n"
            "History: Fall on outstretched hand while cycling 3 hours ago. No numbness.\n"
            "Vitals: Stable. Pain score 7/10.\n"
            "Exam: Distal radius tenderness, limited wrist motion, snuffbox mildly tender.\n"
            "Current medicines: None. Last tetanus unknown.\n"
            "Requested study: Left wrist radiograph.\n"
            "Clinical question: Distal radius or scaphoid fracture.\n"
            "Plan requested: Correlate radiograph with trauma history and advise care.\n"
            "Image source: public teaching radiograph (see IMAGE_SOURCES.txt)."
        ),
    }
    return notes[case_id]


def generate_all_cases(overwrite=False):
    """
    Download public teaching images (if missing) and write fictional notes.

    Parameters
    ----------
    overwrite : bool
        If True, re-download images and rewrite notes.

    Returns
    -------
    list[dict]
        Metadata for each case.
    """
    cases = []
    for case_id in range(1, NUM_CASES + 1):
        image_path = _download_image(case_id, overwrite=overwrite)
        text_path = os.path.join(ROOT, "patient_{:02d}.txt".format(case_id))
        if overwrite or not os.path.exists(text_path):
            with open(text_path, "w", encoding="utf-8") as handle:
                handle.write(prescription_text(case_id))
        cases.append(
            {
                "case_id": case_id,
                "image": os.path.basename(image_path),
                "text": os.path.basename(text_path),
                "modality": GROUND_TRUTH[case_id]["modality"],
                "image_credit": CASES[case_id]["credit"],
            }
        )
    _write_sources()
    return cases


def list_input_pairs():
    """List the five input pairs that the assistant must process."""
    pairs = []
    for case_id in range(1, NUM_CASES + 1):
        image_path, text_path = _case_paths(case_id)
        pairs.append((case_id, image_path, text_path))
    return pairs


if __name__ == "__main__":
    created = generate_all_cases(overwrite=True)
    for item in created:
        print("Ready", item)
