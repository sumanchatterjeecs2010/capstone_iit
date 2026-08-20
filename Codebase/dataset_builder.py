"""
dataset_builder.py
------------------
Five public teaching images (radiology + pathology) plus fictional notes.
"""

import os

import requests

from paths import SAMPLE_DIR
NUM_CASES = 5

USER_AGENT = (
    "HPPCS04-MultimodalAssistant/1.0 "
    "(HPPCS04-MultimodalAssistant/1.0; capstone project)"
)

CASES = {
    1: {
        "filename": "X-ray_of_lobar_pneumonia.jpg",
        "dest": "patient_01.jpg",
        "license": "CC BY-SA 4.0",
        "credit": "Mikael Haggstrom, M.D., Wikimedia Commons, File:X-ray of lobar pneumonia.jpg",
        "domain": "radiology",
        "modality": "chest_xray",
    },
    2: {
        "filename": "Nodular_lymphocyte_predominant_Hodgkin_lymphoma_-_high_mag.jpg",
        "dest": "patient_02.jpg",
        "license": "CC BY-SA 3.0",
        "credit": "Nephron, Wikimedia Commons, File:Nodular lymphocyte predominant Hodgkin lymphoma - high mag.jpg",
        "domain": "pathology",
        "modality": "pathology_slide",
    },
    3: {
        "filename": "Intracerebral.jpg",
        "dest": "patient_03.jpg",
        "license": "CC BY-SA 3.0",
        "credit": "Lucien Monfils, Wikimedia Commons, File:Intracerebral.jpg",
        "domain": "radiology",
        "modality": "brain_ct",
    },
    4: {
        "filename": "Breast_carcinoma_in_a_lymph_node.jpg",
        "dest": "patient_04.jpg",
        "license": "CC BY-SA 3.0",
        "credit": "Nephron, Wikimedia Commons, File:Breast carcinoma in a lymph node.jpg",
        "domain": "pathology",
        "modality": "pathology_slide",
    },
    5: {
        "filename": "Collesfracture.jpg",
        "dest": "patient_05.jpg",
        "license": "CC BY-SA 3.0",
        "credit": "Wikimedia Commons, File:Collesfracture.jpg",
        "domain": "radiology",
        "modality": "bone_xray",
    },
}


def _case_paths(case_id):
    text_path = os.path.join(SAMPLE_DIR, "patient_{:02d}.txt".format(case_id))
    planned = os.path.join(SAMPLE_DIR, CASES[case_id]["dest"])
    if os.path.exists(planned):
        return planned, text_path
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = os.path.join(SAMPLE_DIR, "patient_{:02d}{}".format(case_id, ext))
        if os.path.exists(candidate):
            return candidate, text_path
    return planned, text_path


def _download_image(case_id, overwrite=False):
    meta = CASES[case_id]
    dest = os.path.join(SAMPLE_DIR, meta["dest"])
    if os.path.exists(dest) and not overwrite:
        return dest
    url = "https://commons.wikimedia.org/wiki/Special:FilePath/{}?width=768".format(meta["filename"])
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=90, allow_redirects=True)
    response.raise_for_status()
    content_type = (response.headers.get("Content-Type") or "").lower()
    if "html" in content_type or len(response.content) < 2000:
        raise RuntimeError("Download did not return an image for {}".format(meta["filename"]))
    with open(dest, "wb") as handle:
        handle.write(response.content)
    return dest


def _write_sources():
    path = os.path.join(SAMPLE_DIR, "IMAGE_SOURCES.txt")
    lines = [
        "Public teaching images (radiology and pathology only).",
        "Notes in patient_XX.txt are fictional.",
        "",
    ]
    for case_id in range(1, NUM_CASES + 1):
        meta = CASES[case_id]
        lines.append(
            "patient_{:02d} [{}]: {} | {} | {}".format(
                case_id, meta["domain"], meta["filename"], meta["license"], meta["credit"]
            )
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def prescription_text(case_id):
    notes = {
        1: (
            "PATIENT DETAILS / PRESCRIPTION (fictional teaching case)\n"
            "Case ID: patient_01\n"
            "Domain: radiology\n"
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
            "Domain: pathology\n"
            "Age/Sex: 24-year-old male\n"
            "Chief complaint: Painless cervical lymphadenopathy for 6 weeks.\n"
            "History: Night sweats, low-grade fever, 3 kg weight loss. No HIV.\n"
            "Vitals: HR 88, T 37.8 C, BP 118/76, SpO2 98%.\n"
            "Exam: Firm left cervical node 3 cm. No hepatosplenomegaly documented.\n"
            "Current medicines: None.\n"
            "Requested study: Lymph-node histopathology (H&E).\n"
            "Clinical question: Hodgkin lymphoma versus reactive lymphadenitis.\n"
            "Plan requested: Correlate slide with B symptoms and advise next steps.\n"
            "Image source: public teaching histopathology slide (see IMAGE_SOURCES.txt)."
        ),
        3: (
            "PATIENT DETAILS / PRESCRIPTION (fictional teaching case)\n"
            "Case ID: patient_03\n"
            "Domain: radiology\n"
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
            "Domain: pathology\n"
            "Age/Sex: 54-year-old female\n"
            "Chief complaint: Palpable right breast lump noticed 3 weeks ago.\n"
            "History: No prior breast cancer. Family history: mother with breast carcinoma at 62.\n"
            "Vitals: Stable. No fever.\n"
            "Exam: Firm 2 cm mass, upper outer quadrant, not fixed to chest wall.\n"
            "Current medicines: None.\n"
            "Requested study: Core-needle biopsy histopathology (H&E).\n"
            "Clinical question: Invasive ductal carcinoma versus other breast neoplasm.\n"
            "Plan requested: Correlate histology with the palpable mass and advise oncology pathway.\n"
            "Image source: public teaching histopathology slide (see IMAGE_SOURCES.txt)."
        ),
        5: (
            "PATIENT DETAILS / PRESCRIPTION (fictional teaching case)\n"
            "Case ID: patient_05\n"
            "Domain: radiology\n"
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
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    cases = []
    for case_id in range(1, NUM_CASES + 1):
        image_path = _download_image(case_id, overwrite=overwrite)
        text_path = os.path.join(SAMPLE_DIR, "patient_{:02d}.txt".format(case_id))
        if overwrite or not os.path.exists(text_path):
            with open(text_path, "w", encoding="utf-8") as handle:
                handle.write(prescription_text(case_id))
        cases.append(
            {
                "case_id": case_id,
                "image": os.path.basename(image_path),
                "text": os.path.basename(text_path),
                "domain": CASES[case_id]["domain"],
                "modality": CASES[case_id]["modality"],
                "image_credit": CASES[case_id]["credit"],
            }
        )
    _write_sources()
    return cases


def list_input_pairs():
    pairs = []
    for case_id in range(1, NUM_CASES + 1):
        image_path, text_path = _case_paths(case_id)
        pairs.append((case_id, image_path, text_path))
    return pairs


if __name__ == "__main__":
    created = generate_all_cases(overwrite=True)
    for item in created:
        print("Ready", item)
