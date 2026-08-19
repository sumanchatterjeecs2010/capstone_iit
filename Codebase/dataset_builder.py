"""
dataset_builder.py
------------------
Creates the five synthetic (de-identified) multimodal input pairs required
by HPPCS[04]: one medical image and one matching prescription / patient-detail
text file per case.

All images are procedurally generated. No real patient data is used.
"""

import os

import cv2
import numpy as np


# Directory that contains this file (Codebase/). All paths stay local ("./").
ROOT = os.path.dirname(os.path.abspath(__file__))

# Number of cases required by the project specification.
NUM_CASES = 5

# Expected clinical cues used later for lightweight keyword evaluation.
# These labels are known because the images and notes are synthetic.
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
    """
    Return the image and text filenames for a given case number.

    Parameters
    ----------
    case_id : int
        Case index from 1 to 5.

    Returns
    -------
    tuple[str, str]
        Absolute paths of the PNG image and TXT prescription file.
    """
    image_path = os.path.join(ROOT, "patient_{:02d}.png".format(case_id))
    text_path = os.path.join(ROOT, "patient_{:02d}.txt".format(case_id))
    return image_path, text_path


def _add_gaussian_noise(image, sigma):
    """
    Add light Gaussian noise so synthetic images look less cartoon-like.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (uint8).
    sigma : float
        Standard deviation of the noise.

    Returns
    -------
    numpy.ndarray
        Noisy image clipped to the valid 0-255 range.
    """
    noise = np.random.normal(0, sigma, image.shape)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def build_chest_xray():
    """
    Build a synthetic frontal chest radiograph with a right-sided opacity.

    Returns
    -------
    numpy.ndarray
        Grayscale PNG-ready image (256 x 256).
    """
    img = np.zeros((256, 256), dtype=np.uint8)
    img[:] = 18
    # Soft tissue / mediastinum
    cv2.ellipse(img, (128, 140), (42, 70), 0, 0, 360, 70, -1)
    # Lung fields
    cv2.ellipse(img, (78, 130), (50, 78), 0, 0, 360, 110, -1)
    cv2.ellipse(img, (178, 130), (50, 78), 0, 0, 360, 110, -1)
    # Rib-like arcs
    for i, y in enumerate(range(60, 200, 16)):
        cv2.ellipse(img, (128, y), (92, 18 + i), 0, 200, 340, 150, 1)
    # Simulated right lower-zone consolidation (opacity)
    cv2.ellipse(img, (175, 175), (28, 22), 15, 0, 360, 55, -1)
    cv2.GaussianBlur(img, (9, 9), 0, img)
    return _add_gaussian_noise(img, 4)


def build_skin_lesion():
    """
    Build a synthetic close-up photograph of an irregular pigmented lesion.

    Returns
    -------
    numpy.ndarray
        BGR colour image (256 x 256).
    """
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :] = (170, 190, 220)  # light skin tone in BGR
    # Irregular dark lesion (asymmetric, colour variation)
    cv2.ellipse(img, (128, 132), (38, 28), 25, 0, 360, (20, 30, 40), -1)
    cv2.ellipse(img, (142, 120), (16, 18), -10, 0, 360, (10, 15, 70), -1)
    cv2.circle(img, (118, 140), 8, (30, 40, 90), -1)
    cv2.GaussianBlur(img, (7, 7), 0, img)
    return _add_gaussian_noise(img, 3)


def build_brain_ct():
    """
    Build a synthetic axial brain CT slice with a hyperdense wedge.

    Returns
    -------
    numpy.ndarray
        Grayscale image (256 x 256).
    """
    img = np.zeros((256, 256), dtype=np.uint8)
    img[:] = 10
    cv2.circle(img, (128, 128), 110, 40, -1)  # skull
    cv2.circle(img, (128, 128), 98, 90, -1)  # parenchyma
    # Midline
    cv2.line(img, (128, 40), (128, 216), 70, 2)
    # Simulated acute hemorrhage / dense lesion on the left
    pts = np.array([[70, 90], [110, 80], [115, 140], [60, 150]], np.int32)
    cv2.fillConvexPoly(img, pts, 200)
    cv2.GaussianBlur(img, (7, 7), 0, img)
    return _add_gaussian_noise(img, 3)


def build_fundus():
    """
    Build a synthetic retinal fundus photograph with blot-haemorrhage cues.

    Returns
    -------
    numpy.ndarray
        BGR colour image (256 x 256).
    """
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(img, (128, 128), 120, (20, 40, 120), -1)
    cv2.circle(img, (128, 128), 118, (15, 55, 160), -1)
    # Optic disc
    cv2.circle(img, (168, 128), 16, (80, 160, 220), -1)
    # Vessel-like branches
    for dx, dy in [(-50, -40), (-55, 35), (40, -45), (30, 50), (-20, -70)]:
        cv2.line(img, (168, 128), (168 + dx, 128 + dy), (10, 20, 80), 2)
    # Dot-blot haemorrhages
    for cx, cy in [(90, 100), (100, 150), (80, 140), (110, 90)]:
        cv2.circle(img, (cx, cy), 4, (10, 10, 40), -1)
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 120, 255, -1)
    img[mask == 0] = (0, 0, 0)
    cv2.GaussianBlur(img, (5, 5), 0, img)
    return _add_gaussian_noise(img, 2)


def build_wrist_xray():
    """
    Build a synthetic wrist radiograph with a lucent fracture line.

    Returns
    -------
    numpy.ndarray
        Grayscale image (256 x 256).
    """
    img = np.zeros((256, 256), dtype=np.uint8)
    img[:] = 25
    # Radius and ulna shafts
    cv2.rectangle(img, (90, 40), (120, 170), 170, -1)
    cv2.rectangle(img, (140, 50), (165, 175), 160, -1)
    # Distal radius flare
    cv2.ellipse(img, (105, 185), (28, 22), 0, 0, 360, 190, -1)
    # Carpal bones
    for i, x in enumerate(range(80, 180, 22)):
        cv2.circle(img, (x, 220), 10, 175, -1)
    # Fracture lucency through distal radius
    cv2.line(img, (88, 168), (128, 188), 40, 2)
    cv2.line(img, (92, 40), (92, 170), 210, 1)
    cv2.line(img, (118, 40), (118, 170), 210, 1)
    return _add_gaussian_noise(img, 4)


def prescription_text(case_id):
    """
    Return the de-identified prescription / patient-detail note for a case.

    Parameters
    ----------
    case_id : int
        Case index from 1 to 5.

    Returns
    -------
    str
        Plain-text clinical note.
    """
    notes = {
        1: (
            "PATIENT DETAILS / PRESCRIPTION (synthetic, de-identified)\n"
            "Case ID: patient_01\n"
            "Age/Sex: 62-year-old male\n"
            "Chief complaint: Productive cough, fever 38.6 C, and dyspnoea for 4 days.\n"
            "History: Type 2 diabetes mellitus, former smoker (20 pack-years).\n"
            "Vitals: HR 108, RR 24, SpO2 91% on room air, BP 138/84 mmHg.\n"
            "Exam: Right basal crepitations, no wheeze.\n"
            "Current medicines: Metformin 500 mg twice daily.\n"
            "Requested study: Chest radiograph (PA).\n"
            "Clinical question: Community-acquired pneumonia versus heart failure.\n"
            "Plan requested: Correlate image with symptoms and advise triage / next steps."
        ),
        2: (
            "PATIENT DETAILS / PRESCRIPTION (synthetic, de-identified)\n"
            "Case ID: patient_02\n"
            "Age/Sex: 47-year-old female\n"
            "Chief complaint: Changing dark mole on the left forearm for 3 months.\n"
            "History: Fair skin, several sunburns in childhood, no prior skin cancer.\n"
            "Exam: Asymmetric pigmented lesion ~9 mm, irregular border, colour variation.\n"
            "ABCDE: Asymmetry yes; Border irregular; Colour mixed; Diameter >6 mm; Evolving yes.\n"
            "Current medicines: None.\n"
            "Requested study: Clinical photograph of the lesion.\n"
            "Clinical question: Suspicious pigmented lesion. Need risk stratification.\n"
            "Plan requested: Correlate image with history and recommend dermatology pathway."
        ),
        3: (
            "PATIENT DETAILS / PRESCRIPTION (synthetic, de-identified)\n"
            "Case ID: patient_03\n"
            "Age/Sex: 71-year-old male\n"
            "Chief complaint: Sudden right-sided weakness and speech difficulty for 40 minutes.\n"
            "History: Hypertension, atrial fibrillation (not on anticoagulation).\n"
            "Vitals: BP 188/102, HR 96 irregular, SpO2 97%, NIHSS-like deficits present.\n"
            "Exam: Right hemiparesis, facial droop, aphasia. Symptom onset witnessed.\n"
            "Current medicines: Amlodipine 5 mg daily.\n"
            "Requested study: Non-contrast CT brain.\n"
            "Clinical question: Acute stroke. Haemorrhage versus ischaemia.\n"
            "Plan requested: Urgent image-text correlation and emergency triage."
        ),
        4: (
            "PATIENT DETAILS / PRESCRIPTION (synthetic, de-identified)\n"
            "Case ID: patient_04\n"
            "Age/Sex: 58-year-old female\n"
            "Chief complaint: Gradual blurring of vision in both eyes for 6 months.\n"
            "History: Type 2 diabetes mellitus for 14 years, HbA1c 9.2%, hypertension.\n"
            "Vitals: BP 152/90, BMI 31.\n"
            "Exam: Reduced visual acuity 6/18 both eyes. No pain or flashing lights.\n"
            "Current medicines: Insulin, ramipril, atorvastatin.\n"
            "Requested study: Fundus photograph (left eye).\n"
            "Clinical question: Diabetic retinopathy screening / grading support.\n"
            "Plan requested: Correlate fundus appearance with diabetic history."
        ),
        5: (
            "PATIENT DETAILS / PRESCRIPTION (synthetic, de-identified)\n"
            "Case ID: patient_05\n"
            "Age/Sex: 29-year-old male\n"
            "Chief complaint: Pain and swelling of the left wrist after a FOOSH fall.\n"
            "History: Fall on outstretched hand while cycling 3 hours ago. No numbness.\n"
            "Vitals: Stable. Pain score 7/10.\n"
            "Exam: Distal radius tenderness, limited wrist motion, snuffbox mildly tender.\n"
            "Current medicines: None. Last tetanus unknown.\n"
            "Requested study: Left wrist radiograph.\n"
            "Clinical question: Distal radius or scaphoid fracture.\n"
            "Plan requested: Correlate radiograph with trauma history and advise care."
        ),
    }
    return notes[case_id]


def generate_all_cases(overwrite=True):
    """
    Write all five image-text pairs into the Codebase directory.

    Parameters
    ----------
    overwrite : bool
        If False, skip files that already exist.

    Returns
    -------
    list[dict]
        Metadata for each generated case (paths and modality).
    """
    np.random.seed(42)
    builders = {
        1: build_chest_xray,
        2: build_skin_lesion,
        3: build_brain_ct,
        4: build_fundus,
        5: build_wrist_xray,
    }
    cases = []
    for case_id in range(1, NUM_CASES + 1):
        image_path, text_path = _case_paths(case_id)
        if overwrite or not os.path.exists(image_path):
            image = builders[case_id]()
            cv2.imwrite(image_path, image)
        if overwrite or not os.path.exists(text_path):
            with open(text_path, "w", encoding="utf-8") as handle:
                handle.write(prescription_text(case_id))
        cases.append(
            {
                "case_id": case_id,
                "image": os.path.basename(image_path),
                "text": os.path.basename(text_path),
                "modality": GROUND_TRUTH[case_id]["modality"],
            }
        )
    return cases


def list_input_pairs():
    """
    List the five input pairs that the assistant must process.

    Returns
    -------
    list[tuple[int, str, str]]
        Tuples of (case_id, image_path, text_path).
    """
    pairs = []
    for case_id in range(1, NUM_CASES + 1):
        image_path, text_path = _case_paths(case_id)
        pairs.append((case_id, image_path, text_path))
    return pairs


if __name__ == "__main__":
    # Allow regenerating the dataset independently of the full assistant run.
    created = generate_all_cases(overwrite=True)
    for item in created:
        print("Created", item)
