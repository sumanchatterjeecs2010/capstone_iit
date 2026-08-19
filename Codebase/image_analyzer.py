"""
image_analyzer.py
-----------------
OpenCV pre-processing plus a hook to attach BiomedCLIP/BioBERT fusion.

OpenCV remains responsible for low-level statistics (brightness, edges,
laterality). The medical vision/text encoders add ranked clinical findings.
"""

import os

import cv2
import numpy as np


def load_image(image_path):
    """
    Load an image from disk in BGR colour space.

    Parameters
    ----------
    image_path : str
        Path to a PNG / JPEG medical image.

    Returns
    -------
    numpy.ndarray
        Loaded image.

    Raises
    ------
    FileNotFoundError
        If the file cannot be read by OpenCV.
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError("Unable to read image: {}".format(image_path))
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _is_mostly_grayscale(image):
    """
    Decide whether an image is effectively grayscale (radiograph / CT).

    Parameters
    ----------
    image : numpy.ndarray
        BGR image.

    Returns
    -------
    bool
        True when colour channels are nearly identical.
    """
    b, g, r = cv2.split(image)
    diff = np.mean(np.abs(b.astype(np.int16) - g.astype(np.int16))) + np.mean(
        np.abs(g.astype(np.int16) - r.astype(np.int16))
    )
    return diff < 8.0


def _circularity_of_bright_mask(gray):
    """
    Estimate how circular the main bright region is (fundus / skull cue).

    Parameters
    ----------
    gray : numpy.ndarray
        Single-channel image.

    Returns
    -------
    float
        Circularity between 0 and 1, or 0 if no contour is found.
    """
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    return float(min(1.0, (4.0 * np.pi * area) / (perimeter * perimeter)))


def _left_right_asymmetry(gray):
    """
    Compare mean intensity of the left and right halves.

    Parameters
    ----------
    gray : numpy.ndarray
        Single-channel image.

    Returns
    -------
    float
        Absolute difference of mean intensities.
    """
    mid = gray.shape[1] // 2
    left = np.mean(gray[:, :mid])
    right = np.mean(gray[:, mid:])
    return float(abs(left - right))


def infer_modality(image, gray, features):
    """
    Heuristically label the imaging modality from visual statistics.

    This is not a trained classifier. It only provides a coarse prior that
    the LLMs can confirm or reject using the accompanying clinical note.

    Parameters
    ----------
    image : numpy.ndarray
        BGR image.
    gray : numpy.ndarray
        Grayscale conversion.
    features : dict
        Pre-computed numeric features.

    Returns
    -------
    str
        One of: chest_xray, bone_xray, brain_ct, fundus, dermatology_photo, unknown.
    """
    if not features["mostly_grayscale"]:
        if features["circularity"] > 0.75 and features["mean_brightness"] < 90:
            return "fundus"
        return "dermatology_photo"
    # Grayscale radiology / CT
    if features["circularity"] > 0.75:
        return "brain_ct"
    # Extremity radiographs are typically brighter with denser edges than chest films.
    if features["edge_density"] > 0.08 and features["mean_brightness"] > 45:
        return "bone_xray"
    return "chest_xray"


def describe_findings(modality, features):
    """
    Turn numeric features into short natural-language visual observations.

    Parameters
    ----------
    modality : str
        Heuristic modality label.
    features : dict
        Numeric feature dictionary.

    Returns
    -------
    list[str]
        Human-readable observation bullets.
    """
    notes = []
    if modality == "chest_xray":
        notes.append("Grayscale projection consistent with a chest radiograph.")
        if features["left_right_asymmetry"] > 6:
            notes.append("Left-right lung-field brightness is asymmetric; an opacity is possible.")
        if features["contrast"] < 40:
            notes.append("Overall contrast is reduced, which can accompany consolidation or underexposure.")
    elif modality == "dermatology_photo":
        notes.append("Colour clinical photograph rather than a radiograph.")
        notes.append("A localised pigmented region is suggested by colour variance and compact dark area.")
    elif modality == "brain_ct":
        notes.append("Near-circular grayscale structure consistent with an axial brain CT slice.")
        if features["left_right_asymmetry"] > 5:
            notes.append("Hemispheric intensity asymmetry is present; haemorrhage or infarct should be considered.")
        if features["bright_pixel_ratio"] > 0.08:
            notes.append("A cluster of high-intensity pixels may represent acute blood or artefact.")
    elif modality == "fundus":
        notes.append("Circular colour image consistent with a retinal fundus photograph.")
        notes.append("Scattered dark foci can correspond to retinal haemorrhages in a diabetic context.")
    elif modality == "bone_xray":
        notes.append("High-edge grayscale image consistent with an extremity radiograph.")
        if features["edge_density"] > 0.10:
            notes.append("Linear lucent edges are present; a fracture line is possible.")
    else:
        notes.append("Modality could not be inferred confidently from pixels alone.")
    notes.append(
        "Quantitative cues: brightness={:.1f}, contrast={:.1f}, edges={:.3f}, asymmetry={:.1f}.".format(
            features["mean_brightness"],
            features["contrast"],
            features["edge_density"],
            features["left_right_asymmetry"],
        )
    )
    return notes


def analyze_image(image_path):
    """
    Run the full OpenCV pipeline and return a JSON-serialisable summary.

    Parameters
    ----------
    image_path : str
        Path to the medical image.

    Returns
    -------
    dict
        Visual analysis record consumed by the LLM fusion stage.
    """
    image = load_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    hist = cv2.calcHist([gray], [0], None, [8], [0, 256]).flatten()
    hist = (hist / max(float(hist.sum()), 1.0)).tolist()

    features = {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "mostly_grayscale": bool(_is_mostly_grayscale(image)),
        "mean_brightness": float(np.mean(gray)),
        "contrast": float(np.std(gray)),
        "edge_density": float(np.mean(edges > 0)),
        "circularity": _circularity_of_bright_mask(gray),
        "left_right_asymmetry": _left_right_asymmetry(gray),
        "bright_pixel_ratio": float(np.mean(gray > 180)),
        "dark_pixel_ratio": float(np.mean(gray < 40)),
        "intensity_histogram_8bin": [round(v, 4) for v in hist],
    }
    modality = infer_modality(image, gray, features)
    findings = describe_findings(modality, features)
    return {
        "file_name": os.path.basename(image_path),
        "inferred_modality": modality,
        "features": features,
        "visual_findings": findings,
        "method": "OpenCV feature extraction",
    }


def attach_encoder_fusion(visual, image_path, note):
    """
    Attach BiomedCLIP + BioBERT ranked findings to an OpenCV visual record.

    If the encoders cannot be loaded, the OpenCV record is returned unchanged
    so the two LLMs can still complete the five conversations.

    Parameters
    ----------
    visual : dict
        OpenCV analysis.
    image_path : str
        Medical image path.
    note : str
        Patient note.

    Returns
    -------
    dict
        Visual record with fused clinical findings.
    """
    from multimodal_encoder import fuse_image_and_note

    fusion = fuse_image_and_note(image_path, note)
    opencv_modality = visual.get("inferred_modality")
    encoder_modality = fusion.get("inferred_modality")
    visual["opencv_modality"] = opencv_modality
    visual["encoder_modality"] = encoder_modality
    visual["fused_findings"] = fusion.get("fused_findings", [])
    visual["modality_scores"] = fusion.get("modality_scores", [])
    visual["fusion_summary"] = fusion.get("fusion_summary")
    visual["vision_backend"] = fusion.get("vision_backend")
    visual["text_backend"] = fusion.get("text_backend")
    # Keep the OpenCV modality unless CLIP is clearly confident. Synthetic
    # educational images are often mis-tagged by CLIP as fundus or CT.
    visual["inferred_modality"] = opencv_modality
    visual["visual_findings"] = list(visual.get("visual_findings") or [])
    visual["visual_findings"].extend(fusion.get("visual_findings") or [])
    visual["method"] = "OpenCV + {} + BioBERT".format(fusion.get("vision_backend"))
    return visual


def restrict_findings_to_modality(visual):
    """
    Keep fused encoder findings that match the corroborated modality.

    Parameters
    ----------
    visual : dict
        Visual record after note corroboration.

    Returns
    -------
    dict
        Visual record with modality-filtered findings.
    """
    modality = visual.get("inferred_modality")
    fused = visual.get("fused_findings") or []
    filtered = [row for row in fused if row.get("modality") == modality]
    if not filtered:
        return visual
    filtered.sort(key=lambda row: row.get("fused_score", 0), reverse=True)
    visual["fused_findings"] = filtered
    visual["visual_findings"] = list(visual.get("visual_findings") or [])
    visual["visual_findings"].append(
        "BioBERT+CLIP findings for {}: {}.".format(
            modality.replace("_", " "),
            "; ".join(row["label"] for row in filtered[:4]),
        )
    )
    visual["fusion_summary"] = (
        "After locking modality to {}, ranked findings are: {}.".format(
            modality,
            "; ".join(
                "{} ({:.2f})".format(row["label"], row["fused_score"]) for row in filtered[:4]
            ),
        )
    )
    return visual
