"""
ingestion.py
------------
Upload modules for medical text (notes, reports) and images
(radiology, pathology, dermatology, ophthalmology).

Every upload is de-identified before it is written to disk.
Original bytes are not stored.
"""

import os
import uuid

from privacy import deidentify_image_bytes, deidentify_text


ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ROOT = os.path.join(ROOT, "uploads", "processed")

TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dcm", ".dicom"}
IMAGE_DOMAINS = {
    "radiology",
    "pathology",
    "dermatology",
    "ophthalmology",
    "other",
}


def _safe_stem(name):
    base = os.path.basename(name or "upload")
    stem, ext = os.path.splitext(base)
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)[:40] or "file"
    return stem, ext.lower()


def ingest_text(raw, filename, session_dir=None):
    """
    De-identify and store a clinical note / report.

    Parameters
    ----------
    raw : bytes or str
        Uploaded text.
    filename : str
        Original filename.
    session_dir : str or None
        Destination folder.

    Returns
    -------
    dict
        Paths and privacy audit.
    """
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = raw
    stem, ext = _safe_stem(filename)
    if ext not in TEXT_EXTENSIONS and ext != "":
        raise RuntimeError("Unsupported text type '{}'. Use .txt or .md notes/reports.".format(ext))
    session_dir = session_dir or os.path.join(UPLOAD_ROOT, uuid.uuid4().hex[:12])
    os.makedirs(session_dir, exist_ok=True)
    redacted, audit = deidentify_text(text)
    dest = os.path.join(session_dir, stem + ".txt")
    with open(dest, "w", encoding="utf-8") as handle:
        handle.write(redacted)
        if not redacted.endswith("\n"):
            handle.write("\n")
    return {
        "kind": "text",
        "path": dest,
        "privacy": audit,
        "session_dir": session_dir,
    }


def ingest_image(raw, filename, image_domain="radiology", session_dir=None):
    """
    De-identify and store a medical image.

    Parameters
    ----------
    raw : bytes
        Uploaded image or DICOM bytes.
    filename : str
        Original filename.
    image_domain : str
        radiology | pathology | dermatology | ophthalmology | other
    session_dir : str or None
        Destination folder.

    Returns
    -------
    dict
        Paths and privacy audit.
    """
    domain = (image_domain or "other").lower()
    if domain not in IMAGE_DOMAINS:
        raise RuntimeError("Unknown image domain '{}'. Choose: {}".format(domain, ", ".join(sorted(IMAGE_DOMAINS))))
    stem, ext = _safe_stem(filename)
    if ext not in IMAGE_EXTENSIONS:
        raise RuntimeError(
            "Unsupported image type '{}'. Use JPEG/PNG (radiology or pathology) or DICOM.".format(ext)
        )
    session_dir = session_dir or os.path.join(UPLOAD_ROOT, uuid.uuid4().hex[:12])
    os.makedirs(session_dir, exist_ok=True)
    dest = os.path.join(session_dir, stem + (ext if ext in {".jpg", ".jpeg", ".png"} else ".png"))
    audit = deidentify_image_bytes(raw, filename, dest)
    out_path = audit.get("output_path") or dest
    audit["image_domain"] = domain
    return {
        "kind": "image",
        "path": out_path,
        "domain": domain,
        "privacy": audit,
        "session_dir": session_dir,
    }


def ingest_pair(text_raw, text_name, image_raw, image_name, image_domain="radiology"):
    """Ingest a note and an image into one de-identified session folder."""
    session_dir = os.path.join(UPLOAD_ROOT, uuid.uuid4().hex[:12])
    text_info = ingest_text(text_raw, text_name, session_dir=session_dir)
    image_info = ingest_image(image_raw, image_name, image_domain=image_domain, session_dir=session_dir)
    return {
        "session_dir": session_dir,
        "text": text_info,
        "image": image_info,
    }


def prepare_existing_paths(text_path, image_path, image_domain="radiology"):
    """
    Run local files through the same de-identification gate used for uploads.
    """
    with open(text_path, "r", encoding="utf-8") as handle:
        text_raw = handle.read()
    with open(image_path, "rb") as handle:
        image_raw = handle.read()
    return ingest_pair(
        text_raw,
        os.path.basename(text_path),
        image_raw,
        os.path.basename(image_path),
        image_domain=image_domain,
    )
