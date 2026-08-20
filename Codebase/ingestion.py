"""
ingestion.py
------------
Accept a clinical note and a radiology/pathology image, de-identify them, and
place safe copies in an ephemeral work directory for model input.

Supported extensions
--------------------
- Text: ``.txt``, ``.md``
- Image: ``.jpg``, ``.jpeg``, ``.png``, ``.dcm`` / ``.dicom``

Accepted clinical domains (enforced later by MedGemma): ``radiology``, ``pathology``.

Reuse
-----
Prefer ``prepare_existing_paths(text_path, image_path)`` when you already have
files on disk (CLI/TUI). Use ``ingest_pair`` when you hold raw bytes in memory.
"""

import os
import tempfile
import uuid

from privacy import deidentify_image_bytes, deidentify_text


TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dcm", ".dicom"}
IMAGE_DOMAINS = {"radiology", "pathology"}


def _safe_stem(name):
    """Sanitize a filename stem for safe use in a work directory."""
    base = os.path.basename(name or "upload")
    stem, ext = os.path.splitext(base)
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)[:40] or "file"
    return stem, ext.lower()


def _work_dir(session_dir=None):
    """
    Return an ephemeral workspace for de-identified files.

    If *session_dir* is given it is created/reused; otherwise a unique temp dir
    is allocated under the system temporary directory.
    """
    if session_dir:
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
    return tempfile.mkdtemp(prefix="mm_assist_{}_".format(uuid.uuid4().hex[:8]))


def ingest_text(raw, filename, session_dir=None):
    """
    De-identify a clinical note and write it into the work folder.

    Parameters
    ----------
    raw : bytes or str
        Note contents.
    filename : str
        Original filename (used only for extension/stem).
    session_dir : str or None
        Shared work folder; created if omitted.

    Returns
    -------
    dict
        ``kind``, ``path``, ``privacy`` audit, ``session_dir``.
    """
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = raw
    stem, ext = _safe_stem(filename)
    if ext not in TEXT_EXTENSIONS and ext != "":
        raise RuntimeError("Unsupported text type '{}'. Use .txt or .md notes/reports.".format(ext))
    session_dir = _work_dir(session_dir)
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


def ingest_image(raw, filename, session_dir=None):
    """
    De-identify a medical image (strip metadata / DICOM PHI) and write a safe copy.

    Returns
    -------
    dict
        ``kind``, ``path``, ``domain`` (None until MedGemma runs), ``privacy``, ``session_dir``.
    """
    stem, ext = _safe_stem(filename)
    if ext not in IMAGE_EXTENSIONS:
        raise RuntimeError(
            "Unsupported image type '{}'. Use JPEG/PNG (radiology or pathology) or DICOM.".format(ext)
        )
    session_dir = _work_dir(session_dir)
    dest = os.path.join(session_dir, stem + (ext if ext in {".jpg", ".jpeg", ".png"} else ".png"))
    audit = deidentify_image_bytes(raw, filename, dest)
    out_path = audit.get("output_path") or dest
    audit["image_domain"] = "auto"
    return {
        "kind": "image",
        "path": out_path,
        "domain": None,
        "privacy": audit,
        "session_dir": session_dir,
    }


def ingest_pair(text_raw, text_name, image_raw, image_name):
    """
    Ingest a note and an image into one shared temporary work folder.

    Returns
    -------
    dict
        ``session_dir``, ``text`` (ingest_text result), ``image`` (ingest_image result).
    """
    session_dir = _work_dir()
    text_info = ingest_text(text_raw, text_name, session_dir=session_dir)
    image_info = ingest_image(image_raw, image_name, session_dir=session_dir)
    return {
        "session_dir": session_dir,
        "text": text_info,
        "image": image_info,
    }


def prepare_existing_paths(text_path, image_path):
    """
    Load files from disk and run them through the same de-identification gate.

    This is the helper used by ``medical_assistant.process_case``.
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
    )
