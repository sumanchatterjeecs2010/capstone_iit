"""De-identify clinical text and strip image metadata before storage or LLM use."""

import os
import re
import struct
import zlib


REDACTION_PATTERNS = [
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("phone", re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("mrn", re.compile(r"\b(?:MRN|Medical Record(?: Number)?|Patient ID|Chart #)\s*[:#]?\s*[A-Z0-9-]{4,}\b", re.I)),
    ("insurance_id", re.compile(r"\b(?:Member ID|Policy(?: Number)?|Insurance ID)\s*[:#]?\s*[A-Z0-9-]{5,}\b", re.I)),
    ("date", re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")),
    ("labeled_name", re.compile(r"\b(?:Patient Name|Name|Next of Kin)\s*:\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")),
    ("labeled_address", re.compile(r"\b(?:Address|Street|Residence)\s*:\s*.+$", re.M | re.I)),
    ("zipcode", re.compile(r"\b\d{5}(?:-\d{4})?\b")),
]


def deidentify_text(text):
    """Redact common direct identifiers; return (clean_text, audit)."""
    if not text:
        return "", {"n_redactions": 0, "by_type": {}}
    cleaned = text
    by_type = {}
    for label, pattern in REDACTION_PATTERNS:
        matches = pattern.findall(cleaned)
        if not matches:
            continue
        by_type[label] = by_type.get(label, 0) + len(matches)
        cleaned = pattern.sub("[{}]".format(label.upper()), cleaned)
    cleaned = _redact_age_over_89(cleaned, by_type)
    return cleaned.strip(), {
        "n_redactions": int(sum(by_type.values())),
        "by_type": by_type,
        "method": "regex HIPAA-inspired educational scrubber",
        "stored_original": False,
    }


def _redact_age_over_89(text, by_type):
    def repl(match):
        years = int(match.group(1))
        if years > 89:
            by_type["age_over_89"] = by_type.get("age_over_89", 0) + 1
            return match.group(0).replace(str(years), "[AGE_OVER_89]")
        return match.group(0)

    return re.sub(r"\b(\d{2,3})[ -]?year[ -]?old\b", repl, text, flags=re.I)


def strip_jpeg_metadata(data):
    """Remove JPEG APP/COM segments (EXIF, comments)."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return data
    out = bytearray(b"\xff\xd8")
    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            out.extend(data[i:])
            break
        while i < len(data) and data[i] == 0xFF:
            i += 1
        if i >= len(data):
            break
        marker = data[i]
        i += 1
        if marker == 0xD9:
            out.extend(b"\xff\xd9")
            break
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:
            out.extend(bytes((0xFF, marker)))
            continue
        if marker == 0xDA:
            out.extend(bytes((0xFF, marker)))
            out.extend(data[i:])
            break
        if i + 2 > len(data):
            break
        length = int.from_bytes(data[i : i + 2], "big")
        if marker in range(0xE0, 0xF0) or marker == 0xFE:
            i += length
            continue
        out.extend(bytes((0xFF, marker)))
        out.extend(data[i : i + length])
        i += length
    return bytes(out)


def strip_png_metadata(data):
    """Remove PNG textual and time chunks."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return data
    drop = {b"tEXt", b"zTXt", b"iTXt", b"tIME"}
    out = bytearray(data[:8])
    i = 8
    while i + 12 <= len(data):
        length = int.from_bytes(data[i : i + 4], "big")
        ctype = data[i + 4 : i + 8]
        end = i + 12 + length
        if end > len(data):
            break
        chunk = data[i:end]
        if ctype not in drop:
            out.extend(chunk)
        i = end
        if ctype == b"IEND":
            break
    return bytes(out)


def write_grayscale_png(path, pixels):
    """Write an 8-bit grayscale PNG without Pillow."""
    height = len(pixels)
    width = len(pixels[0]) if height else 0

    def chunk(tag, payload):
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    raw = b"".join(b"\x00" + bytes(row) for row in pixels)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as handle:
        handle.write(png)


def anonymize_dicom(raw, dest_png):
    """Clear DICOM PHI tags and write a grayscale PNG for the vision LLM."""
    try:
        import pydicom
        from io import BytesIO
    except ImportError as exc:
        raise RuntimeError("pydicom is required for DICOM uploads: pip install pydicom") from exc

    ds = pydicom.dcmread(BytesIO(raw), force=True)
    removed = []
    for tag_name in (
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientSex",
        "PatientAddress",
        "OtherPatientIDs",
        "OtherPatientNames",
        "InstitutionName",
        "ReferringPhysicianName",
        "OperatorsName",
        "StudyDate",
        "SeriesDate",
        "AcquisitionDate",
        "ContentDate",
        "PatientTelephoneNumbers",
    ):
        if hasattr(ds, tag_name) and getattr(ds, tag_name):
            setattr(ds, tag_name, "")
            removed.append(tag_name)
    pixels = ds.pixel_array
    if pixels.ndim == 3:
        pixels = pixels[0]
    flat_min = float(pixels.min())
    flat_max = float(pixels.max()) or 1.0
    scaled = ((pixels.astype("float64") - flat_min) / (flat_max - flat_min) * 255.0).clip(0, 255)
    rows = [row.astype("uint8").tolist() for row in scaled]
    write_grayscale_png(dest_png, rows)
    return {
        "format": "dicom",
        "phi_tags_cleared": removed,
        "stored_as": os.path.basename(dest_png),
        "stored_original": False,
    }


def deidentify_image_bytes(raw, filename, dest_path):
    """Strip identifiers from an uploaded image; original bytes are not stored."""
    name = (filename or "").lower()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if name.endswith(".dcm") or name.endswith(".dicom") or raw[:4] == b"DICM" or raw[128:132] == b"DICM":
        png_path = os.path.splitext(dest_path)[0] + ".png"
        audit = anonymize_dicom(raw, png_path)
        audit["output_path"] = png_path
        return audit
    if raw[:2] == b"\xff\xd8":
        safe = strip_jpeg_metadata(raw)
        with open(dest_path, "wb") as handle:
            handle.write(safe)
        return {
            "format": "jpeg",
            "exif_stripped": safe != raw,
            "stored_original": False,
            "output_path": dest_path,
        }
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        safe = strip_png_metadata(raw)
        with open(dest_path, "wb") as handle:
            handle.write(safe)
        return {
            "format": "png",
            "text_chunks_stripped": safe != raw,
            "stored_original": False,
            "output_path": dest_path,
        }
    raise RuntimeError("Unsupported image type. Upload JPEG, PNG, or DICOM.")
