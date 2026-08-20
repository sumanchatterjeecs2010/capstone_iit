"""Lightweight tests for de-identification and entity normalization (no LLM)."""

import os
import tempfile

from entity_normalizer import normalize_clinical_entities
from ingestion import ingest_pair
from privacy import deidentify_text, strip_jpeg_metadata


def test_text_phi_is_redacted():
    raw = (
        "Patient Name: Jane Doe\n"
        "MRN: 44556677\n"
        "Phone: 555-123-9876\n"
        "Email: jane.doe@hospital.org\n"
        "DOB: 12/01/1964\n"
        "Chief complaint: cough and fever.\n"
    )
    cleaned, audit = deidentify_text(raw)
    assert "Jane Doe" not in cleaned
    assert "jane.doe@hospital.org" not in cleaned
    assert "555-123-9876" not in cleaned
    assert "44556677" not in cleaned
    assert audit["n_redactions"] >= 3
    assert "[EMAIL]" in cleaned
    assert "[PHONE]" in cleaned


def test_upload_does_not_keep_original_phi():
    note = b"Name: John Smith\nEmail: john@example.com\nCough for 3 days.\n"
    jpeg = b"\xff\xd8\xff\xe1\x00\x10EXIFdummy\xff\xd9"
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(os.path.dirname(__file__))
        result = ingest_pair(note, "note.txt", jpeg, "skin.jpg", image_domain="pathology")
        stored = open(result["text"]["path"], encoding="utf-8").read()
        assert "John Smith" not in stored
        assert "john@example.com" not in stored
        assert result["text"]["privacy"]["stored_original"] is False
        assert result["image"]["privacy"]["stored_original"] is False
        assert result["image"]["domain"] == "pathology"


def test_entity_normalization():
    extracted = {
        "age_sex": "62-year-old male",
        "chief_complaint": "community-acquired pneumonia versus heart failure",
        "vitals": "HR 108, RR 24, SpO2 91%, BP 138/84",
        "medications": "Metformin 500 mg",
        "requested_study": "Chest radiograph (PA)",
    }
    norm = normalize_clinical_entities(extracted, extracted["chief_complaint"])
    assert norm["demographics"]["age_years"] == 62
    assert norm["demographics"]["sex"] == "male"
    assert norm["vitals"]["spo2_percent"] == 91
    codes = {row["code"] for row in norm["conditions"]}
    assert "COND-PNEUMONIA" in codes
    assert norm["medications"][0]["code"] == "MED-METFORMIN"
    assert norm["requested_study"]["code"] == "IMG-CXR"


def test_jpeg_exif_stripped():
    raw = b"\xff\xd8\xff\xe1\x00\x08ABCD\xff\xda\x00\x02\xff\xd9"
    stripped = strip_jpeg_metadata(raw)
    assert b"ABCD" not in stripped
    assert stripped.startswith(b"\xff\xd8")


def test_clinical_references():
    from clinical_references import attach_references

    refs = attach_references(
        {
            "clinical_reasoning": {"differential": ["Pneumonia"]},
            "visual_analysis": {"visual_findings": ["consolidation"]},
        }
    )
    assert refs
    assert any("pneumonia" in ref["url"].lower() or "pneumonia" in ref["title"].lower() for ref in refs)


def test_session_id_validation():
    from medical_assistant import _valid_session_id

    assert _valid_session_id("9aae6980aaa1")
    assert not _valid_session_id("../escape")
    assert not _valid_session_id("")


if __name__ == "__main__":
    test_text_phi_is_redacted()
    test_upload_does_not_keep_original_phi()
    test_entity_normalization()
    test_jpeg_exif_stripped()
    test_clinical_references()
    test_session_id_validation()
    print("ingestion/privacy/normalization tests passed")
