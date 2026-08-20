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
        result = ingest_pair(note, "note.txt", jpeg, "skin.jpg")
        stored = open(result["text"]["path"], encoding="utf-8").read()
        assert "John Smith" not in stored
        assert "john@example.com" not in stored
        assert result["text"]["privacy"]["stored_original"] is False
        assert result["image"]["privacy"]["stored_original"] is False
        assert result["image"]["domain"] is None


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
    from clinical_references import _host_allowed, attach_references

    refs = attach_references(
        {
            "clinical_reasoning": {"differential": ["Pneumonia"]},
            "visual_analysis": {"visual_findings": ["consolidation"]},
        }
    )
    assert refs
    assert refs[0].get("cite_key") == "[1]"
    assert any("pneumonia" in ref["url"].lower() or "pneumonia" in ref["title"].lower() for ref in refs)
    assert _host_allowed("https://www.who.int/news-room/fact-sheets/detail/pneumonia")
    assert not _host_allowed("https://example.com/fake-guideline")


def test_rejects_non_radiology_pathology_domain():
    from graph_pipeline import UnsupportedImageDomainError, resolve_detected_domain

    try:
        resolve_detected_domain({"image_domain": "unsupported", "inferred_modality": "unknown"})
        assert False, "unsupported domain should be rejected"
    except UnsupportedImageDomainError as exc:
        assert "radiology" in str(exc).lower()

    assert resolve_detected_domain({"image_domain": "pathology", "inferred_modality": "pathology_slide"}) == "pathology"
    assert resolve_detected_domain({"inferred_modality": "chest_xray"}) == "radiology"


def test_evaluation_metrics_include_correlation_and_satisfaction():
    from evaluation import evaluate_record, aggregate_metrics

    record = {
        "note_text": "pneumonia cough fever chest radiograph spo2 91",
        "visual_analysis": {
            "image_domain": "radiology",
            "visual_findings": ["right lower lobe opacity", "chest consolidation"],
            "image_description": "PA chest radiograph",
            "image_note_correlation": "The radiograph shows consolidation consistent with the note describing pneumonia and hypoxia.",
        },
        "clinical_reasoning": {
            "impression": "Community-acquired pneumonia",
            "triage": "urgent",
            "triage_rationale": "Hypoxia with radiographic consolidation.",
            "recommendations": ["Oxygen", "Antibiotics"],
            "follow_up_questions": ["Trend SpO2?"],
            "differential": ["Pneumonia"],
        },
        "references": [{"title": "WHO pneumonia", "url": "https://www.who.int/"}],
        "conversation": [{"role": "assistant", "content": "Summary"}],
        "privacy": {"originals_stored": False},
        "input": {"image_domain": "radiology"},
    }
    ev = evaluate_record(record)
    assert ev["cross_modal_correlation"]["score"] > 0
    assert ev["robustness_to_data_variation"]["score"] > 0
    assert ev["explanation_quality"]["score"] > 0
    assert ev["user_satisfaction"]["interface_likert_1_to_5"] >= 1
    summary = aggregate_metrics([{"evaluation": ev}])
    assert "cross_modal_correlation_score" in summary
    assert "interface_likert_1_to_5" in summary["user_satisfaction"]


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
    test_rejects_non_radiology_pathology_domain()
    test_evaluation_metrics_include_correlation_and_satisfaction()
    print("ingestion/privacy/normalization tests passed")
