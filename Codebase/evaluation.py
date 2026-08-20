"""Clinical evaluation metrics written into conversation.json and evaluation_summary.json.

All metrics are label-free so they work on any upload (including evaluator-provided samples).
"""

import re


def cross_modal_correlation(record):
    visual = record.get("visual_analysis") or {}
    reasoning = record.get("clinical_reasoning") or {}
    note = str(record.get("note_text") or "").lower()
    findings = " ".join(str(item).lower() for item in (visual.get("visual_findings") or []))
    correlation = str(
        visual.get("image_note_correlation") or reasoning.get("image_text_correlation") or ""
    ).strip()
    tokens = [tok for tok in re.findall(r"[a-z]{4,}", findings) if tok in note]
    overlap = round(len(set(tokens)) / 8.0, 3) if tokens else 0.0
    overlap = min(overlap, 1.0)
    present = 1.0 if len(correlation) >= 40 else 0.0
    score = round(0.6 * present + 0.4 * overlap, 3)
    return {
        "score": score,
        "image_note_correlation_present": bool(present),
        "shared_tokens": sorted(set(tokens))[:12],
        "correlation_excerpt": correlation[:240],
    }


def robustness_to_variation(record):
    visual = record.get("visual_analysis") or {}
    privacy = record.get("privacy") or {}
    detected = str(
        (record.get("input") or {}).get("image_domain") or visual.get("image_domain") or ""
    ).lower()
    domain_ok = 1.0 if detected in {"radiology", "pathology"} else 0.0
    originals_dropped = 0.0 if privacy.get("originals_stored") else 1.0
    fields = [
        visual.get("visual_findings"),
        visual.get("image_description"),
        (record.get("clinical_reasoning") or {}).get("impression"),
        (record.get("clinical_reasoning") or {}).get("triage"),
    ]
    complete = sum(1 for item in fields if item) / float(len(fields))
    score = round((domain_ok + originals_dropped + complete) / 3.0, 3)
    return {
        "score": score,
        "domain_detected": detected,
        "domain_match": domain_ok,
        "originals_not_stored": originals_dropped,
        "required_fields_complete": round(complete, 3),
    }


def explanation_quality(record):
    reasoning = record.get("clinical_reasoning") or {}
    visual = record.get("visual_analysis") or {}
    checks = {
        "impression": bool(str(reasoning.get("impression") or "").strip()),
        "triage_rationale": bool(str(reasoning.get("triage_rationale") or "").strip()),
        "visual_findings": bool(visual.get("visual_findings")),
        "image_note_correlation": bool(
            str(visual.get("image_note_correlation") or reasoning.get("image_text_correlation") or "").strip()
        ),
        "recommendations": bool(reasoning.get("recommendations")),
        "references": bool(record.get("references")),
        "follow_up_questions": bool(reasoning.get("follow_up_questions")),
    }
    score = round(sum(1 for ok in checks.values() if ok) / float(len(checks)), 3)
    return {"score": score, "checklist": checks, "likert_1_to_5": round(1 + 4 * score, 2)}


def user_satisfaction(record):
    conversation = record.get("conversation") or []
    reasoning = record.get("clinical_reasoning") or {}
    interface_checks = {
        "session_record": True,
        "image_findings_panel": bool((record.get("visual_analysis") or {}).get("visual_findings")),
        "triage_panel": bool(reasoning.get("triage")),
        "chat_transcript": bool(conversation),
        "follow_up_prompts": bool(reasoning.get("follow_up_questions")),
        "guideline_links": bool(record.get("references")),
    }
    interface = round(sum(1 for ok in interface_checks.values() if ok) / float(len(interface_checks)), 3)
    explanation = explanation_quality(record)
    return {
        "interface_score": interface,
        "interface_likert_1_to_5": round(1 + 4 * interface, 2),
        "explanation_score": explanation["score"],
        "explanation_likert_1_to_5": explanation["likert_1_to_5"],
        "interface_checklist": interface_checks,
        "scale": "1–5 Likert proxy from completeness of interface and explanations on this case",
    }


def evaluate_record(record):
    """Return label-free metrics for any upload."""
    return {
        "cross_modal_correlation": cross_modal_correlation(record),
        "robustness_to_data_variation": robustness_to_variation(record),
        "explanation_quality": explanation_quality(record),
        "user_satisfaction": user_satisfaction(record),
    }


def aggregate_metrics(records):
    evals = [item.get("evaluation") or evaluate_record(item) for item in records]

    def mean(path):
        values = []
        for item in evals:
            cursor = item
            for key in path:
                cursor = (cursor or {}).get(key)
            if isinstance(cursor, (int, float)):
                values.append(float(cursor))
        return round(sum(values) / float(len(values) or 1), 3)

    return {
        "n_cases": len(evals),
        "cross_modal_correlation_score": mean(["cross_modal_correlation", "score"]),
        "robustness_score": mean(["robustness_to_data_variation", "score"]),
        "explanation_quality_score": mean(["explanation_quality", "score"]),
        "user_satisfaction": {
            "interface_likert_1_to_5": mean(["user_satisfaction", "interface_likert_1_to_5"]),
            "explanation_likert_1_to_5": mean(["user_satisfaction", "explanation_likert_1_to_5"]),
        },
        "per_case": evals,
        "note": (
            "Label-free metrics: cross-modal correlation, robustness, explanation quality, "
            "and user-satisfaction Likert proxies."
        ),
    }
