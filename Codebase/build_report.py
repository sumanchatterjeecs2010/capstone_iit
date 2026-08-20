"""Build Report.docx and Report.pdf using the Report1 sample heading format."""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from paths import REPORT_DIR

TITLE = "Multimodal Medical Assistant for Image-Text Clinical Triage"

ABSTRACT = (
    "This project implements a clinician-facing multimodal assistant that combines a radiology "
    "scan or histopathology slide with a structured clinical note to produce triage guidance, "
    "image–text correlation, evidence-linked summaries, and interactive follow-up chat. The "
    "clinical aim is to reduce missed cross-modal cues by forcing vision and text models to share "
    "one case record. Compact local models are used deliberately: MedGemma 4B for vision and "
    "Llama 3.2 (3B, with 1B fallback) for text reasoning. These sizes were chosen for memory "
    "efficiency so the stack runs on a typical 16 GB CPU laptop or on Google Colab (CPU or free "
    "T4 GPU), reducing cost, energy use, and data movement compared with large cloud-only "
    "multimodal APIs. Ollama auto-adapts to CPU or GPU; the app unloads the vision model after "
    "analysis so both LLMs share limited RAM/VRAM. Modalities are restricted to radiology and "
    "pathology with auto domain detection. LangGraph sequences analysis and follow-up re-triage. "
    "Uploads are de-identified before storage. Evaluation metrics appear in the TUI "
    "and GUI after every upload. Dual interfaces share this backend."
)

OBJECTIVES = [
    "Provide a TUI and a GUI so clinicians can upload a note and image, ask questions, and receive triage summaries.",
    "Return image-referenced findings, note context, and numbered guideline citations.",
    "Support adaptive follow-up with Llama re-triage of impression, differential, recommendations, and next questions.",
    "Auto-detect radiology versus pathology and reject out-of-scope images.",
    "De-identify text and images before any model call or disk write.",
    "Emit evaluation metrics (correlation, robustness, explanation quality, satisfaction) on every upload.",
]


def _set_run_font(run, size=12, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)


def _add_page_number(paragraph):
    run = paragraph.add_run()
    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "end")
    run._r.append(fld1)
    run._r.append(instr)
    run._r.append(fld2)
    _set_run_font(run, size=10)


def _para(doc, text, *, size=12, bold=False, italic=False, center=False, space_after=6):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    _set_run_font(run, size=size, bold=bold, italic=italic)
    return p


def _heading(doc, text):
    return _para(doc, text, size=12, bold=True, space_after=6)


def _bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_after = Pt(2)
    if p.runs:
        p.runs[0].text = text
        _set_run_font(p.runs[0], size=12)
    else:
        run = p.add_run(text)
        _set_run_font(run, size=12)
    return p


def build_docx():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)
    footer = section.footer
    footer.is_linked_to_previous = False
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _add_page_number(fp)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    _para(doc, TITLE, size=16, bold=True, center=True, space_after=10)
    _para(doc, "HPPCS[04] Capstone Report", size=12, italic=True, center=True, space_after=12)

    _heading(doc, "Abstract")
    _para(doc, ABSTRACT)

    _heading(doc, "1. Introduction")
    _bullet(
        doc,
        "Context and background: Imaging and notes are usually read in separate tools. This system "
        "binds them in one LangGraph state so triage uses both visual findings and vitals or history.",
    )
    _bullet(
        doc,
        "Motivation: Local MedGemma + Llama on Ollama keep PHI on-premises and match the "
        "HPPCS[04] multimodal brief.",
    )

    _heading(doc, "2. Problem Statement")
    _para(
        doc,
        "Given one radiology or pathology image and one clinical note, the system must (i) detect "
        "domain, (ii) correlate image findings with the note, (iii) assign triage "
        "(emergency / urgent / soon / routine), (iv) answer clinician follow-up questions with "
        "updated recommendations, and (v) report evaluation and interface metrics. "
        "Out-of-scope photographs (dermatology, fundus, clinical snapshots) must be rejected.",
    )

    _heading(doc, "3. Objectives")
    for item in OBJECTIVES:
        _bullet(doc, item)

    _heading(doc, "4. Methodology")
    _bullet(
        doc,
        "Tools and technologies: Python 3.12, FastAPI/Uvicorn, LangGraph, Ollama HTTP API, "
        "MedGemma 4B (vision), Llama 3.2 3B (text; 1B fallback), runtime_profile.py, python-docx.",
    )
    _bullet(
        doc,
        "Workflow: ingest_pair de-identifies files → medgemma_analyze (domain + findings) → "
        "llama_entities → llama_reason (triage JSON) → llama_converse (cited summary) → optional "
        "chat (refresh_reasoning_after_follow_up). Metrics by evaluation.py.",
    )
    _para(
        doc,
        "Conceptual pipeline: Note+Image → Privacy gate → MedGemma → Llama entities → Llama triage "
        "→ Clinician summary → Follow-up re-triage → Evaluation panel in TUI/GUI.",
        italic=True,
    )

    _heading(doc, "5. System Design / Implementation")
    _para(
        doc,
        "Architecture (layers): Presentation (tui_app.py, upload_app.py) → Application "
        "(medical_assistant.py sessions) → Orchestration (graph_pipeline.py LangGraph) → Models "
        "(ollama_client.py load/unload) → Data (ingestion.py, privacy.py, entity_normalizer.py, "
        "clinical_references.py, evaluation.py). sample_data/ sits outside Codebase; live sessions "
        "are Codebase/uploads/processed/<id>/.",
    )
    _bullet(doc, "main.py — Mode 1 TUI / Mode 2 GUI; --device auto|cpu|gpu; Ollama model checks.")
    _bullet(doc, "runtime_profile.py — detects NVIDIA GPU vs CPU; tunes keep-alive, timeouts, unload policy.")
    _bullet(doc, "graph_pipeline.py — four-node graph; follow-up JSON re-triage and citations.")
    _bullet(doc, "evaluation.py — correlation, robustness, explanation, and satisfaction metrics.")
    _bullet(doc, "upload_app.py / tui_app.py — dashboard and terminal chat; metrics panel in both.")

    _heading(doc, "6. Observations and Results")
    _para(
        doc,
        "Evaluation is performed during a live run through Mode 1 (TUI) or Mode 2 (GUI): upload, "
        "analysis, follow-up chat, and the Evaluation panel. Faculty or clinicians may supply their "
        "own de-identified note and image at demo time.",
    )
    _para(
        doc,
        "Qualitative observations on representative radiology and pathology uploads include: "
        "MedGemma correctly distinguishes radiology versus pathology domains; Llama produces "
        "structured triage (emergency / urgent / soon / routine) with impression, differential, "
        "and recommendations; image–note correlation text links visual findings to note context; "
        "clinical_references.py attaches WHO/NIH-style guideline links; and follow-up questions "
        "trigger re-triage with updated recommendations and cited replies.",
    )
    _para(
        doc,
        "Quantitative results are computed by evaluation.py on each upload and shown in the "
        "Evaluation panel:",
    )
    _bullet(doc, "Cross-modal correlation — presence of an image–note narrative and token overlap between findings and note.")
    _bullet(doc, "Robustness — valid domain detection, de-identified storage, and completeness of required output fields.")
    _bullet(doc, "Explanation quality — checklist over impression, rationale, findings, correlation, recommendations, references, and follow-up prompts (mapped to a 1–5 Likert).")
    _bullet(doc, "User satisfaction — interface completeness proxy from triage, findings, chat, prompt chips, and guideline links (1–5 Likert).")
    _para(
        doc,
        "For demonstration, run python main.py, choose TUI or GUI, upload one radiology or "
        "pathology case, and read the four scores in the Evaluation section alongside triage "
        "output and follow-up chat.",
    )

    _heading(doc, "7. Conclusion and Future Work")
    _bullet(
        doc,
        "Summary of contributions: dual-LLM multimodal triage with auto domain detection, privacy "
        "gate, dual clinician interfaces, adaptive cited follow-up, and evaluation metric outputs.",
    )
    _bullet(
        doc,
        "Possible extensions: larger GPU serving when hardware allows, reader-study Likert "
        "collection from clinicians, and SNOMED coding via entity_normalizer.py.",
    )

    _heading(doc, "9. References")
    _bullet(doc, "HPPCS[04] capstone brief.")
    _bullet(doc, "Google Health AI Developer Foundations, MedGemma model card.")
    _bullet(doc, "Meta, Llama 3.2 model card; Ollama library (medgemma, llama3.2).")
    _bullet(doc, "LangGraph documentation, https://langchain-ai.github.io/langgraph/")
    _bullet(doc, "WHO/NIH public guideline pages linked from clinical_references.py.")
    _bullet(doc, "Wikimedia Commons teaching images; optional demo files in sample_data/.")

    _heading(doc, "Acknowledgement")
    _para(
        doc,
        "Thanks to HAAI++ faculty for the HPPCS[04] brief and to the MedGemma, Llama, LangGraph, "
        "and Ollama communities. Teaching images are public; accompanying notes are fictional.",
    )

    out = Path(REPORT_DIR) / "Report.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        doc.save(out)
    except PermissionError:
        out = Path(REPORT_DIR) / "Report_new.docx"
        doc.save(out)
        print("Report.docx is open — wrote", out, "instead. Close Word and rerun build_report.py.")
    return out


def docx_to_pdf(docx_path):
    docx_path = Path(docx_path)
    pdf_path = docx_path.with_suffix(".pdf")
    try:
        from docx2pdf import convert

        convert(str(docx_path), str(pdf_path))
        return pdf_path
    except Exception as exc:
        print("docx2pdf failed ({}); trying Word COM…".format(exc))
    import win32com.client

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    doc = word.Documents.Open(str(docx_path.resolve()))
    doc.SaveAs(str(pdf_path.resolve()), FileFormat=17)
    doc.Close()
    word.Quit()
    return pdf_path


if __name__ == "__main__":
    docx = build_docx()
    print("Wrote", docx)
    print("Wrote", docx_to_pdf(docx))
