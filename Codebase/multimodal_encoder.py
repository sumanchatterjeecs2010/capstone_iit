"""
multimodal_encoder.py
---------------------
Medical image-text encoders from the project brief.

- BiomedCLIP (preferred) or PubMedCLIP / MedCLIP-family fallback:
  scores the image against clinical finding prompts.
- BioBERT (sentence embedding): scores the same prompts against the
  patient note.
- Fusion: combine image evidence and note evidence so the two LLMs
  reason over ranked findings rather than raw pixels.

These encoders are not LLMs. They improve quality of the evidence that
Llama then turns into conversation and triage.
"""

import gc
import os

import numpy as np
from PIL import Image


# Clinical candidate statements scored by both the vision encoder and BioBERT.
# Wording is kept as natural sentences because CLIP-style models rank those better
# than single-word labels.
CANDIDATE_FINDINGS = [
    {"id": "mod_chest", "modality": "chest_xray", "label": "chest radiograph",
     "prompt": "a frontal chest radiograph of the lungs"},
    {"id": "mod_skin", "modality": "dermatology_photo", "label": "dermatology photograph",
     "prompt": "a close-up clinical photograph of a skin lesion"},
    {"id": "mod_brain", "modality": "brain_ct", "label": "brain CT",
     "prompt": "an axial CT scan of the brain"},
    {"id": "mod_fundus", "modality": "fundus", "label": "retinal fundus photograph",
     "prompt": "a colour retinal fundus photograph of the eye"},
    {"id": "mod_bone", "modality": "bone_xray", "label": "extremity radiograph",
     "prompt": "a radiograph of the wrist and distal radius"},
    {"id": "pna", "modality": "chest_xray", "label": "pneumonia / consolidation",
     "prompt": "a chest X-ray showing pneumonia, infiltrate or pulmonary consolidation"},
    {"id": "edema", "modality": "chest_xray", "label": "pulmonary edema",
     "prompt": "a chest X-ray showing pulmonary edema or congestive heart failure"},
    {"id": "normal_cxr", "modality": "chest_xray", "label": "normal chest radiograph",
     "prompt": "a normal chest radiograph without consolidation"},
    {"id": "melanoma", "modality": "dermatology_photo", "label": "suspicious pigmented lesion",
     "prompt": "a photograph of an irregular pigmented skin lesion concerning for melanoma"},
    {"id": "nevus", "modality": "dermatology_photo", "label": "benign nevus",
     "prompt": "a photograph of a regular benign melanocytic nevus"},
    {"id": "ich", "modality": "brain_ct", "label": "intracranial hemorrhage",
     "prompt": "a brain CT showing intracranial hemorrhage or hyperdense blood"},
    {"id": "infarct", "modality": "brain_ct", "label": "acute ischaemic stroke",
     "prompt": "a brain CT showing acute ischaemic stroke or infarct"},
    {"id": "normal_ct", "modality": "brain_ct", "label": "normal brain CT",
     "prompt": "a normal non-contrast CT of the brain"},
    {"id": "dr", "modality": "fundus", "label": "diabetic retinopathy",
     "prompt": "a fundus photograph showing diabetic retinopathy with retinal haemorrhages"},
    {"id": "normal_fundus", "modality": "fundus", "label": "normal fundus",
     "prompt": "a normal retinal fundus photograph without haemorrhage"},
    {"id": "fracture", "modality": "bone_xray", "label": "wrist / distal radius fracture",
     "prompt": "a wrist radiograph showing a distal radius or scaphoid fracture"},
    {"id": "normal_bone", "modality": "bone_xray", "label": "normal wrist radiograph",
     "prompt": "a normal wrist radiograph without fracture"},
]


_clip_bundle = None
_biobert = None


def _softmax(values):
    """
    Numerically stable softmax over a 1-D list of scores.

    Parameters
    ----------
    values : list[float] or numpy.ndarray
        Raw similarity logits.

    Returns
    -------
    numpy.ndarray
        Probabilities that sum to 1.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    return exp / max(float(exp.sum()), 1e-9)


def load_biomedclip():
    """
    Load BiomedCLIP, or a MedCLIP/PubMedCLIP fallback, onto CPU.

    Returns
    -------
    dict
        Keys: backend, model, preprocess/tokenizer or processor.
    """
    global _clip_bundle
    if _clip_bundle is not None:
        return _clip_bundle

    # A stale Hugging Face token yields 401 even on public medical checkpoints.
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        os.environ.pop(key, None)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

    try:
        import open_clip
        import torch
        from huggingface_hub import hf_hub_download

        print("Loading BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT-ViT-B-32)...", flush=True)
        # Force an unauthenticated public download first so a bad token cannot 401.
        hf_hub_download(
            repo_id="microsoft/BiomedCLIP-PubMedBERT-ViT-B-32",
            filename="open_clip_config.json",
            token=False,
        )
        model, preprocess = open_clip.create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT-ViT-B-32"
        )
        tokenizer = open_clip.get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT-ViT-B-32")
        model.eval()
        _clip_bundle = {
            "backend": "biomedclip",
            "torch": torch,
            "model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
        }
        print("BiomedCLIP ready.", flush=True)
        return _clip_bundle
    except Exception as exc:
        print("BiomedCLIP unavailable ({}). Trying PubMedCLIP/MedCLIP fallback...".format(exc), flush=True)

    from transformers import CLIPModel, CLIPProcessor
    import torch

    name = "flaviagiammarino/pubmed-clip-vit-base-patch32"
    print("Loading MedCLIP-family model ({})...".format(name), flush=True)
    model = CLIPModel.from_pretrained(name, token=False)
    processor = CLIPProcessor.from_pretrained(name, token=False)
    model.eval()
    _clip_bundle = {
        "backend": "pubmedclip",
        "torch": torch,
        "model": model,
        "processor": processor,
    }
    print("PubMedCLIP ready.", flush=True)
    return _clip_bundle


def load_biobert():
    """
    Load a BioBERT sentence encoder for clinical-note embeddings.

    Returns
    -------
    sentence_transformers.SentenceTransformer
        Mean-pooling BioBERT encoder.
    """
    global _biobert
    if _biobert is not None:
        return _biobert

    from sentence_transformers import SentenceTransformer

    candidates = [
        "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        "dmis-lab/biobert-v1.1",
        "emilyalsentzer/Bio_ClinicalBERT",
    ]
    last_error = None
    for name in candidates:
        try:
            print("Loading BioBERT encoder ({})...".format(name), flush=True)
            _biobert = SentenceTransformer(name)
            print("BioBERT ready:", name, flush=True)
            return _biobert
        except Exception as exc:
            last_error = exc
            print("Could not load {}: {}".format(name, exc), flush=True)
    raise RuntimeError("Unable to load BioBERT/ClinicalBERT: {}".format(last_error))


def preload_encoders():
    """
    Download and cache BiomedCLIP and BioBERT once at process start.

    Returns
    -------
    dict
        Names of the loaded encoder backends.
    """
    clip = load_biomedclip()
    bert = load_biobert()
    return {
        "vision_encoder": clip["backend"],
        "text_encoder": bert.__class__.__name__,
    }


def score_image_with_clip(image_path):
    """
    Rank candidate clinical prompts against the image with BiomedCLIP/MedCLIP.

    Parameters
    ----------
    image_path : str
        Path to the medical image.

    Returns
    -------
    list[dict]
        Candidates annotated with an image similarity score (higher is better).
    """
    bundle = load_biomedclip()
    torch = bundle["torch"]
    image = Image.open(image_path).convert("RGB")
    prompts = [item["prompt"] for item in CANDIDATE_FINDINGS]

    with torch.no_grad():
        if bundle["backend"] == "biomedclip":
            image_tensor = bundle["preprocess"](image).unsqueeze(0)
            text_tokens = bundle["tokenizer"](prompts)
            image_features = bundle["model"].encode_image(image_tensor)
            text_features = bundle["model"].encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ text_features.T).squeeze(0).cpu().numpy()
        else:
            inputs = bundle["processor"](
                text=prompts,
                images=image,
                return_tensors="pt",
                padding=True,
            )
            outputs = bundle["model"](**inputs)
            logits = outputs.logits_per_image.squeeze(0).cpu().numpy()

    probs = _softmax(logits)
    ranked = []
    for item, logit, prob in zip(CANDIDATE_FINDINGS, logits, probs):
        row = dict(item)
        row["image_logit"] = float(logit)
        row["image_score"] = float(prob)
        ranked.append(row)
    ranked.sort(key=lambda row: row["image_score"], reverse=True)
    return ranked


def score_note_with_biobert(note):
    """
    Rank candidate findings by BioBERT similarity to the patient note.

    Parameters
    ----------
    note : str
        Prescription / patient-detail text.

    Returns
    -------
    list[dict]
        Candidates annotated with a note similarity score in [0, 1] after scaling.
    """
    encoder = load_biobert()
    prompts = [item["prompt"] for item in CANDIDATE_FINDINGS]
    note_vec = encoder.encode([note], normalize_embeddings=True)[0]
    prompt_vecs = encoder.encode(prompts, normalize_embeddings=True)
    sims = np.matmul(prompt_vecs, note_vec)
    # Cosine similarities are shifted from [-1, 1] to [0, 1] for fusion.
    scaled = (sims + 1.0) / 2.0
    ranked = []
    for item, sim, score in zip(CANDIDATE_FINDINGS, sims, scaled):
        row = dict(item)
        row["note_cosine"] = float(sim)
        row["note_score"] = float(score)
        ranked.append(row)
    ranked.sort(key=lambda row: row["note_score"], reverse=True)
    return ranked


def infer_modality_from_scores(fused_rows):
    """
    Pick the imaging modality from CLIP image scores only.

    Note similarity must not elect the modality: a pneumonia note would
    otherwise force a fundus photograph into the chest class, and vice versa.

    Parameters
    ----------
    fused_rows : list[dict]
        Fused candidate rows that still contain image_score.

    Returns
    -------
    str
        One of the dataset modality labels.
    """
    modality_ids = {"mod_chest", "mod_skin", "mod_brain", "mod_fundus", "mod_bone"}
    modality_rows = [row for row in fused_rows if row["id"] in modality_ids]
    if not modality_rows:
        return "unknown"
    return max(modality_rows, key=lambda row: row["image_score"])["modality"]


def fuse_image_and_note(image_path, note):
    """
    Build a cross-modal finding list from BiomedCLIP and BioBERT.

    Image score and note score are combined so that a finding must be
    supported by the picture, the prescription, or both. Agreement is
    reported explicitly for the reasoning LLM.

    Parameters
    ----------
    image_path : str
        Medical image path.
    note : str
        Patient note.

    Returns
    -------
    dict
        Fusion record consumed by the assistant.
    """
    image_ranked = score_image_with_clip(image_path)
    note_ranked = score_note_with_biobert(note)
    note_by_id = {row["id"]: row for row in note_ranked}

    fused = []
    for img_row in image_ranked:
        note_row = note_by_id[img_row["id"]]
        # Vision gets slightly more weight because the note is already given
        # verbatim to the LLMs; CLIP is the new image evidence.
        fused_score = 0.40 * img_row["image_score"] + 0.60 * note_row["note_score"]
        fused.append(
            {
                "id": img_row["id"],
                "label": img_row["label"],
                "modality": img_row["modality"],
                "prompt": img_row["prompt"],
                "image_score": round(img_row["image_score"], 4),
                "note_score": round(note_row["note_score"], 4),
                "fused_score": round(float(fused_score), 4),
                "agreement": bool(
                    img_row["image_score"] >= 0.08 and note_row["note_score"] >= 0.55
                ),
            }
        )
    fused.sort(key=lambda row: row["fused_score"], reverse=True)
    top = [row for row in fused if not row["id"].startswith("mod_")][:6]
    modality = infer_modality_from_scores(fused)
    findings = [
        "{} (image={:.2f}, note={:.2f}, fused={:.2f})".format(
            row["label"], row["image_score"], row["note_score"], row["fused_score"]
        )
        for row in top[:4]
    ]
    return {
        "vision_backend": load_biomedclip()["backend"],
        "text_backend": "biobert",
        "inferred_modality": modality,
        "fused_findings": top,
        "modality_scores": [row for row in fused if row["id"].startswith("mod_")][:5],
        "visual_findings": findings,
        "fusion_summary": (
            "BiomedCLIP/MedCLIP scored the image; BioBERT scored the note; "
            "top fused findings: {}.".format("; ".join(findings) if findings else "none")
        ),
    }


def release_encoders():
    """
    Drop encoder weights so an LLM can reuse RAM if needed.
    """
    global _clip_bundle, _biobert
    _clip_bundle = None
    _biobert = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
