"""
upload_app.py
-------------
Clinician dashboard: upload note + image, view findings, chat follow-ups.

Served by `python main.py` (FastAPI + inline HTML/JS, Python stack only).
"""

import asyncio
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from ingestion import ingest_pair
from medical_assistant import (
    dashboard_payload,
    get_graph,
    process_case,
    process_follow_up,
    save_json,
    session_dir_from_id,
)
from ollama_client import OllamaClient, OllamaError
from runtime_profile import get_profile


app = FastAPI(title="Multimodal Medical Assistant", version="2.0")

IMAGE_MEDIA = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


class ChatRequest(BaseModel):
    session_id: str
    message: str


def _ollama_host():
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    if not host.startswith("http"):
        host = "http://127.0.0.1:11434"
    return host


def _vision_model():
    return os.environ.get("VISION_MODEL", "medgemma:4b")


def _llama_model():
    return os.environ.get("LLAMA_MODEL", "llama3.2:3b")


def _maybe_unload_vision(client):
    if not get_profile().unload_vision_after_analysis:
        return
    try:
        client.unload_model(_vision_model())
    except OllamaError:
        pass


def _client():
    client = OllamaClient(host=_ollama_host())
    if not client.is_available():
        raise HTTPException(503, "Ollama is not running. Start Ollama, then refresh.")
    return client


def _read_upload(file: UploadFile):
    if file is None:
        raise HTTPException(400, "Missing file")
    return file.file.read(), file.filename or "upload"


def _session_image_path(session_id):
    session_dir = session_dir_from_id(session_id)
    for name in os.listdir(session_dir):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            return os.path.join(session_dir, name)
    raise HTTPException(404, "Image not found for session")


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Multimodal Medical Assistant</title>
  <style>
    :root {
      --bg: #f6f4ef;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #5f6c7b;
      --accent: #1f6f5f;
      --accent-soft: #e6f3ef;
      --border: #d9e2ec;
      --emergency: #b42318;
      --urgent: #c05600;
      --soon: #946200;
      --routine: #2f6b3a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      background: var(--accent);
      color: #fff;
      padding: 1rem 1.5rem;
    }
    header h1 { margin: 0; font-size: 1.25rem; }
    header p { margin: 0.35rem 0 0; opacity: 0.9; font-size: 0.92rem; }
    header .tagline { font-size: 0.82rem; opacity: 0.85; margin-top: 0.25rem; }
    footer {
      text-align: center;
      color: var(--muted);
      font-size: 0.82rem;
      padding: 1rem;
      border-top: 1px solid var(--border);
      margin-top: 1.5rem;
    }
    main { max-width: 1200px; margin: 0 auto; padding: 1rem; }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem;
      margin-bottom: 1rem;
    }
    .grid {
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 1rem;
    }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
    }
    label { display: block; margin-top: 0.75rem; font-weight: 600; font-size: 0.9rem; }
    input[type=file], select, input[type=text], textarea, button {
      width: 100%;
      margin-top: 0.35rem;
      font: inherit;
    }
    button {
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 0.65rem 1rem;
      cursor: pointer;
      width: auto;
    }
    button:disabled { opacity: 0.6; cursor: wait; }
    .note {
      background: var(--accent-soft);
      border-radius: 8px;
      padding: 0.75rem;
      font-size: 0.9rem;
      color: var(--muted);
    }
    #dashboard { display: none; }
    #preview {
      width: 100%;
      max-height: 260px;
      object-fit: contain;
      background: #111;
      border-radius: 8px;
    }
    .badge {
      display: inline-block;
      padding: 0.25rem 0.65rem;
      border-radius: 999px;
      color: #fff;
      font-size: 0.82rem;
      font-weight: 700;
      text-transform: uppercase;
    }
    .badge.emergency { background: var(--emergency); }
    .badge.urgent { background: var(--urgent); }
    .badge.soon { background: var(--soon); }
    .badge.routine { background: var(--routine); }
    ul.compact { margin: 0.4rem 0 0 1rem; padding: 0; }
    ul.compact li { margin: 0.25rem 0; font-size: 0.92rem; }
    #chat {
      height: 420px;
      overflow-y: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem;
      background: #fafafa;
    }
    .msg { margin-bottom: 0.75rem; max-width: 92%; }
    .msg.clinician {
      margin-left: auto;
      background: var(--accent-soft);
      border-radius: 12px 12px 2px 12px;
      padding: 0.65rem 0.8rem;
    }
    .msg.assistant {
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 12px 12px 12px 2px;
      padding: 0.65rem 0.8rem;
    }
    .msg .role {
      font-size: 0.72rem;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 0.25rem;
    }
    .chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
    .chip {
      background: #eef2f6;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 0.25rem 0.65rem;
      font-size: 0.82rem;
      cursor: pointer;
    }
    .refs a { color: var(--accent); font-size: 0.88rem; }
    #status { color: var(--muted); font-size: 0.9rem; min-height: 1.2rem; }
    .chat-row { display: flex; gap: 0.5rem; margin-top: 0.75rem; }
    .chat-row input { flex: 1; padding: 0.55rem 0.7rem; border: 1px solid var(--border); border-radius: 8px; }
  </style>
</head>
<body>
  <header>
    <h1>Multimodal Medical Assistant</h1>
    <p>Clinical decision support for radiology and pathology — image–text fusion, triage, and interactive follow-up.</p>
    <p class="tagline">MedGemma 4B · Llama 3.2 · LangGraph · Auto domain detection · Evidence-linked responses</p>
  </header>
  <main>
    <section id="upload-panel" class="card">
      <h2 style="margin-top:0;font-size:1.05rem;">1. Upload case</h2>
      <p class="note">Upload a de-identified clinical note and a radiology scan (X-ray, CT, MRI) or histopathology slide. Image domain is auto-detected. All files are privacy-scrubbed before analysis.</p>
      <form id="upload-form">
        <label>Clinical note (.txt / .md)</label>
        <input type="file" name="note" accept=".txt,.md,text/plain" required/>
        <label>Radiology or pathology image (.jpg / .png / .dcm)</label>
        <input type="file" name="image" accept=".jpg,.jpeg,.png,.dcm,.dicom,image/jpeg,image/png" required/>
        <p style="margin-top:1rem;"><button type="submit" id="upload-btn">Analyze with MedGemma + Llama</button></p>
      </form>
      <p id="status"></p>
    </section>

    <section id="dashboard">
      <div class="grid">
        <aside>
          <div class="card">
            <h3 style="margin-top:0;">Image</h3>
            <p id="domain-badge" style="font-size:0.82rem;color:var(--muted);margin:0 0 0.5rem 0;"></p>
            <img id="preview" alt="Uploaded medical image"/>
            <p id="image-desc" style="font-size:0.9rem;color:var(--muted);"></p>
          </div>
          <div class="card">
            <h3 style="margin-top:0;">Triage</h3>
            <span id="triage-badge" class="badge urgent">—</span>
            <p id="triage-why" style="font-size:0.9rem;"></p>
          </div>
          <div class="card">
            <h3 style="margin-top:0;">Image findings</h3>
            <ul id="findings" class="compact"></ul>
          </div>
          <div class="card">
            <h3 style="margin-top:0;">Impression &amp; differential</h3>
            <p id="impression" style="font-size:0.92rem;"></p>
            <ul id="differential" class="compact"></ul>
          </div>
          <div class="card">
            <h3 style="margin-top:0;">Recommendations</h3>
            <ul id="recommendations" class="compact"></ul>
          </div>
          <div class="card refs">
            <h3 style="margin-top:0;">References</h3>
            <ul id="references" class="compact"></ul>
          </div>
          <div class="card">
            <h3 style="margin-top:0;">Evaluation metrics</h3>
            <ul id="metrics" class="compact"></ul>
          </div>
        </aside>
        <section class="card">
          <h2 style="margin-top:0;font-size:1.05rem;">2. Clinician chat</h2>
          <p class="note" id="correlation"></p>
          <div id="chat"></div>
          <div class="chips" id="suggested"></div>
          <div class="chat-row">
            <input id="chat-input" type="text" placeholder="Ask a follow-up question…"/>
            <button type="button" id="send-btn">Send</button>
          </div>
        </section>
      </div>
    </section>
  </main>
  <footer>
    Multimodal Clinical Assistant · Session data stored as conversation.json · Follow-up chat with dynamic re-triage
  </footer>
  <script>
    let sessionId = null;

    function triageClass(level) {
      const v = (level || "urgent").toLowerCase();
      return ["emergency", "urgent", "soon", "routine"].includes(v) ? v : "urgent";
    }

    function renderDashboard(data) {
      sessionId = data.session_id;
      document.getElementById("dashboard").style.display = "block";
      document.getElementById("preview").src = data.image_url + "?t=" + Date.now();
      const domain = (data.image_domain || "unknown").toLowerCase();
      document.getElementById("domain-badge").textContent =
        domain === "unknown"
          ? "Detected domain: pending"
          : "Auto-detected domain: " + domain;
      document.getElementById("image-desc").textContent = data.image_description || "";
      const badge = document.getElementById("triage-badge");
      badge.textContent = data.triage || "—";
      badge.className = "badge " + triageClass(data.triage);
      document.getElementById("triage-why").textContent = data.triage_rationale || "";
      document.getElementById("impression").textContent = data.impression || "";
      document.getElementById("correlation").textContent = data.image_note_correlation || "";
      fillList("findings", data.visual_findings || []);
      fillList("differential", data.differential || []);
      const recList = document.getElementById("recommendations");
      if (recList) fillList("recommendations", data.recommendations || []);
      const refs = document.getElementById("references");
      refs.innerHTML = "";
      (data.references || []).forEach(r => {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = r.url;
        a.target = "_blank";
        a.rel = "noopener";
        a.textContent = (r.cite_key ? r.cite_key + " " : "") + r.title;
        li.appendChild(a);
        refs.appendChild(li);
      });
      if (!(data.references || []).length) {
        refs.innerHTML = "<li>No linked guidelines for this case.</li>";
      }
      const ev = data.evaluation || {};
      const sat = ev.user_satisfaction || {};
      fillList("metrics", [
        "Cross-modal correlation: " + ((ev.cross_modal_correlation || {}).score),
        "Robustness to data variation: " + ((ev.robustness_to_data_variation || {}).score),
        "Explanation quality: " + ((ev.explanation_quality || {}).score),
        "User satisfaction (interface, 1–5): " + sat.interface_likert_1_to_5,
        "User satisfaction (explanations, 1–5): " + sat.explanation_likert_1_to_5
      ]);
      renderChat(data.conversation || []);
      renderChips(data.follow_up_questions || []);
    }

    function fillList(id, items) {
      const el = document.getElementById(id);
      el.innerHTML = "";
      (items || []).forEach(text => {
        const li = document.createElement("li");
        li.textContent = text;
        el.appendChild(li);
      });
    }

    function renderChat(conversation) {
      const chat = document.getElementById("chat");
      chat.innerHTML = "";
      conversation.forEach(turn => {
        const div = document.createElement("div");
        const role = turn.role === "clinician" ? "clinician" : "assistant";
        div.className = "msg " + role;
        div.innerHTML = "<div class=\"role\">" + role + "</div>" + escapeHtml(turn.content || "");
        chat.appendChild(div);
      });
      chat.scrollTop = chat.scrollHeight;
    }

    function renderChips(questions) {
      const box = document.getElementById("suggested");
      box.innerHTML = "";
      questions.forEach(q => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "chip";
        chip.textContent = q;
        chip.onclick = () => {
          document.getElementById("chat-input").value = q;
          sendMessage();
        };
        box.appendChild(chip);
      });
    }

    function escapeHtml(text) {
      return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\n/g, "<br/>");
    }

    function appendChatMessage(role, content) {
      const chat = document.getElementById("chat");
      const div = document.createElement("div");
      div.className = "msg " + role;
      div.innerHTML = "<div class=\"role\">" + role + "</div>" + escapeHtml(content || "");
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
    }

    document.getElementById("upload-form").addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const btn = document.getElementById("upload-btn");
      const status = document.getElementById("status");
      btn.disabled = true;
      status.textContent = "Uploading and running MedGemma + Llama (this may take several minutes)…";
      const form = ev.target;
      const body = new FormData(form);
      try {
        const res = await fetch("/ingest", { method: "POST", body });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || data.run_error || "Upload failed");
        renderDashboard(data);
        status.textContent = "Analysis complete. Ask follow-up questions in the chat panel.";
      } catch (err) {
        status.textContent = "Error: " + err.message;
      } finally {
        btn.disabled = false;
      }
    });

    async function sendMessage() {
      const input = document.getElementById("chat-input");
      const message = input.value.trim();
      if (!message || !sessionId) return;
      const btn = document.getElementById("send-btn");
      btn.disabled = true;
      input.value = "";
      appendChatMessage("clinician", message);
        document.getElementById("status").textContent =
        "Llama is re-triaging and replying (CPU or GPU via Ollama)…";
      try {
        const res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, message }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Chat failed");
        renderDashboard(data);
        document.getElementById("status").textContent = "Reply received.";
      } catch (err) {
        document.getElementById("status").textContent = "Error: " + err.message;
      } finally {
        btn.disabled = false;
      }
    }

    document.getElementById("send-btn").addEventListener("click", sendMessage);
    document.getElementById("chat-input").addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") sendMessage();
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    """Dashboard + upload page."""
    return HTML


@app.post("/ingest")
async def ingest(
    note: UploadFile = File(...),
    image: UploadFile = File(...),
):
    """De-identify uploads, run the LangGraph pipeline, return dashboard JSON."""
    try:
        note_bytes, note_name = _read_upload(note)
        image_bytes, image_name = _read_upload(image)
        ingested = ingest_pair(note_bytes, note_name, image_bytes, image_name)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc

    session_id = os.path.basename(ingested["session_dir"])
    client = _client()
    try:
        get_graph(client)
        record = process_case(
            client=client,
            vision_model=_vision_model(),
            llama_model=_llama_model(),
            case_id=0,
            image_path=ingested["image"]["path"],
            text_path=ingested["text"]["path"],
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OllamaError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    record["privacy"] = {
        "text": ingested["text"]["privacy"],
        "image": ingested["image"]["privacy"],
        "originals_stored": False,
    }

    save_json(os.path.join(ingested["session_dir"], "conversation.json"), record)
    _maybe_unload_vision(client)
    payload = dashboard_payload(record, session_id)
    payload["status"] = "analyzed"
    return JSONResponse(payload)


def _run_follow_up(session_id, message):
    client = _client()
    _maybe_unload_vision(client)
    return process_follow_up(client, session_id, message, _llama_model())


@app.post("/chat")
async def chat(body: ChatRequest):
    try:
        payload = await asyncio.to_thread(_run_follow_up, body.session_id, body.message)
        payload["status"] = "ok"
        return JSONResponse(payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except OllamaError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/session/{session_id}/image")
def session_image(session_id: str):
    """Serve the de-identified session image."""
    try:
        path = _session_image_path(session_id)
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    ext = os.path.splitext(path)[1].lower()
    media = IMAGE_MEDIA.get(ext, "application/octet-stream")
    return FileResponse(path, media_type=media)


def serve(host="127.0.0.1", port=8000):
    """Start the dashboard (used by python main.py)."""
    import uvicorn

    uvicorn.run("upload_app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    serve()
