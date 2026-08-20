"""
upload_app.py
-------------
Clinician dashboard: upload note + image, view findings, chat follow-ups.

Served by `python main.py` (FastAPI + inline HTML/JS, Python stack only).
"""

import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from ingestion import IMAGE_DOMAINS, ingest_pair
from medical_assistant import (
    dashboard_payload,
    get_graph,
    process_case,
    process_follow_up,
    save_json,
    session_dir_from_id,
)
from ollama_client import OllamaClient, OllamaError


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
    <p>Upload a de-identified note and image, review findings, then ask follow-up questions.</p>
  </header>
  <main>
    <section id="upload-panel" class="card">
      <h2 style="margin-top:0;font-size:1.05rem;">1. Upload case</h2>
      <p class="note">Files are de-identified before storage or LLM use. Educational prototype only.</p>
      <form id="upload-form">
        <label>Clinical note (.txt / .md)</label>
        <input type="file" name="note" accept=".txt,.md,text/plain" required/>
        <label>Medical image (.jpg / .png / .dcm)</label>
        <input type="file" name="image" accept=".jpg,.jpeg,.png,.dcm,.dicom,image/jpeg,image/png" required/>
        <label>Image domain</label>
        <select name="image_domain">
          <option value="radiology">Radiology</option>
          <option value="pathology">Pathology</option>
          <option value="dermatology">Dermatology</option>
          <option value="ophthalmology">Ophthalmology</option>
          <option value="other">Other</option>
        </select>
        <p style="margin-top:1rem;"><button type="submit" id="upload-btn">Analyze with MedGemma + Llama</button></p>
      </form>
      <p id="status"></p>
    </section>

    <section id="dashboard">
      <div class="grid">
        <aside>
          <div class="card">
            <h3 style="margin-top:0;">Image</h3>
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
          <div class="card refs">
            <h3 style="margin-top:0;">References</h3>
            <ul id="references" class="compact"></ul>
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
      document.getElementById("image-desc").textContent = data.image_description || "";
      const badge = document.getElementById("triage-badge");
      badge.textContent = data.triage || "—";
      badge.className = "badge " + triageClass(data.triage);
      document.getElementById("triage-why").textContent = data.triage_rationale || "";
      document.getElementById("impression").textContent = data.impression || "";
      document.getElementById("correlation").textContent = data.image_note_correlation || "";
      fillList("findings", data.visual_findings || []);
      fillList("differential", data.differential || []);
      const refs = document.getElementById("references");
      refs.innerHTML = "";
      (data.references || []).forEach(r => {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = r.url;
        a.target = "_blank";
        a.rel = "noopener";
        a.textContent = r.title;
        li.appendChild(a);
        refs.appendChild(li);
      });
      if (!(data.references || []).length) {
        refs.innerHTML = "<li>No linked guidelines for this case.</li>";
      }
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
        .replace(/\\n/g, "<br/>");
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
      document.getElementById("status").textContent = "Thinking…";
      try {
        const res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, message }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Chat failed");
        renderDashboard(data);
        input.value = "";
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
    image_domain: str = Form("radiology"),
):
    """De-identify uploads, run the LangGraph pipeline, return dashboard JSON."""
    if image_domain not in IMAGE_DOMAINS:
        raise HTTPException(400, "Invalid image_domain")
    try:
        note_bytes, note_name = _read_upload(note)
        image_bytes, image_name = _read_upload(image)
        ingested = ingest_pair(note_bytes, note_name, image_bytes, image_name, image_domain)
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
            image_domain=image_domain,
        )
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
    payload = dashboard_payload(record, session_id)
    payload["status"] = "analyzed"
    return JSONResponse(payload)


@app.post("/chat")
async def chat(body: ChatRequest):
    try:
        payload = process_follow_up(_client(), body.session_id, body.message, _llama_model())
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
