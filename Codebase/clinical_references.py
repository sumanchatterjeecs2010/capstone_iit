"""
clinical_references.py
----------------------
Guideline links: Llama calls a LangChain search tool (WHO / NIH / MedlinePlus / CDC).
Offline tests fall back to keyword matching if no LLM client is passed.
"""

import json

from langchain_core.tools import tool

from ollama_client import extract_json, strip_thinking


ALLOWED_HOSTS = (
    "who.int",
    "nih.gov",
    "medlineplus.gov",
    "cancer.gov",
    "cdc.gov",
)

FALLBACK_GUIDELINES = [
    {
        "keywords": ["pneumonia", "cap", "consolidation", "infiltrate"],
        "title": "WHO — Community-acquired pneumonia overview",
        "url": "https://www.who.int/news-room/fact-sheets/detail/pneumonia",
    },
    {
        "keywords": ["heart failure", "cardiomegaly", "dyspnoea", "dyspnea"],
        "title": "WHO — Cardiovascular diseases fact sheet",
        "url": "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
    },
    {
        "keywords": ["stroke", "hemiparesis", "aphasia", "brain", "ischemia", "hemorrhage"],
        "title": "WHO — Stroke fact sheet",
        "url": "https://www.who.int/news-room/fact-sheets/detail/stroke",
    },
    {
        "keywords": ["lymphoma", "hodgkin", "reed", "lymphadenopathy"],
        "title": "NIH NCI — Hodgkin lymphoma",
        "url": "https://www.cancer.gov/types/lymphoma/patient/adult-hodgkin-treatment-pdq",
    },
    {
        "keywords": ["carcinoma", "ductal", "breast", "histology", "biopsy"],
        "title": "NIH NCI — Breast cancer",
        "url": "https://www.cancer.gov/types/breast/patient/breast-treatment-pdq",
    },
    {
        "keywords": ["fracture", "wrist", "bone", "orthopedic", "scaphoid"],
        "title": "NIH MedlinePlus — Fractures overview",
        "url": "https://medlineplus.gov/fractures.html",
    },
]

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_clinical_guidelines",
            "description": (
                "Search public clinical sources (WHO, NIH, MedlinePlus, CDC) "
                "for guideline pages matching a case. Pass a short medical query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'community acquired pneumonia WHO guideline'",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def _host_allowed(url):
    url = (url or "").lower()
    return any(host in url for host in ALLOWED_HOSTS) and url.startswith("http")


def web_search_guidelines(query, max_results=6):
    """Run a live web search restricted to trusted clinical hosts."""
    query = (query or "").strip()
    if not query:
        return []
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
    except ImportError:
        return []

    search = "{} site:who.int OR site:nih.gov OR site:medlineplus.gov OR site:cdc.gov".format(query)
    hits = []
    try:
        with DDGS() as ddgs:
            for row in ddgs.text(search, max_results=max_results) or []:
                url = row.get("href") or row.get("url") or ""
                if not _host_allowed(url):
                    continue
                hits.append(
                    {
                        "title": row.get("title") or url,
                        "url": url,
                        "snippet": row.get("body") or "",
                    }
                )
    except Exception:
        return []
    return hits[:5]


@tool
def search_clinical_guidelines(query: str) -> str:
    """Search WHO, NIH, MedlinePlus, and CDC for clinical guideline pages."""
    return json.dumps(web_search_guidelines(query), ensure_ascii=False)


def _run_tool(name, arguments):
    if name != "search_clinical_guidelines":
        return json.dumps({"error": "unknown tool"})
    query = arguments.get("query") if isinstance(arguments, dict) else ""
    return search_clinical_guidelines.invoke({"query": query})


def _normalize_refs(items):
    refs, seen = [], set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or url).strip()
        if not _host_allowed(url) or url in seen:
            continue
        seen.add(url)
        refs.append(
            {
                "id": len(refs) + 1,
                "cite_key": "[{}]".format(len(refs) + 1),
                "title": title,
                "url": url,
                "type": "clinical guideline",
                "source": item.get("source") or "tool_search",
            }
        )
        if len(refs) >= 5:
            break
    return refs


def _keyword_fallback(record):
    blob = json.dumps(record).lower()
    items = []
    for item in FALLBACK_GUIDELINES:
        if any(kw in blob for kw in item["keywords"]):
            items.append({"title": item["title"], "url": item["url"], "source": "offline_fallback"})
    return _normalize_refs(items)


def _case_brief(record):
    visual = record.get("visual_analysis") or {}
    reasoning = record.get("clinical_reasoning") or {}
    return {
        "impression": reasoning.get("impression"),
        "differential": reasoning.get("differential"),
        "triage": reasoning.get("triage"),
        "visual_findings": (visual.get("visual_findings") or [])[:6],
        "modality": visual.get("inferred_modality"),
        "note_excerpt": str(record.get("note") or record.get("note_text") or "")[:800],
    }


def attach_references(record, client=None, llama_model=None):
    """
    Prefer Llama + LangChain search tool. If no client (unit tests) or search fails,
    fall back to keyword matching on the local guideline list.
    """
    if client is None or not llama_model:
        return _keyword_fallback(record)
    try:
        refs = _attach_with_tools(record, client, llama_model)
        if refs:
            return refs
    except Exception:
        pass
    return _keyword_fallback(record)


def _attach_with_tools(record, client, llama_model):
    print("  Llama: selecting guideline search via tool...", flush=True)
    brief = json.dumps(_case_brief(record), indent=2)
    messages = [
        {
            "role": "system",
            "content": (
                "You retrieve clinical guidelines. Call search_clinical_guidelines "
                "with one focused query. Only WHO, NIH, MedlinePlus, or CDC pages."
            ),
        },
        {
            "role": "user",
            "content": (
                "Call the search_clinical_guidelines tool for this case, then wait for results.\n\n"
                "{}".format(brief)
            ),
        },
    ]
    first = client.chat_messages(
        model=llama_model,
        messages=messages,
        tools=TOOL_SCHEMA,
        temperature=0.1,
        max_tokens=220,
        keep_alive="10m",
    )
    tool_calls = first.get("tool_calls") or []
    if not tool_calls:
        parsed = extract_json(strip_thinking(first.get("content") or "")) or {}
        query = parsed.get("query") or parsed.get("search") or ""
        if not query:
            impression = str((record.get("clinical_reasoning") or {}).get("impression") or "")
            query = impression[:120] or "clinical guideline radiology pathology"
        hits = web_search_guidelines(query)
        return _normalize_refs(hits)

    messages.append(
        {
            "role": "assistant",
            "content": first.get("content") or "",
            "tool_calls": tool_calls,
        }
    )
    for call in tool_calls:
        function = call.get("function") or call
        name = function.get("name") or call.get("name") or "search_clinical_guidelines"
        raw_args = function.get("arguments") or call.get("arguments") or {}
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                raw_args = {"query": raw_args}
        result = _run_tool(name, raw_args)
        messages.append(
            {
                "role": "tool",
                "tool_name": name,
                "content": result,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": (
                "From the tool results, return JSON only: "
                '{"references": [{"title": "...", "url": "..."}]} '
                "Keep 2-5 WHO/NIH/MedlinePlus/CDC links."
            ),
        }
    )
    second = client.chat_messages(
        model=llama_model,
        messages=messages,
        temperature=0.1,
        max_tokens=280,
        keep_alive="10m",
        json_mode=True,
    )
    parsed = extract_json(strip_thinking(second.get("content") or "")) or {}
    items = parsed.get("references") if isinstance(parsed, dict) else parsed
    refs = _normalize_refs(items)
    if refs:
        return refs
    # If the model did not return JSON, use the raw tool hits.
    last_tool = None
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            last_tool = msg.get("content")
            break
    try:
        hits = json.loads(last_tool) if last_tool else []
    except (TypeError, json.JSONDecodeError):
        hits = []
    return _normalize_refs(hits)


def format_reference_block(refs):
    if not refs:
        return ""
    lines = ["Cite these evidence sources inline as [1], [2], etc. when stating clinical facts:"]
    for ref in refs:
        lines.append("{} {} — {}".format(ref["cite_key"], ref["title"], ref["url"]))
    return "\n".join(lines)


def format_reference_footer(refs):
    if not refs:
        return ""
    lines = ["References:"]
    for ref in refs:
        lines.append("{} {} — {}".format(ref["cite_key"], ref["title"], ref["url"]))
    return "\n".join(lines)


def enrich_reply_with_citations(text, refs):
    text = (text or "").strip()
    if not refs:
        return text
    footer = format_reference_footer(refs)
    if footer and footer.split("\n", 1)[0] not in text:
        text = text + "\n\n" + footer
    return text
