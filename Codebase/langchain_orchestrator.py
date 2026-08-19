"""
langchain_orchestrator.py
-------------------------
LangChain is used here for the jobs named in the project brief:

- Prompt templates: reusable, named prompts for extraction, reasoning and dialogue.
- Retrieval: a retriever over CLIP+BioBERT fused findings so the LLM sees
  only the top cross-modal evidence (not the raw image tensor).
- Orchestration: LCEL chains compose retrieve → prompt → Llama.

The actual Llama weights still run through our Ollama HTTP client so
keep_alive and JSON repair stay under our control.
"""

from typing import Any, List

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from pydantic import ConfigDict, Field


class LocalOllamaLLM(LLM):
    """
    LangChain LLM adapter around the local Ollama HTTP client.

    Parameters
    ----------
    client : object
        An OllamaClient instance.
    model_name : str
        Ollama model tag, for example llama3.2:3b.
    system_prompt : str
        System instruction sent with each call.
    temperature : float
        Decoding temperature.
    max_tokens : int
        Maximum generated tokens.
    json_mode : bool
        Request constrained JSON from Ollama.
    keep_alive : str
        How long Ollama should keep weights loaded.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any
    model_name: str
    system_prompt: str = ""
    temperature: float = 0.2
    max_tokens: int = 400
    json_mode: bool = False
    keep_alive: str = "5m"

    @property
    def _llm_type(self):
        """Return the LangChain type name for this wrapper."""
        return "local_ollama"

    def _call(
        self,
        prompt,
        stop=None,
        run_manager=None,
        **kwargs
    ):
        """
        Generate text for one LangChain prompt.

        Parameters
        ----------
        prompt : str
            Prompt produced by a PromptTemplate.
        stop : list or None
            Unused; Ollama stop sequences are not required here.
        run_manager : CallbackManagerForLLMRun or None
            LangChain callback manager.
        **kwargs
            Extra LangChain arguments (ignored).

        Returns
        -------
        str
            Raw model text.
        """
        _ = stop, run_manager, kwargs
        return self.client.generate(
            model=self.model_name,
            prompt=prompt,
            system=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            keep_alive=self.keep_alive,
            json_mode=self.json_mode,
        )


class CrossModalFindingRetriever(BaseRetriever):
    """
    Retrieve the top fused image-text findings for a clinical query.

    The 'documents' are not a hospital EHR dump. They are the CLIP+BioBERT
    ranked findings for this case. The query is the patient note (or a
    clinician follow-up). This is the brief's 'cross-modal retrieval' step:
    given a text query, return the image-linked findings that match it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    findings: List[dict] = Field(default_factory=list)
    k: int = 4

    def _get_relevant_documents(
        self,
        query,
        *,
        run_manager=None,
    ):
        """
        Return the top-k fused findings as LangChain Documents.

        Parameters
        ----------
        query : str
            Patient note or clinician question. Used only as the retrieval
            query object; ranking already lives in fused_score.
        run_manager : CallbackManagerForRetrieverRun or None
            LangChain callback manager.

        Returns
        -------
        list[Document]
            Top findings with scores in metadata.
        """
        _ = query, run_manager
        ranked = sorted(
            self.findings or [],
            key=lambda row: float(row.get("fused_score") or 0.0),
            reverse=True,
        )
        documents = []
        for row in ranked[: self.k]:
            page = (
                "{label} (fused={fused:.2f}, image={image:.2f}, note={note:.2f})"
            ).format(
                label=row.get("label", "unknown"),
                fused=float(row.get("fused_score") or 0.0),
                image=float(row.get("image_score") or 0.0),
                note=float(row.get("note_score") or 0.0),
            )
            documents.append(Document(page_content=page, metadata=row))
        return documents


ENTITY_PROMPT = PromptTemplate.from_template(
    "Extract key facts from this patient note. Return JSON only with keys: "
    "age_sex, chief_complaint, history, vitals, exam, medications, "
    "requested_study, clinical_question.\n\nNOTE:\n{note}"
)

REASONING_PROMPT = PromptTemplate.from_template(
    "Fuse the retrieved image-text findings with the patient note. "
    "Return JSON only with keys: impression, differential (list of clinical conditions), "
    "triage (emergency|urgent|soon|routine), triage_rationale, recommendations (list), "
    "follow_up_questions (list), image_text_correlation, safety_flags (list).\n"
    "Use RETRIEVED_FINDINGS as image evidence. Do not treat exam signs as if they were on the image.\n\n"
    "RETRIEVED_FINDINGS:\n{retrieved}\n\n"
    "OTHER_EVIDENCE:\n{evidence}\n\n"
    "FULL_NOTE:\n{note}\n"
)

DIALOGUE_PROMPT = PromptTemplate.from_template(
    "{instruction}\n\nUse only these facts:\n{context}\n\n"
    "Write 80-120 words, plain language for a clinician. "
    "End with the educational disclaimer sentence."
)


def format_retrieved_docs(docs):
    """
    Turn retriever Documents into a bullet list for the reasoning prompt.

    Parameters
    ----------
    docs : list[Document]
        Findings returned by CrossModalFindingRetriever.

    Returns
    -------
    str
        Human-readable retrieved evidence.
    """
    if not docs:
        return "(no fused findings retrieved)"
    return "\n".join("- " + doc.page_content for doc in docs)


def retrieve_findings(visual, query, k=4):
    """
    Run LangChain retrieval over fused CLIP+BioBERT findings.

    Parameters
    ----------
    visual : dict
        Visual record that contains fused_findings.
    query : str
        Patient note or clinician question.
    k : int
        Number of findings to keep.

    Returns
    -------
    list[Document]
        Top-k retrieved finding documents.
    """
    retriever = CrossModalFindingRetriever(
        findings=list(visual.get("fused_findings") or []),
        k=k,
    )
    documents = retriever.invoke(query)
    visual["retrieved_findings"] = [doc.page_content for doc in documents]
    visual["langchain_retriever"] = "CrossModalFindingRetriever"
    return documents


def run_entity_chain(client, model, note, system_prompt):
    """
    LangChain chain: entity PromptTemplate → Llama JSON.

    Parameters
    ----------
    client : OllamaClient
        Local Ollama wrapper.
    model : str
        Llama tag.
    note : str
        Patient note.
    system_prompt : str
        System instruction.

    Returns
    -------
    str
        Raw Llama output.
    """
    llm = LocalOllamaLLM(
        client=client,
        model_name=model,
        system_prompt=system_prompt,
        temperature=0.1,
        max_tokens=280,
        json_mode=True,
        keep_alive="5m",
    )
    chain = ENTITY_PROMPT | llm
    return chain.invoke({"note": note})


def run_reasoning_chain(client, model, visual, evidence_json, note, system_prompt):
    """
    LangChain chain: retrieve findings → reasoning PromptTemplate → Llama JSON.

    Parameters
    ----------
    client : OllamaClient
        Local Ollama wrapper.
    model : str
        Llama tag.
    visual : dict
        Visual record with fused_findings.
    evidence_json : str
        Compact extra evidence (entities, modality).
    note : str
        Patient note.
    system_prompt : str
        System instruction.

    Returns
    -------
    str
        Raw Llama output.
    """
    llm = LocalOllamaLLM(
        client=client,
        model_name=model,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=520,
        json_mode=True,
        keep_alive="5m",
    )
    retriever = CrossModalFindingRetriever(
        findings=list(visual.get("fused_findings") or []),
        k=4,
    )

    def _retrieve(payload):
        """Retrieve findings for the note inside the LCEL chain."""
        docs = retriever.invoke(payload["note"])
        payload = dict(payload)
        payload["retrieved"] = format_retrieved_docs(docs)
        visual["retrieved_findings"] = [doc.page_content for doc in docs]
        visual["langchain_retriever"] = "CrossModalFindingRetriever"
        return payload

    chain = RunnableLambda(_retrieve) | REASONING_PROMPT | llm
    return chain.invoke({"note": note, "evidence": evidence_json})


def run_dialogue_chain(client, model, instruction, context, system_prompt):
    """
    LangChain chain: dialogue PromptTemplate → Llama prose.

    Parameters
    ----------
    client : OllamaClient
        Local Ollama wrapper.
    model : str
        Llama tag.
    instruction : str
        What this conversation turn should do.
    context : str
        Allowed facts.
    system_prompt : str
        System instruction.

    Returns
    -------
    str
        Assistant utterance.
    """
    llm = LocalOllamaLLM(
        client=client,
        model_name=model,
        system_prompt=system_prompt,
        temperature=0.3,
        max_tokens=260,
        json_mode=False,
        keep_alive="5m",
    )
    chain = DIALOGUE_PROMPT | llm
    return chain.invoke({"instruction": instruction, "context": context})
