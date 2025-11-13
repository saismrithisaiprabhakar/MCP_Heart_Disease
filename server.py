# ===========================
# server.py — Async + Dynamic Context Ranking + Custom Evaluators
# ===========================

import nltk
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("omw-1.4", quiet=True)

from fastmcp import FastMCP
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv, find_dotenv
from groq import Groq
import os
import re
import json
import asyncio
import traceback
import time
import contextlib  # <-- needed for nullcontext

# ================== LANGFUSE (v3.x API) ==================
from langfuse import get_client

try:
    langfuse = get_client()  # reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
    print("[Langfuse] Initialized successfully (v3 API).")
except Exception as e:
    langfuse = None
    print(f"[Langfuse] Initialization failed: {e}")

# ================== FLAGS & GLOBALS ==================
EVAL_ENABLED = True
SIM_READY = False
sim_model = None

# Guardrail config
PII_POLICY = os.getenv("PII_POLICY", "block").lower()   # "block" or "redact"
EVAL_THRESHOLDS = {
    "faithfulness": float(os.getenv("THRESH_FAITH", 0.70)),
    "fairness":     float(os.getenv("THRESH_FAIR",  0.70)),
    "relevance":    float(os.getenv("THRESH_REL",   0.70)),
}

# ================== PATHS ==================
RAG_DB_PATH = "./data_summaries_chroma"
RAG_COLLECTION_NAME = "chunk_summaries"
PDF_DB_PATH = "./pdf_database_docs"
PDF_COLLECTION_NAME = "pdf_context_collection"

# ================== MODEL SETTINGS ==================
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

app = FastMCP("RAG + MCP Combined Server")

# ================== LOAD VECTOR STORES ==================
def load_vector_stores():
    rag_client = chromadb.PersistentClient(
        path=RAG_DB_PATH, settings=Settings(anonymized_telemetry=False)
    )
    rag_collection = rag_client.get_or_create_collection(
        RAG_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    pdf_client = chromadb.PersistentClient(
        path=PDF_DB_PATH, settings=Settings(anonymized_telemetry=False)
    )
    pdf_collection = pdf_client.get_collection(PDF_COLLECTION_NAME)
    return rag_collection, pdf_collection

rag_collection, pdf_collection = load_vector_stores()

# ================== LOAD ENV KEYS ==================
from cryptography.fernet import Fernet

dotenv_path = find_dotenv("keys.env", usecwd=True)
load_dotenv(dotenv_path)
print(dotenv_path)

fernet_key = os.getenv("FERNET_KEY")
encrypted_key = os.getenv("ENCRYPTED_GROQ_API_KEY")

if fernet_key and encrypted_key:
    try:
        groq_api_key = Fernet(fernet_key.encode()).decrypt(encrypted_key.encode()).decode()
        os.environ["GROQ_API_KEY"] = groq_api_key
        print("Encrypted GROQ key decrypted successfully.")
    except Exception as e:
        print(f"Failed to decrypt GROQ_API_KEY: {e}")
else:
    print("Using plain GROQ_API_KEY from .env (not encrypted).")

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

# ================== SENTENCE SIMILARITY ==================
try:
    from sentence_transformers import SentenceTransformer, util
    sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    SIM_READY = True
except Exception as e:
    print("[WARN] SentenceTransformer unavailable:", e)
    SIM_READY = False

# ================== SANITIZER PLACEHOLDER ==================
def sanitize_input(text: str) -> str:
    return text

# ================== PII DETECTION / REDACTION (Lightweight) ==================
# NOTE: This is intentionally lightweight. For production, consider a stronger NER/PII detector.
_PII_PATTERNS = {
    "email":  re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone":  re.compile(r"\b(?:\+?\d{1,2}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"),
    "ssn":    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "name":   re.compile(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b"),  # naive full-name pattern
    # Add more as needed (MRN, DOB formats, addresses, etc.)
}

def detect_pii(text: str) -> list[str]:
    if not text:
        return []
    hits = []
    for k, pat in _PII_PATTERNS.items():
        if pat.search(text):
            hits.append(k)
    return sorted(set(hits))

def redact_pii(text: str) -> str:
    if not text:
        return text
    redacted = text
    for k, pat in _PII_PATTERNS.items():
        redacted = pat.sub("[REDACTED]", redacted)
    return redacted

# ================== TEXT NORMALIZER ==================
def _normalize_for_eval(text: str) -> str:
    if not text:
        return ""
    text = text.replace("###", " ").replace("---", " ")
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ================== CUSTOM SCORING HELPERS ==================
def _groq_text(prompt: str) -> str:
    try:
        resp = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=64,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("[EVAL ERROR] Groq call failed:", e)
        return "0.0"

async def _soft_llm_score(prompt: str) -> float:
    text = await asyncio.to_thread(_groq_text, prompt)
    m = re.findall(r"\d*\.?\d+", text)
    try:
        score = float(m[-1]) if m else 0.0
        return max(0.0, min(1.0, score))
    except:
        return 0.0

# ================== SCORING PROMPTS ==================
def build_faithfulness_prompt(context, response):
    return f"""
You are an impartial evaluator for medical question answering.
Rate how faithfully the response follows the provided context — i.e., whether it stays factually consistent without adding unrelated or incorrect details.

Scoring (continuous 0.0–1.0):
- 0.90–1.00 → Fully supported; no contradictions or new claims.
- 0.70–0.89 → Mostly supported; small omissions or rewordings.
- 0.40–0.69 → Partially supported; mix of supported and inferred info.
- 0.10–0.39 → Weakly supported or speculative.
- 0.00–0.09 → Contradicts or invents facts.

Context:
{context}

Response:
{response}
""".strip()

def build_fairness_prompt(context, response):
    return f"""
You are an evaluator of fairness and neutrality for medical explanations.
Judge whether the response is unbiased, balanced, and avoids unfair generalizations.

Scoring (continuous 0.0–1.0):
- 0.90–1.00 → Completely neutral and balanced.
- 0.70–0.89 → Mostly fair; minor imbalance or assertive tone.
- 0.40–0.69 → Somewhat biased or one-sided.
- 0.10–0.39 → Clearly biased, alarmist, or exaggerated.
- 0.00–0.09 → Strongly biased or discriminatory.

Context:
{context}

Response:
{response}
""".strip()

# ================== MAIN EVALUATOR ==================
async def evaluate_response_async(question: str, context: str, answer: str) -> dict:
    results = {
        "faithfulness": None,
        "relevance": None,
        "fairness": None,
        "semantic_similarity": None,
        "summary": None,
        "evaluator": "Groq Soft Evaluator",
    }

    clean_context = _normalize_for_eval(context or "")
    faith_score = await _soft_llm_score(build_faithfulness_prompt(clean_context, answer))
    fair_score = await _soft_llm_score(build_fairness_prompt(clean_context, answer))
    results["faithfulness"] = round(faith_score, 2)
    results["fairness"] = round(fair_score, 2)

    if SIM_READY and context and answer:
        try:
            emb_answer = sim_model.encode(answer, convert_to_tensor=True)
            emb_context = sim_model.encode(context, convert_to_tensor=True)
            sim_val = float(util.cos_sim(emb_answer, emb_context).item())
            results["semantic_similarity"] = round(sim_val * 100, 1)
        except Exception as e:
            print("[EVAL WARNING] Semantic-similarity failed:", e)

    try:
        from llama_index.core.evaluation.context_relevancy import (
            ContextRelevancyEvaluator as RelevanceEvaluator,
        )
        from llama_index.llms.groq import Groq as LlamaGroq
        llama_eval_llm = LlamaGroq(model=MODEL, api_key=os.getenv("GROQ_API_KEY"))
        relevance_eval = RelevanceEvaluator(llm=llama_eval_llm)
        rel_obj = await asyncio.to_thread(
            relevance_eval.evaluate,
            query=question,
            contexts=[clean_context],
            response=answer,
        )
        results["relevance"] = round(float(rel_obj.score), 2)
    except Exception as e:
        print("[EVAL WARNING] Relevance evaluator failed:", e)
        results["relevance"] = 0.0

    results["summary"] = (
        f"Faithfulness: {results['faithfulness']}, "
        f"Relevance: {results['relevance']}, "
        f"Fairness: {results['fairness']}, "
        f"Semantic similarity: {results.get('semantic_similarity', 0.0)}"
    )
    for key in ["faithfulness", "relevance", "fairness", "semantic_similarity"]:
        if results[key] is None:
            results[key] = 0.0

    return results

# ================== MAIN TOOL ==================
@app.tool()
async def ask_with_combined_context(question: str) -> dict:
    """Ask across RAG & PDF vector stores, re-rank context, generate & evaluate."""
    try:
        if not question.strip():
            return {"error": "Question cannot be empty."}

        print(f"\n[REQUEST] {question}")

        # ---------- PII Guardrail: detect + enforce policy ----------
        pii_hits = detect_pii(question)
        if pii_hits:
            print(f"[GUARDRAIL] PII detected: {pii_hits} (policy={PII_POLICY})")
            if langfuse:
                try:
                    # If a span exists later, this metadata attaches to the active trace;
                    # otherwise it still records at client-level trace.
                    langfuse.update_trace(metadata={"pii_detected": pii_hits, "pii_policy": PII_POLICY})
                except Exception as e:
                    print(f"[Langfuse] update_trace (PII) failed: {e}")

            if PII_POLICY == "block":
                return {
                    "answer": "⚠️ Personal identifiers detected. Please remove names, emails, phone numbers, or IDs and try again.",
                    "evaluation": {"faithfulness": 0.0, "fairness": 0.0, "relevance": 0.0, "semantic_similarity": 0.0}
                }
            # else redact
            question_sanitized = redact_pii(question)
        else:
            question_sanitized = question

        # ---------- Root span using Langfuse v3 ----------
        if langfuse:
            span_ctx = langfuse.start_as_current_span(
                name="ask_with_combined_context",
                input={"question": question_sanitized},
            )
        else:
            span_ctx = None

        with (span_ctx or contextlib.nullcontext()):
            if langfuse:
                try:
                    langfuse.update_trace(
                        name="ask_with_combined_context",
                        metadata={
                            "source": "FastMCP-Server",
                            "model": MODEL,
                            "context_type": "combined",
                            "pii_policy_active": PII_POLICY if pii_hits else "none",
                        },
                    )
                except Exception as e:
                    print(f"[Langfuse] update_trace failed: {e}")

            # 1) Query vector stores
            start = time.time()

            async def query_collection(collection, query_texts, n_results):
                return await asyncio.to_thread(
                    collection.query,
                    query_texts=query_texts,
                    n_results=n_results,
                    include=["documents", "distances"],
                )

            df_results, pdf_results = await asyncio.gather(
                query_collection(rag_collection, [question_sanitized], 12),
                query_collection(pdf_collection, [question_sanitized], 12),
            )

            df_docs = [doc for sub in df_results.get("documents", []) for doc in sub]
            pdf_docs = [doc for sub in pdf_results.get("documents", []) for doc in sub]
            df_dist = df_results.get("distances", [[]])[0]
            pdf_dist = pdf_results.get("distances", [[]])[0]

            def _pair(docs, dist, src):
                out = []
                for i, d in enumerate(docs):
                    base = 1 - dist[i] if i < len(dist) and dist[i] is not None else 0.0
                    out.append({"text": d, "source": src, "base": base})
                return out

            pairs = _pair(df_docs, df_dist, "RAG") + _pair(pdf_docs, pdf_dist, "PDF")
            print(f"[INFO] Retrieved {len(pairs)} documents (SIM_READY={SIM_READY})")

            if SIM_READY:
                q_emb = sim_model.encode(question_sanitized, convert_to_tensor=True)
                for p in pairs:
                    emb_doc = sim_model.encode(p["text"], convert_to_tensor=True)
                    sim = float(util.cos_sim(q_emb, emb_doc).item())
                    p["score"] = (p["base"] + (sim + 1) / 2) / 2
            else:
                for p in pairs:
                    p["score"] = p["base"]

            pairs.sort(key=lambda x: x["score"], reverse=True)

            # 2) Combine context
            TOP_K = 10
            top_texts = [p["text"] for p in pairs[:TOP_K]]
            combined_context = _normalize_for_eval("\n\n".join(top_texts))

            # 3) Build LLM prompt (use sanitized question)
            prompt = f"""
            You are a helpful medical assistant.
            Use only the given context to answer.
            Use subheadings to give your answer and be mindful of tone.

            CONTEXT:
            {combined_context}

            QUESTION:
            {question_sanitized}

            ANSWER:
            """.strip()

            # 4) Generate answer (tracked as generation span)
            with langfuse.start_as_current_generation(
                name="groq_generation",
                model=MODEL,
                input=prompt,
            ) if langfuse else contextlib.nullcontext() as gen:
                answer = await asyncio.to_thread(
                    lambda: groq_client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=450,
                    ).choices[0].message.content.strip()
                )
                if gen:
                    gen.update(output=answer)

            print(f"[ANSWER] Generated answer ({len(answer)} chars).")

            # 5) Evaluate response
            evaluation = await evaluate_response_async(question_sanitized, combined_context, answer)

            # ---------- Threshold enforcement (optional) ----------
            threshold_flags = {k: evaluation.get(k, 0.0) >= v for k, v in EVAL_THRESHOLDS.items()}
            passed_all = all(threshold_flags.values())
            if langfuse:
                try:
                    langfuse.update_trace(metadata={
                        "guardrail_status": "pass" if passed_all else "fail",
                        "thresholds": EVAL_THRESHOLDS,
                        "scores": {k: evaluation.get(k, 0.0) for k in ["faithfulness","fairness","relevance"]},
                    })
                except Exception as e:
                    print(f"[Langfuse] update_trace (thresholds) failed: {e}")

            if not passed_all:
                warning_msg = "⚠️ Guardrail triggered: response quality below thresholds."
                print(warning_msg)
                return {
                    "answer": warning_msg,
                    "evaluation": evaluation
                }

            # Optional evaluation span
            if langfuse:
                try:
                    with langfuse.start_as_current_span(
                        name="evaluation",
                        input={"question": question_sanitized},
                        output=evaluation,
                    ):
                        pass
                except Exception as e:
                    print(f"[Langfuse] evaluation span failed: {e}")

            # Ensure Langfuse sends data
            if langfuse:
                langfuse.flush()

            print("[EVAL]", json.dumps(evaluation, indent=2))
            return {"answer": answer, "evaluation": evaluation}

    except Exception as e:
        print("[FATAL ERROR]", e)
        traceback.print_exc()
        return {"error": str(e)}

# ================== MAIN ENTRY ==================
if __name__ == "__main__":
    app.run(transport="http", host="0.0.0.0", port=8000)
