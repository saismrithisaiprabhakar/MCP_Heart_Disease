# ===========================
# server.py — Full RAG + MCP + PII Guardrails + Evaluation
# ===========================

from fastmcp import FastMCP
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv, find_dotenv
from groq import Groq
from cryptography.fernet import Fernet
import os
import re
import json
import asyncio
import traceback

# ================== PRESIDIO — Full PII Detection ==================
# ================== PRESIDIO (Full PII Detection — Version Compatible) ==================
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

PRESIDIO_THRESHOLD = float(os.getenv("PRESIDIO_THRESHOLD", 0.50))

# PII categories we want to detect
_PII_TARGETS = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "PERSON",
    "US_SSN",
    "LOCATION",
    "ADDRESS",
]

def _init_presidio():
    try:
        print("[Presidio] Loading en_core_web_lg via NlpEngineProvider…")

        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "en", "model_name": "en_core_web_lg"}
            ]
        }

        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        nlp_engine = provider.create_engine()

        registry = RecognizerRegistry()

        # Load all recognizers your version supports
        registry.load_predefined_recognizers()

        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            registry=registry
        )
        anonymizer = AnonymizerEngine()

        print("[Presidio] SUCCESS: All built-in recognizers loaded.")
        return analyzer, anonymizer, True

    except Exception as e:
        print("[Presidio INIT ERROR]:", e)
        return None, None, False

analyzer, anonymizer, PRESIDIO_READY = _init_presidio()


def detect_pii(text: str):
    if not text or not PRESIDIO_READY:
        return []
    try:
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=_PII_TARGETS
        )
        return sorted(set(r.entity_type for r in results))
    except Exception as e:
        print("[PII DETECTION ERROR]", e)
        return []


def redact_pii(text: str) -> str:
    if not text or not PRESIDIO_READY:
        return text
    try:
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=_PII_TARGETS
        )

        if not results:
            return text

        operators = {
            r.entity_type: OperatorConfig("replace", {"new_value": "[REDACTED]"})
            for r in results
        }

        return anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators
        ).text

    except Exception as e:
        print("[PII REDACTION ERROR]", e)
        return text
# ================== GLOBAL FLAGS ==================
PII_POLICY = os.getenv("PII_POLICY", "redact").lower()
EVAL_ENABLED = True
SIM_READY = False

# ================== PATHS ==================
RAG_DB_PATH = "./data_summaries_chroma"
RAG_COLLECTION_NAME = "chunk_summaries"
PDF_DB_PATH = "./pdf_database_docs"
PDF_COLLECTION_NAME = "pdf_context_collection"

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

app = FastMCP("RAG + MCP Combined Server")

# ================== LOAD VECTOR STORES ==================
def load_vector_stores():
    rag_client = chromadb.PersistentClient(
        path=RAG_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )

    rag_collection = rag_client.get_or_create_collection(
        RAG_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    pdf_client = chromadb.PersistentClient(
        path=PDF_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    pdf_collection = pdf_client.get_collection(PDF_COLLECTION_NAME)

    return rag_collection, pdf_collection

rag_collection, pdf_collection = load_vector_stores()

# ================== LOAD KEYS ==================
dotenv_path = find_dotenv("keys.env", usecwd=True)
load_dotenv(dotenv_path)

fernet_key = os.getenv("FERNET_KEY")
encrypted_key = os.getenv("ENCRYPTED_GROQ_API_KEY")

if fernet_key and encrypted_key:
    try:
        groq_api_key = Fernet(fernet_key.encode()).decrypt(
            encrypted_key.encode()
        ).decode()
        os.environ["GROQ_API_KEY"] = groq_api_key
        print("Encrypted GROQ key decrypted.")
    except:
        pass

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ================== SENTENCE SIMILARITY ==================
try:
    from sentence_transformers import SentenceTransformer, util
    sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    SIM_READY = True
    print("[SIM] Loaded sentence-transformers.")
except:
    print("[SIM WARNING] Not available.")
    SIM_READY = False

# ================== NORMALIZE ==================
def _normalize(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

# ================== EVALUATION PROMPTS ==================
def build_faithfulness_prompt(context, response):
    return f"""
Evaluate FAITHFULNESS.

Definition:
How accurately does the response reflect ONLY the context?

Scores:
1.0–0.90 = Fully faithful.
0.89–0.70 = Mostly faithful.
0.69–0.40 = Some unsupported claims.
0.39–0.10 = Largely unsupported.
0.09–0.00 = Contradicts/makes up facts.

Context:
{context}

Response:
{response}

Only output a number.
""".strip()

def build_relevance_prompt(context, response):
    return f"""
Evaluate RELEVANCE.

Definition:
How well does the answer address the question and use the context?

Scores:
1.0–0.90 = Highly relevant.
0.89–0.70 = Mostly relevant.
0.69–0.40 = Somewhat relevant.
0.39–0.10 = Weak relevance.
0.09–0.00 = Not relevant.

Context:
{context}

Response:
{response}

Only output a number.
""".strip()

def build_fairness_prompt(context, response):
    return f"""
Evaluate FAIRNESS.

Definition:
Neutral tone, no assumptions, no bias, no fear-inducing wording.

Scores:
1.0–0.90 = Completely fair.
0.89–0.70 = Mostly fair.
0.69–0.40 = Some bias.
0.39–0.10 = Strong bias.
0.09–0.00 = Harmful or discriminatory.

Context:
{context}

Response:
{response}

Only output a number.
""".strip()

# ================== GROQ EVALUATORS ==================
def _groq_eval(prompt):
    try:
        resp = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=32
        )
        return resp.choices[0].message.content.strip()
    except:
        return "0.0"

async def _soft_score(prompt):
    text = await asyncio.to_thread(_groq_eval, prompt)
    nums = re.findall(r"\d*\.?\d+", text)
    try:
        return max(0.0, min(1.0, float(nums[-1])))
    except:
        return 0.0

# ================== MAIN EVALUATION ==================
async def evaluate_response_async(question, context, answer):
    context = _normalize(context)

    faith = await _soft_score(build_faithfulness_prompt(context, answer))
    relev = await _soft_score(build_relevance_prompt(context, answer))
    fair  = await _soft_score(build_fairness_prompt(context, answer))

    sim = 0.0
    if SIM_READY:
        try:
            emb1 = sim_model.encode(answer, convert_to_tensor=True)
            emb2 = sim_model.encode(context, convert_to_tensor=True)
            sim = float(util.cos_sim(emb1, emb2).item()) * 100
        except:
            sim = 0.0

    return {
        "faithfulness": round(faith, 2),
        "relevance": round(relev, 2),
        "fairness": round(fair, 2),
        "semantic_similarity": round(sim, 1)
    }

# ================== MAIN MCP TOOL ==================
@app.tool()
async def ask_with_combined_context(question: str):
    try:
        print(f"\n[REQUEST] {question}")

        # ---------- 1. PII SANITIZATION ----------
        pii_hits = detect_pii(question)
        print("DEBUG PII:", detect_pii(question))
        print("DEBUG REDACTED:", redact_pii(question))
        if PII_POLICY == "block" and pii_hits:
            return {
                "answer": "⚠️ Sensitive information detected. Please remove personal identifiers."
            }

        question_sanitized = redact_pii(question) if pii_hits else question
        print("[SANITIZED QUESTION]", question_sanitized)

        # ---------- 2. RETRIEVAL ----------
        async def query(collection, q, n):
            return await asyncio.to_thread(
                collection.query,
                query_texts=[q],
                n_results=n,
                include=["documents", "distances"]
            )

        df_res, pdf_res = await asyncio.gather(
            query(rag_collection, question_sanitized, 10),
            query(pdf_collection, question_sanitized, 10)
        )

        df_docs = [d for sub in df_res.get("documents", []) for d in sub]
        df_dist = df_res.get("distances", [[]])[0]

        pdf_docs = [d for sub in pdf_res.get("documents", []) for d in sub]
        pdf_dist = pdf_res.get("distances", [[]])[0]

        def pack(docs, dist):
            items = []
            for i, d in enumerate(docs):
                score = 1 - dist[i] if i < len(dist) else 0
                items.append({"text": d, "score": score})
            return items

        pairs = pack(df_docs, df_dist) + pack(pdf_docs, pdf_dist)

        if SIM_READY:
            q_emb = sim_model.encode(question_sanitized, convert_to_tensor=True)
            for p in pairs:
                emb = sim_model.encode(p["text"], convert_to_tensor=True)
                sim = float(util.cos_sim(q_emb, emb).item())
                p["score"] = (p["score"] + (sim + 1) / 2) / 2

        pairs.sort(key=lambda x: x["score"], reverse=True)

        context_docs = "\n\n".join(p["text"] for p in pairs[:10])

        # ---------- 3. PROMPT ----------
        prompt = f"""
You are a medical assistant.
Use ONLY the provided context.
If information is missing, say so.
Do NOT hallucinate.
Use subheadings.

CONTEXT:
{context_docs}

QUESTION:
{question_sanitized}

ANSWER:
""".strip()

        # ---------- 4. LLM GENERATION ----------
        answer = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=450
            ).choices[0].message.content.strip()
        )

        # ---------- 5. SANITIZE ANSWER ----------
        answer_sanitized = redact_pii(answer)
        print("DEBUG PII:", detect_pii(answer))
        print("DEBUG REDACTED:", redact_pii(answer))
        # ---------- 6. EVALUATION ----------
        evaluation = await evaluate_response_async(
            question_sanitized,
            context_docs,
            answer_sanitized
        )

        return {
            "answer": answer_sanitized,
            "evaluation": evaluation
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

# ================== ENTRY ==================
if __name__ == "__main__":
    app.run(transport="http", host="0.0.0.0", port=8000)
