import streamlit as st
import requests
import json

# --- Streamlit Page Config ---
st.set_page_config(page_title="RAG + MCP Medical Assistant", layout="centered")

st.title("Medical Assistant (RAG + MCP + Groq + Llama Evaluator)")
st.markdown(
    "Ask a medical question based on the loaded documents and MCP context. "
    "The model response is then evaluated for **faithfulness**, **relevance**, **fairness**, "
    "and **semantic similarity** using LlamaIndex evaluators."
)

# --- Input box ---
question = st.text_area(
    "Enter your question:",
    placeholder="e.g., What does a high oldpeak value indicate?",
    height=120,
)

API_URL = "http://127.0.0.1:8000/mcp"
BASE_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}

# --- Step 1: Initialize and get session ID ---
def mcp_initialize():
    """Initialize MCP and return session ID from response headers."""
    init_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "Streamlit MCP Client", "version": "1.0"},
        },
    }

    r = requests.post(API_URL, headers=BASE_HEADERS, json=init_payload, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Initialization failed: HTTP {r.status_code} - {r.text}")

    session_id = r.headers.get("Mcp-Session-Id")
    if not session_id:
        raise RuntimeError(f"No Mcp-Session-Id header found. Headers:\n{r.headers}")
    return session_id


# --- Step 2: Call tool with session header ---
def mcp_call_tool(session_id: str, question: str):
    """Call ask_with_combined_context tool (requires Mcp-Session-Id header)."""
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "ask_with_combined_context",
            "arguments": {"question": question},
        },
    }

    headers = BASE_HEADERS.copy()
    headers["Mcp-Session-Id"] = session_id

    r = requests.post(API_URL, headers=headers, json=payload, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"Tool call failed: HTTP {r.status_code} - {r.text}")

    # Try JSON decode first
    try:
        return r.json()
    except Exception:
        # Some MCP servers return event-stream; extract JSON objects manually
        messages = []
        for line in r.text.splitlines():
            if line.strip().startswith("data:"):
                try:
                    messages.append(json.loads(line.split("data:")[1].strip()))
                except Exception:
                    pass
        return messages[-1] if messages else {}


# --- UI Button Action ---
if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Contacting MCP server and generating response..."):
            try:
                # Step 1 — Initialize + get session
                session_id = mcp_initialize()

                # Step 2 — Call tool
                rpc = mcp_call_tool(session_id, question)

            except Exception as e:
                st.error(f"❌ MCP request failed:\n\n{e}")
                st.stop()

            # --- Extract result content ---
            if not rpc:
                st.error("⚠️ No response received from MCP server.")
                st.stop()

            if "error" in rpc:
                st.error(
                    f"⚠️ Server Error {rpc['error'].get('code')}: {rpc['error'].get('message')}"
                )
                st.stop()

            result = rpc.get("result", {})
            structured = result.get("structuredContent", {})

            # --- Fallback parsing for plain dict responses ---
            if not structured and isinstance(result, dict):
                structured = result

            # --- Extract fields ---
            answer = structured.get("answer", "")
            context_rel = structured.get("context_relevance")
            evaluation = structured.get("evaluation", {})

            # --- Display Answer ---
            if answer:
                st.success("### Model Response")
                st.markdown(answer, unsafe_allow_html=True)
            else:
                st.warning("No answer received from the model.")

            # --- Context Relevance ---
            col1, col2 = st.columns(2)
            if context_rel is not None:
                col1.metric("Context Relevance", f"{context_rel}%")

            # --- Evaluation Metrics ---
            if evaluation and any(v is not None for v in evaluation.values()):
                st.markdown("---")
                st.subheader("LlamaIndex Evaluation Metrics")

                def color_for_score(score):
                    if score is None:
                        return "gray"
                    elif score >= 85:
                        return "#1DB954"  # green
                    elif score >= 60:
                        return "#F1C40F"  # yellow
                    else:
                        return "#E74C3C"  # red

                def metric_bar(label, value):
                    if value is None:
                        st.write(f"**{label}:** N/A")
                    else:
                        color = color_for_score(value)
                        st.markdown(
                            f"<div style='margin-top:6px'><b>{label}:</b> "
                            f"{value}%<div style='background:{color};height:10px;width:{value}%;"
                            f"border-radius:6px;margin-top:3px'></div></div>",
                            unsafe_allow_html=True,
                        )

                with col1:
                    metric_bar("Faithfulness", evaluation.get("faithfulness"))
                    metric_bar("Relevance", evaluation.get("relevance"))
                with col2:
                    metric_bar("Fairness", evaluation.get("fairness"))
                    metric_bar("Semantic Similarity", evaluation.get("semantic_similarity"))

                st.info(f"**Summary:** {evaluation.get('summary', 'No evaluation summary available.')}")
                st.caption(f"Evaluator: {evaluation.get('evaluator', 'N/A')}")
            else:
                st.warning("No evaluation metrics returned or evaluation failed.")
