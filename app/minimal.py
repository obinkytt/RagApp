import os
import tempfile
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import json
from urllib import request as _urlreq
from urllib.error import URLError
import urllib.parse

# Load environment variables
load_dotenv()

# Import functions - simplified approach
try:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from ingest import ingest_file
    from rag_chain import stream_answer, answer_question, make_llm
except ImportError as e:
    st.error(f"Could not import required modules: {e}")
    st.stop()

# Page config - must be first Streamlit command
st.set_page_config(page_title="Document Chatbot", page_icon="🤖", layout="wide")

st.title("Chat with your document")
st.caption("Powered by LangChain, Chroma, and Ollama (local)")

def list_ollama_models() -> list[str]:
    """Return installed Ollama model names via local API; empty list if unavailable."""
    try:
        with _urlreq.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.5) as resp:
            data = json.load(resp)
            models = [m.get("name", "") for m in data.get("models", [])]
            return [m for m in models if m]
    except Exception:
        return []

def filter_embedding_models(models: list[str]) -> list[str]:
    """Heuristically filter embedding-capable models from Ollama tags."""
    if not models:
        return []
    needles = (
        "embed",
        "e5",
        "bge",
        "gte",
        "text-embedding",
        "all-minilm",
        "mxbai",
        "nomic",
        "snowflake",
        "minilm",
    )
    out: list[str] = []
    for m in models:
        mm = m.lower()
        if any(n in mm for n in needles):
            out.append(m)
    # prefer common embeddings at the top if present
    order_hint = [
        "nomic-embed-text:latest",
        "mxbai-embed-large",
        "snowflake-arctic-embed",
        "bge-m3",
    ]
    out_sorted = sorted(out, key=lambda x: (order_hint.index(x) if x in order_hint else 999, x))
    return out_sorted

def is_installed(model: str, installed: list[str]) -> bool:
    try:
        return model in installed
    except Exception:
        return False

def pull_ollama_model(model: str) -> tuple[bool, str]:
    """Attempt to pull a model via Ollama API. Returns (ok, message)."""
    try:
        url = "http://127.0.0.1:11434/api/pull"
        payload = json.dumps({"name": model}).encode("utf-8")
        req = _urlreq.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with _urlreq.urlopen(req, timeout=60) as resp:
            # The pull API streams JSON lines; we read all and show the last status
            body = resp.read().decode("utf-8", errors="ignore")
        # Try to parse last non-empty JSON line
        last_line = ""
        for line in body.strip().splitlines():
            if line.strip():
                last_line = line
        if last_line:
            try:
                obj = json.loads(last_line)
                status = obj.get("status") or obj.get("error") or "pulled"
                if "error" in obj:
                    return False, str(status)
                return True, str(status)
            except Exception:
                pass
        return True, "Pull request sent"
    except URLError as e:
        return False, f"Could not reach Ollama API: {e}"
    except Exception as e:
        return False, str(e)

# Settings in sidebar
with st.sidebar:
    st.header("Settings")
    installed_models = list_ollama_models()
    # Per-model preset memory
    if "preset_by_model" not in st.session_state:
        st.session_state.preset_by_model = {}
    if "last_llm" not in st.session_state:
        st.session_state.last_llm = None
    if installed_models:
        default_model = "llama3.1:8b"
        try:
            default_index = installed_models.index(default_model)
        except ValueError:
            default_index = 0
        llm_model = st.selectbox(
            "LLM model (ollama)",
            options=installed_models,
            index=default_index,
            help="Models detected from your local Ollama."
        )
        # When model changes, load saved preset if any
        if st.session_state.last_llm != llm_model:
            saved = st.session_state.preset_by_model.get(llm_model)
            st.session_state.last_llm = llm_model
            if saved:
                st.session_state["minimal_k"] = saved.get("k", st.session_state.get("minimal_k", 4))
                st.session_state["minimal_max_new_tokens"] = saved.get("max_new_tokens", st.session_state.get("minimal_max_new_tokens", 128))
                st.session_state["minimal_max_ctx"] = saved.get("max_ctx", st.session_state.get("minimal_max_ctx", 4000))
                st.session_state["minimal_temperature"] = saved.get("temperature", st.session_state.get("minimal_temperature", 0.3))
                st.toast(f"Loaded saved preset for {llm_model}")
                st.rerun()
        if not is_installed(llm_model, installed_models):
            st.warning(f"Model '{llm_model}' is not installed.")
            if st.button("Pull LLM model", use_container_width=True):
                with st.status(f"Pulling {llm_model}…", expanded=True):
                    ok, msg = pull_ollama_model(llm_model)
                    st.write(msg)
                if ok:
                    st.success("Pull complete.")
                    st.rerun()
    else:
        llm_model = st.text_input(
            "LLM model (ollama)",
            value="llama3.1:8b",
            key="minimal_llm_model"
        )
    # Embedding model selector
    embedding_options = filter_embedding_models(installed_models)
    if embedding_options:
        default_embed = "nomic-embed-text:latest"
        try:
            default_embed_idx = embedding_options.index(default_embed)
        except ValueError:
            default_embed_idx = 0
        embed_model = st.selectbox(
            "Embedding model (ollama)",
            options=embedding_options,
            index=default_embed_idx,
            help="Embedding models detected from your local Ollama.",
            key="minimal_embed_model_select",
        )
        if not is_installed(embed_model, installed_models):
            st.warning(f"Embedding model '{embed_model}' is not installed.")
            if st.button("Pull embedding model", use_container_width=True):
                with st.status(f"Pulling {embed_model}…", expanded=True):
                    ok, msg = pull_ollama_model(embed_model)
                    st.write(msg)
                if ok:
                    st.success("Pull complete.")
                    st.rerun()
    else:
        embed_model = st.text_input(
            "Embedding model (ollama)", 
            value="nomic-embed-text:latest",
            key="minimal_embed_model"
        )
    stream_mode = st.checkbox("Stream responses", value=False, help="Turn on token streaming (may appear slower on some systems)")
    # Sliders with persistent keys so presets can edit them
    top_k = st.slider(
        "Top chunks (k)", min_value=1, max_value=8, value=st.session_state.get("minimal_k", 4),
        key="minimal_k", help="How many chunks to retrieve for context"
    )
    max_new_tokens = st.slider(
        "Max new tokens", min_value=64, max_value=512, value=st.session_state.get("minimal_max_new_tokens", 128), step=64,
        key="minimal_max_new_tokens", help="Lower is faster"
    )
    max_ctx_chars = st.slider(
        "Max context size (chars)", min_value=1000, max_value=10000, value=st.session_state.get("minimal_max_ctx", 4000), step=500,
        key="minimal_max_ctx", help="Trim retrieved context to this many characters"
    )
    # Presets
    preset = st.selectbox("Preset", ["Speed", "Balanced", "Quality"], index=1, help="Quickly set sensible speed/quality tradeoffs.")
    if st.button("Apply preset"):
        if preset == "Speed":
            st.session_state["minimal_k"] = 2
            st.session_state["minimal_max_new_tokens"] = 96
            st.session_state["minimal_max_ctx"] = 2500
            st.session_state["minimal_temperature"] = 0.15
        elif preset == "Quality":
            st.session_state["minimal_k"] = 6
            st.session_state["minimal_max_new_tokens"] = 256
            st.session_state["minimal_max_ctx"] = 8000
            st.session_state["minimal_temperature"] = 0.6
        else:  # Balanced
            st.session_state["minimal_k"] = 4
            st.session_state["minimal_max_new_tokens"] = 128
            st.session_state["minimal_max_ctx"] = 4000
            st.session_state["minimal_temperature"] = 0.3
        # Save per-model
        st.session_state.preset_by_model[llm_model] = {
            "k": st.session_state["minimal_k"],
            "max_new_tokens": st.session_state["minimal_max_new_tokens"],
            "max_ctx": st.session_state["minimal_max_ctx"],
            "temperature": st.session_state["minimal_temperature"],
        }
        st.toast(f"Applied {preset} preset for {llm_model}")
        st.rerun()

    # Quick pull by name (generic)
    with st.expander("Pull a model by name"):
        to_pull = st.text_input("Model name to pull", placeholder="e.g. llama3.1:8b or nomic-embed-text:latest", key="minimal_pull_name")
        if st.button("Pull model", key="minimal_pull_button"):
            if not to_pull:
                st.warning("Enter a model name to pull.")
            else:
                with st.status(f"Pulling {to_pull}…", expanded=True):
                    ok, msg = pull_ollama_model(to_pull)
                    st.write(msg)
                if ok:
                    st.success("Pull complete.")
                    st.rerun()
    # Debug controls
    debug_enable = st.checkbox("Show retrieval debug", value=False, help="Display retrieved chunk count and a small preview.")
    # Quick LLM sanity check
    if st.button("Test LLM response", help="Send a tiny ping to the selected model to verify it's responding"):
        try:
            os.environ["OLLAMA_MODEL"] = llm_model
            llm = make_llm(llm_model)
            resp = llm.invoke([{"role": "user", "content": "Say 'ready'"}])
            st.success(f"LLM responded: {getattr(resp, 'content', str(resp))[:200]}")
        except Exception as e:
            st.error(f"LLM test failed: {e}")

# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "history" not in st.session_state:
    st.session_state.history = []

# File upload - ONLY ONE
uploaded = st.file_uploader(
    "Upload a PDF or DOCX", 
    type=["pdf", "docx"], 
    accept_multiple_files=False, 
    key="minimal_file_uploader"
)

if uploaded is not None:
    # Save to temp file
    suffix = "." + uploaded.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        temp_path = tmp.name

    os.environ["EMBED_MODEL"] = embed_model
    try:
        st.info("Indexing document. This may take a moment…")
        retriever = ingest_file(temp_path, embed_model=embed_model)
        st.session_state.retriever = retriever
        st.success("Document indexed. Start chatting below.")
    except Exception as e:
        st.error(f"Failed to index: {e}")
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass

chat_disabled = st.session_state.retriever is None

st.divider()

# Display chat history
for user, assistant in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(user)
    with st.chat_message("assistant"):
        st.markdown(assistant)
    
# Chat input
prompt = st.chat_input(
    "Ask a question about the document…", 
    disabled=chat_disabled,
    key="minimal_chat_input"
)

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            os.environ["OLLAMA_MODEL"] = llm_model
            # Apply performance settings before LLM call
            os.environ["OLLAMA_NUM_PREDICT"] = str(max_new_tokens)
            os.environ["MAX_CONTEXT_CHARS"] = str(max_ctx_chars)
            if "minimal_temperature" in st.session_state:
                os.environ["TEMPERATURE"] = str(st.session_state["minimal_temperature"])
            # Update retriever k without re-indexing
            try:
                if st.session_state.retriever and hasattr(st.session_state.retriever, "search_kwargs"):
                    st.session_state.retriever.search_kwargs["k"] = top_k
            except Exception:
                pass
            # Quick installed check before call
            installed_now = list_ollama_models()
            if llm_model not in installed_now:
                st.warning(f"Model '{llm_model}' not installed or still pulling.")
                response = f"Model '{llm_model}' not available yet. Pull it or wait for completion, then try again."
            else:
                # Pre-retrieve docs for debug visibility
                docs_preview = []
                if st.session_state.retriever:
                    try:
                        # Try both invoke styles manually
                        if hasattr(st.session_state.retriever, "invoke"):
                            try:
                                docs_preview = st.session_state.retriever.invoke(prompt)
                                if not docs_preview:
                                    docs_preview = st.session_state.retriever.invoke({"query": prompt})
                            except Exception:
                                docs_preview = st.session_state.retriever.invoke({"query": prompt})
                        elif hasattr(st.session_state.retriever, "get_relevant_documents"):
                            docs_preview = st.session_state.retriever.get_relevant_documents(prompt)
                    except Exception:
                        docs_preview = []
                if debug_enable:
                    count = len(docs_preview)
                    st.caption(f"Retrieved {count} chunk(s).")
                    if docs_preview:
                        first_text = getattr(docs_preview[0], "page_content", "")[:400]
                        st.code(first_text or "[empty chunk]", language="text")
                if stream_mode:
                    stream = stream_answer(prompt, st.session_state.retriever, st.session_state.history, model=llm_model, pre_retrieved_docs=docs_preview)
                    response = st.write_stream(stream)
                    if not response:
                        response = answer_question(prompt, st.session_state.retriever, st.session_state.history, model=llm_model, pre_retrieved_docs=docs_preview)
                else:
                    with st.spinner("Thinking..."):
                        response = answer_question(prompt, st.session_state.retriever, st.session_state.history, model=llm_model, pre_retrieved_docs=docs_preview)
            # Final blank safeguard
            if not response or not str(response).strip():
                response = "[No output received from model. It may still be initializing or downloading. Please try again shortly.]"
            # Store meta for debug
            context_length = sum(len(getattr(d, "page_content", "")) for d in docs_preview) if docs_preview else 0
            st.session_state.last_response_meta = {
                "model": llm_model,
                "embed_model": embed_model,
                "top_k": top_k,
                "chars_context_cap": max_ctx_chars,
                "stream_mode": stream_mode,
                "response_len": len(str(response)),
                "retrieved_chunks": len(docs_preview),
                "context_length": context_length,
            }
        except Exception as e:
            st.error("LLM call failed. Ensure Ollama is running and the model is pulled.\n" + str(e))
            response = ""

    st.session_state.history.append((prompt, response or ""))