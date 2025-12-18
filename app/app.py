import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from a .env file if present (project root)
load_dotenv()

# Support running both as a package (streamlit run - from project root)
# and as a plain script (streamlit run app/app.py) where absolute imports may fail
try:  # when executed with package context
    from app.ingest import ingest_file
    from app.rag_chain import answer_question, stream_answer
except Exception:  # fallback for direct script execution
    from ingest import ingest_file
    from rag_chain import answer_question, stream_answer

st.set_page_config(page_title="Document Chatbot (Ollama + LangChain)", page_icon="🤖", layout="wide")

st.title("Chat with your document")
st.caption("Powered by LangChain, Chroma, and Ollama (local)")

with st.sidebar:
    st.header("Settings")
    llm_model = st.text_input(
        "LLM model (ollama)",
        value=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        key="llm_model_input"
    )
    embed_model = st.text_input(
        "Embedding model (ollama)",
        value=os.getenv("EMBED_MODEL", "nomic-embed-text:latest"),
        key="embed_model_input"
    )
    st.markdown("""
    Tip: Ensure Ollama is running and you've pulled the models:
    - `ollama pull ` + the LLM model
    - `ollama pull ` + the Embedding model
    """)

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of (user, assistant)

uploaded = st.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx"], accept_multiple_files=False)

if uploaded is not None:
    # Save to a temp file
    suffix = "." + uploaded.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        temp_path = tmp.name

    os.environ["EMBED_MODEL"] = embed_model
    try:
        st.info("Indexing document. This may take a moment…")
        retriever = ingest_file(temp_path)
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

for user, assistant in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(user)
    with st.chat_message("assistant"):
        st.markdown(assistant)

prompt = st.chat_input("Ask a question about the document…", disabled=chat_disabled, key="main_chat_input")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Stream tokens as they arrive
            os.environ["OLLAMA_MODEL"] = llm_model
            stream = stream_answer(prompt, st.session_state.retriever, st.session_state.history, model=llm_model)
            response = st.write_stream(stream)
        except Exception as e:
            st.error(
                "LLM call failed. Ensure Ollama is running and the model is pulled.\n" + str(e)
            )
            response = ""

    st.session_state.history.append((prompt, response or ""))
