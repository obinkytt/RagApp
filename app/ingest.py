import os
import tempfile
import shutil
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")


def load_documents(file_path: str) -> List[Document]:
    """Load a PDF or DOCX file into LangChain Documents.

    Args:
        file_path: Absolute path to a .pdf or .docx file
    Returns:
        List[Document]
    Raises:
        ValueError: if extension unsupported
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or DOCX file.")

    return loader.load()


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    return splitter.split_documents(docs)


def build_vectorstore(docs: List[Document], embed_model: str | None = None):
    """Create a Chroma vector store from documents using Ollama embeddings.

    Tries to persist to disk when possible, but gracefully falls back to
    an in-memory (ephemeral) store if disk space is insufficient.
    Configure with env vars:
      - CHROMA_PERSIST_DIR: path to store index; if unset uses OS temp.
      - CHROMA_DISABLE_PERSIST: set to "1" or "true" to force in-memory.
      - MIN_DISK_FREE_MB: minimum free MB required to allow persistence (default 200).
    """
    model_name = embed_model or DEFAULT_EMBED_MODEL
    embeddings = OllamaEmbeddings(model=model_name)

    # Respect opt-out of persistence
    disable_persist = os.getenv("CHROMA_DISABLE_PERSIST", "0").lower() in {"1", "true", "yes"}

    # Resolve persist directory (env override -> temp default)
    persist_dir = os.getenv("CHROMA_PERSIST_DIR") or os.path.join(tempfile.gettempdir(), "rag_chroma")

    # Ensure directory exists if we plan to persist
    if not disable_persist:
        try:
            os.makedirs(persist_dir, exist_ok=True)
        except Exception:
            # If we cannot create the directory, fall back to in-memory
            disable_persist = True

    # Check free space threshold when persisting
    min_free_mb = int(os.getenv("MIN_DISK_FREE_MB", "200"))
    if not disable_persist:
        try:
            usage = shutil.disk_usage(persist_dir)
            free_mb = usage.free // (1024 * 1024)
            if free_mb < min_free_mb:
                disable_persist = True
        except Exception:
            # If disk usage check fails, be conservative and disable persistence
            disable_persist = True

    # Build the vector store with persistence if allowed, else in-memory
    try:
        if not disable_persist:
            vs = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persist_dir,
            )
        else:
            # Explicitly force ephemeral client to avoid touching disk
            eph_client = chromadb.EphemeralClient()
            vs = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                client=eph_client,
            )
        return vs
    except Exception as e:
        # If failure looks like disk-full, retry once in-memory with a clearer message
        msg = str(e).lower()
        if ("disk is full" in msg) or ("database or disk is full" in msg) or ("no space" in msg):
            eph_client = chromadb.EphemeralClient()
            vs = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                client=eph_client,
            )
            return vs
        raise


def ingest_file(file_path: str, embed_model: str | None = None):
    """Full ingestion: load -> split -> vectorize, returns a retriever."""
    docs = load_documents(file_path)
    chunks = split_documents(docs)
    vs = build_vectorstore(chunks, embed_model=embed_model)
    # Keep retrieval small and fast
    return vs.as_retriever(search_kwargs={"k": 4})