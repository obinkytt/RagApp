# RAG Document Chatbot (Python + LangChain + Ollama)

Chat with your business documents (PDF/DOCX) locally using LangChain, Chroma vector store, and Ollama models.

## Features
- Upload a PDF or Word (.docx) document and chat about its content
- Local-first: runs with Ollama LLMs and embeddings on your machine
- Streamed responses, citation-grounded via retrieval

## Prerequisites
- Python 3.10+
- Ollama installed and running (Windows preview is supported)
  - Download: https://ollama.com/download
  - After install, start Ollama app/service
- Recommended models:
  - LLM: `llama3.1:8b` (or any model you prefer)
  - Embeddings: `nomic-embed-text`

Pull models once (PowerShell):

```powershell
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## Setup (Windows PowerShell)
1. Activate your virtual environment (already present in this repo):
   ```powershell
   .\rag\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Setup (Windows Command Prompt / cmd.exe)
1. Activate the virtual environment:
  ```bat
  rag\Scripts\activate.bat
  ```
2. Install dependencies:
  ```bat
  pip install -r requirements.txt
  ```

## Run the app
```powershell
# Optional: override models
$env:OLLAMA_MODEL = "llama3.1:8b"
$env:EMBED_MODEL = "nomic-embed-text"

# Launch Streamlit UI
streamlit run app/app.py
```

### Run the app (cmd.exe)
```bat
:: Optional: override models for this cmd session
set OLLAMA_MODEL=llama3.1:8b
set EMBED_MODEL=nomic-embed-text

:: Launch Streamlit UI
streamlit run app\app.py
```

If you prefer not to activate the venv, you can invoke Streamlit via the venv's Python directly:
```bat
rag\Scripts\python.exe -m streamlit run app\app.py
```

Then open the URL shown (typically http://localhost:8501), upload a document, and start chatting.

## Notes
- If you see connection errors, ensure the Ollama service is running and that the models are pulled.
- The vector store is kept in a temp directory per session and may be cleared on reboot.
- For larger docs, indexing can take some time. You can tweak chunk size and overlap via env vars `CHUNK_SIZE` and `CHUNK_OVERLAP`.
 - You can also put settings in a `.env` file in the project root (the app loads it automatically):
   ```
   OLLAMA_MODEL=llama3.2:latest
   EMBED_MODEL=nomic-embed-text
   ```

## Next steps
- Persist per-document indexes for reuse across sessions
- Add multi-file collections and document management
- Add citations (page numbers/snippets) to responses
- Optional FastAPI backend for enterprise deployment
