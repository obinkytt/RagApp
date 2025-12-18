import os
from typing import List, Tuple, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

DEFAULT_LLM = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
DEFAULT_TEMP = float(os.getenv("TEMPERATURE", "0.1"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "4000"))

SYSTEM_PROMPT = (
    "You are a helpful business assistant. Answer the user's questions using only the provided context. "
    "If the answer is not in the context, say you don't know. Be concise and precise."
)

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT + "\n\nContext:\n{context}"),
        ("human", "{question}\n\nConversation so far:\n{history}"),
    ]
)

def _retrieve_docs(retriever, question: str):
    """Robustly get documents from a retriever supporting either invoke() or get_relevant_documents()."""
    if retriever is None:
        return []
    try:
        if hasattr(retriever, "invoke"):
            # Some retrievers expect a dict {"query": question}
            try:
                docs = retriever.invoke(question)
                if not docs:
                    raise ValueError("empty from string invoke")
            except Exception:
                docs = retriever.invoke({"query": question})
        elif hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(question)
        else:
            docs = []
    except Exception:
        # Fallback to get_relevant_documents if invoke failed
        try:
            docs = retriever.get_relevant_documents(question)
        except Exception:
            docs = []
    return docs or []


def format_history(history: List[Tuple[str, str]]) -> str:
    # history is a list of (user, assistant) tuples
    out = []
    for u, a in history:
        out.append(f"User: {u}")
        out.append(f"Assistant: {a}")
    return "\n".join(out)


def make_llm(model: str | None = None) -> ChatOllama:
    # Build ChatOllama with optional performance knobs from env
    kwargs: dict = {}
    num_predict = os.getenv("OLLAMA_NUM_PREDICT")
    num_ctx = os.getenv("OLLAMA_NUM_CTX")
    if num_predict:
        try:
            kwargs["num_predict"] = int(num_predict)
        except ValueError:
            pass
    if num_ctx:
        try:
            kwargs["num_ctx"] = int(num_ctx)
        except ValueError:
            pass
    temperature = DEFAULT_TEMP
    try:
        temperature = float(os.getenv("TEMPERATURE", str(DEFAULT_TEMP)))
    except ValueError:
        pass

    return ChatOllama(model=model or DEFAULT_LLM, temperature=temperature, **kwargs)


def answer_question(
    question: str,
    retriever,
    history: Optional[List[Tuple[str, str]]] = None,
    model: Optional[str] = None,
    pre_retrieved_docs: Optional[List] = None,
) -> str:
    """Retrieve top context and query Ollama with a grounded prompt.

    pre_retrieved_docs: optional list of docs passed in by caller (to avoid duplicate retrieval).
    """
    history = history or []
    docs = pre_retrieved_docs if pre_retrieved_docs is not None else _retrieve_docs(retriever, question)
    if not docs:
        return "I couldn't find relevant context in the indexed document. Try rephrasing or asking about a specific section."
    context = "\n\n".join(getattr(d, "page_content", "") for d in docs)
    if MAX_CONTEXT_CHARS and len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    system_content = SYSTEM_PROMPT + "\n\nContext:\n" + context
    human_content = question + "\n\nConversation so far:\n" + format_history(history)

    llm = make_llm(model)
    messages = [SystemMessage(content=system_content), HumanMessage(content=human_content)]
    try:
        resp = llm.invoke(messages)
    except Exception as e:
        return f"[LLM invoke error] {e}"
    content = getattr(resp, "content", "") if resp is not None else ""
    if not content.strip():
        return "[No output received from model. Try again in a few seconds while the model initializes.]"
    return content


def stream_answer(
    question: str,
    retriever,
    history: Optional[List[Tuple[str, str]]] = None,
    model: Optional[str] = None,
    pre_retrieved_docs: Optional[List] = None,
):
    """Generator yielding tokens for streaming UIs (e.g., Streamlit)."""
    history = history or []
    docs = pre_retrieved_docs if pre_retrieved_docs is not None else _retrieve_docs(retriever, question)
    if not docs:
        yield "I couldn't find relevant context in the indexed document. Try rephrasing or asking about a specific section."
        return
    context = "\n\n".join(getattr(d, "page_content", "") for d in docs)
    if MAX_CONTEXT_CHARS and len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    debug = os.getenv("RAG_DEBUG")
    preview = "\n\n[Context preview omitted]"
    if debug:
        joined = context[:800]
        preview = "\n\n[Context preview]\n" + joined

    system_content = SYSTEM_PROMPT + "\n\nContext:\n" + context + (preview if debug else "")
    human_content = question + "\n\nConversation so far:\n" + format_history(history)

    llm = make_llm(model)
    messages = [SystemMessage(content=system_content), HumanMessage(content=human_content)]
    emitted_any = False
    try:
        for chunk in llm.stream(messages):
            if isinstance(chunk, str):
                text = chunk
            else:
                text = getattr(chunk, "content", None)
                if not text and hasattr(chunk, "message"):
                    text = getattr(chunk.message, "content", "")
            text = text or ""
            if text:
                emitted_any = True
            yield text
    except Exception as e:
        yield f"[Stream error] {e}"
        return

    if not emitted_any:
        try:
            resp = llm.invoke(messages)
            fallback = getattr(resp, "content", str(resp))
            if not fallback.strip():
                yield "[No output received from model. Try again shortly.]"
            else:
                yield fallback
        except Exception as e:
            yield f"[Invoke error] {e}"