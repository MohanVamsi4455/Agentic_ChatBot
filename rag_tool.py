"""PDF RAG as a TOOL for the chatbot.

- The user uploads a PDF in a chat -> ingest_pdf() splits it, embeds it with
  Gemini and saves a FAISS index on disk for THAT chat (thread_id).
- The LLM calls the rag_search tool -> it searches only the current chat's PDFs.

On disk (one folder per chat):
    faiss_indexes/<thread_id>/index.faiss     the vectors
    faiss_indexes/<thread_id>/index.pkl       chunk texts + metadata
    faiss_indexes/<thread_id>/documents.json  names of the uploaded PDFs
"""
import json
import os
import shutil
import tempfile
import threading
import warnings

from dotenv import load_dotenv

warnings.filterwarnings("ignore", message=".*langchain-community.*")  # FAISS still lives here

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
INDEX_ROOT = os.path.join(DATA_DIR, "faiss_indexes")
os.makedirs(INDEX_ROOT, exist_ok=True)


def get_env_var(var_name: str, default: str | None = None) -> str | None:
    """Retrieve environment variable, checking os.environ then st.secrets if in Streamlit."""
    val = os.getenv(var_name)
    if not val:
        try:
            import streamlit as st
            if var_name in st.secrets:
                val = st.secrets[var_name]
        except Exception:
            pass
    return val or default


_embeddings_instance = None
_lock = threading.Lock()  # thread safety for uploads and index creation
_stores = {}              # thread_id -> loaded FAISS store (read disk once)


def get_embeddings():
    """Lazily load embeddings so missing keys at import time don't crash the app."""
    global _embeddings_instance
    if _embeddings_instance is None:
        key = get_env_var("GEMINI_API_KEY") or get_env_var("GOOGLE_API_KEY")
        _embeddings_instance = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", api_key=key
        )
    return _embeddings_instance


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def _thread_dir(thread_id: str) -> str:
    return os.path.join(INDEX_ROOT, thread_id)


def _docs_file(thread_id: str) -> str:
    return os.path.join(_thread_dir(thread_id), "documents.json")


def list_documents(thread_id: str) -> list[dict]:
    """PDFs uploaded in this chat: [{"name", "pages", "chunks"}, ...]."""
    try:
        with open(_docs_file(thread_id), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _load_store(thread_id: str):
    """This chat's FAISS store, or None if nothing has been uploaded."""
    if thread_id in _stores:
        return _stores[thread_id]
    if not os.path.exists(os.path.join(_thread_dir(thread_id), "index.faiss")):
        return None
    # index.pkl is a pickle; safe because only this app writes these files.
    store = FAISS.load_local(
        _thread_dir(thread_id),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    _stores[thread_id] = store
    return store


def ingest_pdf(file_bytes: bytes, filename: str, thread_id: str) -> dict:
    """Read a PDF, split it into chunks, embed them and add them to this chat's index."""
    # PyPDFLoader needs a file path, so write the upload to a temporary file.
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(file_bytes)
        tmp.close()
        pages = PyPDFLoader(tmp.name).load()
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

    pages = [p for p in pages if p.page_content.strip()]
    if not pages:
        raise ValueError(f"No text found in {filename}. Is it a scanned (image-only) PDF?")

    for p in pages:  # replace the temp path with the real name; make pages 1-based
        p.metadata = {"source": filename, "page": p.metadata.get("page", 0) + 1}
    chunks = splitter.split_documents(pages)

    with _lock:
        store = _load_store(thread_id)
        if store is None:
            store = FAISS.from_documents(chunks, get_embeddings())   # embeds every chunk
        else:
            store.add_documents(chunks)                              # 2nd, 3rd... PDF in this chat
        os.makedirs(_thread_dir(thread_id), exist_ok=True)
        store.save_local(_thread_dir(thread_id))
        _stores[thread_id] = store

        info = {"name": filename, "pages": len(pages), "chunks": len(chunks)}
        docs = [d for d in list_documents(thread_id) if d["name"] != filename] + [info]
        with open(_docs_file(thread_id), "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2)
    return info


def delete_documents(thread_id: str):
    """Remove this chat's index and document list (used when a chat is deleted)."""
    with _lock:
        _stores.pop(thread_id, None)
        shutil.rmtree(_thread_dir(thread_id), ignore_errors=True)


@tool
def rag_search(query: str, config: RunnableConfig) -> dict:
    """Search the PDF documents the user uploaded in this chat.
    Use this for ANY question about the user's uploaded document, PDF, file, report,
    resume, paper etc. (summaries, facts, numbers, sections, definitions).
    Write a specific search query. Returns the most relevant passages with
    the file name and page number so you can cite them."""
    # `config` is filled in by LangGraph (the model never sees it), so we know
    # which chat is asking and search only that chat's PDFs.
    thread_id = config.get("configurable", {}).get("thread_id")
    store = _load_store(thread_id) if thread_id else None
    if store is None:
        return {
            "error": "No document has been uploaded in this chat. "
                     "Ask the user to attach a PDF using the 📎 button in the message box."
        }

    results = store.similarity_search(query, k=4)
    return {
        "query": query,
        "results": [
            {
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "content": d.page_content,
            }
            for d in results
        ],
    }

