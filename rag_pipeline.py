"""
rag_pipeline.py Ingestion, Hybrid Search (ChromaDB vector + BM25), and RAG generation.
Compatible with Gemma models (no system_instruction support).
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# Constants 
COLLECTION_NAME = "gitlab_handbook"
EMBED_MODEL     = "all-MiniLM-L6-v2"
GEMINI_MODEL    = "gemma-3-27b-it"   # free tier: 30 RPM, 15K TPM — no system_instruction

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
RRF_K         = 60

MAX_RETRIES  = 4
BACKOFF_BASE = 20   # seconds; doubles each retry


# Helpers 
def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def chunk_markdown(text: str, source: str) -> List[dict]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE]
        if chunk.strip():
            chunks.append({"text": chunk, "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank + 1)


def _call_with_retry(fn, *args, **kwargs):
    """Call fn retrying on ResourceExhausted with exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except ResourceExhausted as e:
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            m = re.search(r"retry in ([0-9.]+)s", str(e))
            if m:
                wait = max(wait, float(m.group(1)) + 3)
            if attempt == MAX_RETRIES:
                raise
            print(f"[RAG] Rate limit- waiting {wait:.0f}s (attempt {attempt}/{MAX_RETRIES})…")
            time.sleep(wait)
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}") from e


# Pipeline 
class RAGPipeline:
    def __init__(self, chroma_path: str = "chroma_db", gitlab_path: str = "Gitlab"):
        self.chroma_path = chroma_path
        self.gitlab_path = Path(gitlab_path)

        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        self._client = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: List[dict] = []
        self._try_load_bm25()

    # BM25 
    def _try_load_bm25(self) -> None:
        try:
            total = self._collection.count()
            if total == 0:
                return
            batch, offset, corpus = 1000, 0, []
            while offset < total:
                result = self._collection.get(
                    limit=batch, offset=offset,
                    include=["documents", "metadatas"],
                )
                for doc, meta in zip(result["documents"], result["metadatas"]):
                    corpus.append({"text": doc, "source": meta.get("source", "unknown")})
                offset += batch
            self._build_bm25(corpus)
        except Exception:
            pass

    def _build_bm25(self, corpus: List[dict]) -> None:
        self._bm25_corpus = corpus
        self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in corpus])

    # Ingestion
    def ingest(self) -> int:
        md_files = list(self.gitlab_path.rglob("*.md"))
        if not md_files:
            raise FileNotFoundError(f"No markdown files found in '{self.gitlab_path}'.")

        all_chunks: List[dict] = []
        for fp in md_files:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            all_chunks.extend(chunk_markdown(text, fp.name))

        ids       = [f"chunk_{i}" for i in range(len(all_chunks))]
        documents = [c["text"]    for c in all_chunks]
        metadatas = [{"source": c["source"]} for c in all_chunks]

        for i in range(0, len(ids), 500):
            self._collection.upsert(
                ids=ids[i:i+500],
                documents=documents[i:i+500],
                metadatas=metadatas[i:i+500],
            )

        self._build_bm25(all_chunks)
        return len(all_chunks)

    # Hybrid Retrieval
    def retrieve(
        self,
        question: str,
        n_results: int = 5,
        relevance_threshold: float = 0.35,
    ) -> Tuple[List[str], List[str]]:
        fetch_n = max(n_results * 3, 15)

        # Vector search
        vec_results = self._collection.query(
            query_texts=[question],
            n_results=fetch_n,
            include=["documents", "metadatas", "distances"],
        )
        vec_docs      = vec_results["documents"][0]
        vec_metas     = vec_results["metadatas"][0]
        vec_distances = vec_results["distances"][0]

        vec_map = {
            doc: {"similarity": 1 - dist, "source": meta.get("source", "unknown")}
            for doc, meta, dist in zip(vec_docs, vec_metas, vec_distances)
        }

        # BM25 search
        bm25_map: dict[str, dict] = {}
        if self._bm25 and self._bm25_corpus:
            scores = self._bm25.get_scores(_tokenize(question))
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:fetch_n]
            for idx in top_idx:
                chunk = self._bm25_corpus[idx]
                bm25_map[chunk["text"]] = {
                    "bm25_score": float(scores[idx]),
                    "source": chunk["source"],
                }

        # RRF fusion
        rrf_scores: dict[str, float] = {}
        for rank, doc in enumerate(vec_docs):
            rrf_scores[doc] = rrf_scores.get(doc, 0.0) + _rrf_score(rank)
        bm25_ranked = sorted(bm25_map, key=lambda t: bm25_map[t]["bm25_score"], reverse=True)
        for rank, text in enumerate(bm25_ranked):
            rrf_scores[text] = rrf_scores.get(text, 0.0) + _rrf_score(rank)

        ranked = sorted(rrf_scores, key=lambda t: rrf_scores[t], reverse=True)

        context_chunks, sources, seen = [], [], set()
        for text in ranked:
            if len(context_chunks) >= n_results:
                break
            sim = vec_map.get(text, {}).get("similarity", relevance_threshold)
            if sim < relevance_threshold:
                continue
            context_chunks.append(text)
            src = (vec_map.get(text) or bm25_map.get(text, {})).get("source", "unknown")
            if src not in seen:
                sources.append(src)
                seen.add(src)

        return context_chunks, sources

    # Generation (Gemma-compatible: no system_instruction)
    def query(
        self,
        question: str,
        api_key: str,
        chat_history: List[dict] | None = None,
        n_results: int = 5,
        relevance_threshold: float = 0.35,
    ) -> Tuple[str, List[str]]:
        context_chunks, sources = self.retrieve(question, n_results, relevance_threshold)

        if not context_chunks:
            return (
                "I'm sorry, I couldn't find any relevant information in the "
                "GitLab Handbook for that question. "
                "Please try rephrasing, or the topic may not be covered in this dataset.",
                [],
            )

        context_text = "\n\n---\n\n".join(context_chunks)

        # Gemma has no system_instruction — prepend instructions as first user turn
        system_turn = (
            "You are a helpful assistant that answers questions ONLY about the GitLab Handbook. "
            "You will be given CONTEXT excerpts from the handbook in each message. "
            "Use ONLY that context to answer. "
            "If the answer is not in the context, say you don't know - never hallucinate. "
            "Be concise and factual. Confirm you understand with 'Understood.'"
        )

        # Build history: system turn first, then prior conversation
        gemini_history = [
            {"role": "user",  "parts": [system_turn]},
            {"role": "model", "parts": ["Understood."]},
        ]
        if chat_history:
            for turn in chat_history[-6:]:
                role = "user" if turn["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [turn["content"]]})

        current_turn = (
            f"CONTEXT (GitLab Handbook excerpts):\n"
            f"--- START ---\n{context_text}\n--- END ---\n\n"
            f"Question: {question}"
        )

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        chat  = model.start_chat(history=gemini_history)

        response = _call_with_retry(chat.send_message, current_turn)
        return response.text.strip(), sources
