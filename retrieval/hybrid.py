import os
import threading
from typing import List, Dict, Any, Tuple

from rank_bm25 import BM25Okapi

from langchain_core.documents import Document

# Simple in-memory hybrid retriever. It maintains separate corpora for
# 'bis' and per-user 'user' documents (by thread_id). It must be updated
# whenever documents are ingested.


class HybridRetriever:

    def __init__(self, alpha: float = 0.7):
        self.alpha = float(alpha)
        self.lock = threading.Lock()

        # Bis corpus
        self.bis_texts: List[str] = []
        self.bis_metadatas: List[Dict[str, Any]] = []
        self.bis_bm25 = None

        # User corpora: thread_id -> texts, metadatas, bm25
        self.user_texts: Dict[str, List[str]] = {}
        self.user_metadatas: Dict[str, List[Dict[str, Any]]] = {}
        self.user_bm25: Dict[str, BM25Okapi] = {}

    def index_documents(self, docs: List[Document]):
        """Index documents into BM25 corpora. Documents must have
        metadata['knowledge_type'] either 'bis' or 'user'."""

        with self.lock:
            for doc in docs:
                mt = doc.metadata or {}
                kt = mt.get("knowledge_type", "bis")
                text = (doc.page_content or "").strip()
                if not text:
                    continue

                if kt == "bis":
                    self.bis_texts.append(text)
                    self.bis_metadatas.append(mt)
                elif kt == "user":
                    thread_id = mt.get("thread_id", "default")
                    self.user_texts.setdefault(thread_id, []).append(text)
                    self.user_metadatas.setdefault(thread_id, []).append(mt)

            # rebuild BM25
            if self.bis_texts:
                tokenized = [t.split() for t in self.bis_texts]
                self.bis_bm25 = BM25Okapi(tokenized)

            for tid, texts in self.user_texts.items():
                tokenized = [t.split() for t in texts]
                self.user_bm25[tid] = BM25Okapi(tokenized)

    def bm25_scores(self, query: str, knowledge_type: str = "bis", thread_id: str | None = None) -> List[Tuple[int, float]]:
        q_tokens = query.split()

        if knowledge_type == "bis":
            if not self.bis_bm25:
                return []
            scores = self.bis_bm25.get_scores(q_tokens)
            return list(enumerate(scores))

        # user
        tid = thread_id or "default"
        bm = self.user_bm25.get(tid)
        if not bm:
            return []
        scores = bm.get_scores(q_tokens)
        return list(enumerate(scores))

    def hybrid_search(self, vectorsearch_fn, query: str, knowledge_type: str = "bis", thread_id: str | None = None, vector_k: int = 15, top_k: int = 5, alpha: float | None = None):
        """
        vectorsearch_fn: callable(query, k, filter) -> List[Document]
        Returns top_k documents after combining semantic and lexical scores.
        """

        alpha = self.alpha if alpha is None else float(alpha)

        # Semantic candidates
        filters = {"knowledge_type": knowledge_type} if knowledge_type == "bis" else {"$and": [{"knowledge_type": "user"}, {"thread_id": thread_id or "default"}]}

        sem_docs = vectorsearch_fn(query, k=vector_k, filter=filters)

        if not sem_docs:
            return []

        # BM25 scores
        bm25_list = self.bm25_scores(query, knowledge_type=knowledge_type, thread_id=thread_id)
        # Build mapping index -> score
        bm25_map = {idx: score for idx, score in bm25_list}

        # Combine: semantic rank -> higher score for top positions
        combined = []

        for rank, doc in enumerate(sem_docs):
            metadata = doc.metadata or {}
            # attempt to find this doc's index in BM25 by metadata match
            idx = None
            if knowledge_type == "bis":
                # find first matching metadata index
                for i, m in enumerate(self.bis_metadatas):
                    if m.get("source") == metadata.get("source") and m.get("page") == metadata.get("page"):
                        idx = i
                        break
            else:
                tid = metadata.get("thread_id", "default")
                mets = self.user_metadatas.get(tid, [])
                for i, m in enumerate(mets):
                    if m.get("source") == metadata.get("source") and m.get("page") == metadata.get("page"):
                        idx = i
                        break

            sem_score = max(0.0, 1.0 - (rank / max(1, vector_k)))
            lex_score = 0.0
            if idx is not None:
                lex_score = bm25_map.get(idx, 0.0)

            combined_score = alpha * sem_score + (1 - alpha) * lex_score

            combined.append((combined_score, doc))

        # sort and return top_k
        combined.sort(key=lambda x: x[0], reverse=True)

        return combined[:top_k]
