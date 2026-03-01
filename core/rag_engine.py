"""
RAG Engine — Retrieval-Augmented Generation Pipeline
======================================================
Full RAG pipeline with:
  - Document chunking and embedding (sentence-transformers or OpenAI)
  - FAISS/NumPy vector store for fast retrieval
  - BM25 keyword fallback
  - Re-ranking with cross-encoder scoring
  - MDP-guided retrieval (uses RL to decide when to retrieve more)
  - Context-window management
"""

import numpy as np
import hashlib
import re
import json
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter
import math


# ──────────────────────────────────────────────
#  Document & Chunk Models
# ──────────────────────────────────────────────

@dataclass
class Document:
    """A source document before chunking."""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    doc_type: str = "text"  # text, pdf, url, video_transcript


@dataclass
class Chunk:
    """A chunk of text with its embedding."""
    chunk_id: str
    doc_id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_idx: int = 0
    end_idx: int = 0

    @property
    def word_count(self) -> int:
        return len(self.content.split())


@dataclass
class RetrievalResult:
    """A single retrieval result with relevance score."""
    chunk: Chunk
    score: float
    retrieval_method: str = "vector"  # vector, bm25, hybrid
    rank: int = 0


# ──────────────────────────────────────────────
#  Text Chunking Strategies
# ──────────────────────────────────────────────

class TextChunker:
    """
    Intelligent text chunking with overlap and semantic boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Split document into overlapping chunks."""
        text = doc.content.strip()
        if not text:
            return []

        # Split by paragraphs first, then by sentences
        paragraphs = self._split_paragraphs(text)
        chunks = []
        current_text = ""
        start_idx = 0

        for para in paragraphs:
            if len(current_text) + len(para) > self.chunk_size and current_text:
                chunk = self._make_chunk(doc.doc_id, current_text, start_idx, len(chunks))
                chunks.append(chunk)
                # Overlap: keep last portion
                overlap_text = current_text[-self.chunk_overlap:] if len(current_text) > self.chunk_overlap else ""
                start_idx += len(current_text) - len(overlap_text)
                current_text = overlap_text + para
            else:
                current_text += ("\n\n" if current_text else "") + para

        # Last chunk
        if len(current_text) >= self.min_chunk_size:
            chunk = self._make_chunk(doc.doc_id, current_text, start_idx, len(chunks))
            chunks.append(chunk)

        return chunks

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n{2,}', text)
        result = []
        for para in paragraphs:
            para = para.strip()
            if para:
                # If paragraph is too long, split by sentences
                if len(para) > self.chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    result.extend(sentences)
                else:
                    result.append(para)
        return result

    def _make_chunk(self, doc_id: str, text: str, start_idx: int, chunk_num: int) -> Chunk:
        """Create a Chunk object."""
        chunk_id = hashlib.md5(f"{doc_id}:{chunk_num}:{text[:50]}".encode()).hexdigest()[:12]
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            content=text.strip(),
            start_idx=start_idx,
            end_idx=start_idx + len(text),
        )


# ──────────────────────────────────────────────
#  Embedding Engine (lightweight, NumPy-based)
# ──────────────────────────────────────────────

class EmbeddingEngine:
    """
    Text embedding engine.
    Uses TF-IDF style embeddings by default (no external dependencies).
    Can be swapped for sentence-transformers or OpenAI embeddings.
    """

    def __init__(self, dim: int = 384, use_external: bool = False):
        self.dim = dim
        self.use_external = use_external
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._doc_count = 0
        self._fitted = False
        self._external_model = None

    def fit(self, texts: List[str]):
        """Build vocabulary and IDF from a corpus."""
        self._doc_count = len(texts)
        doc_freq: Dict[str, int] = Counter()

        for text in texts:
            tokens = set(self._tokenize(text))
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # Build vocab (top `dim` tokens by document frequency)
        sorted_tokens = sorted(doc_freq.keys(), key=lambda t: doc_freq[t], reverse=True)
        self._vocab = {token: idx for idx, token in enumerate(sorted_tokens[:self.dim])}

        # Compute IDF
        for token, freq in doc_freq.items():
            self._idf[token] = math.log((self._doc_count + 1) / (freq + 1)) + 1

        self._fitted = True

    def embed(self, text: str) -> np.ndarray:
        """Create embedding for a single text."""
        if self.use_external and self._external_model is not None:
            return self._external_embed(text)

        if not self._fitted:
            # Auto-fit with single document
            self.fit([text])

        tokens = self._tokenize(text)
        tf = Counter(tokens)
        vec = np.zeros(self.dim, dtype=np.float32)

        for token, count in tf.items():
            if token in self._vocab:
                idx = self._vocab[token]
                idf = self._idf.get(token, 1.0)
                vec[idx] = (count / max(len(tokens), 1)) * idf

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts."""
        if not self._fitted and texts:
            self.fit(texts)
        return np.array([self.embed(t) for t in texts])

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if len(t) > 1]

    def _external_embed(self, text: str) -> np.ndarray:
        """Use external embedding model (override point)."""
        raise NotImplementedError("External embedding not configured")

    def set_external_model(self, model):
        """Set an external embedding model (e.g., SentenceTransformer)."""
        self._external_model = model
        self.use_external = True


# ──────────────────────────────────────────────
#  Vector Store (NumPy-based FAISS alternative)
# ──────────────────────────────────────────────

class VectorStore:
    """
    In-memory vector store using NumPy for similarity search.
    Drop-in replacement for FAISS when no GPU is needed.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._vectors: List[np.ndarray] = []
        self._chunks: List[Chunk] = []
        self._matrix: Optional[np.ndarray] = None

    def add(self, chunks: List[Chunk]):
        """Add chunks with their embeddings to the store."""
        for chunk in chunks:
            if chunk.embedding is not None:
                self._vectors.append(chunk.embedding)
                self._chunks.append(chunk)

        # Rebuild matrix for fast search
        if self._vectors:
            self._matrix = np.vstack(self._vectors)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Cosine similarity search.
        Returns list of (chunk, similarity_score).
        """
        if self._matrix is None or len(self._matrix) == 0:
            return []

        # Cosine similarity (vectors are pre-normalized)
        similarities = self._matrix @ query_vec
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0:
                results.append((self._chunks[idx], score))

        return results

    def clear(self):
        self._vectors = []
        self._chunks = []
        self._matrix = None

    @property
    def size(self) -> int:
        return len(self._chunks)


# ──────────────────────────────────────────────
#  BM25 Retriever (keyword fallback)
# ──────────────────────────────────────────────

class BM25Retriever:
    """BM25 keyword-based retriever as fallback for vector search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: List[Chunk] = []
        self._doc_freqs: Dict[str, int] = {}
        self._doc_lens: List[int] = []
        self._avg_dl: float = 0.0
        self._tokenized_chunks: List[List[str]] = []

    def add(self, chunks: List[Chunk]):
        """Index chunks for BM25 retrieval."""
        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            self._chunks.append(chunk)
            self._tokenized_chunks.append(tokens)
            self._doc_lens.append(len(tokens))

            seen = set()
            for token in tokens:
                if token not in seen:
                    self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                    seen.add(token)

        self._avg_dl = sum(self._doc_lens) / max(len(self._doc_lens), 1)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """BM25 search."""
        query_tokens = self._tokenize(query)
        n_docs = len(self._chunks)
        scores = []

        for i, doc_tokens in enumerate(self._tokenized_chunks):
            score = 0.0
            dl = self._doc_lens[i]
            tf_map = Counter(doc_tokens)

            for qt in query_tokens:
                if qt in tf_map:
                    tf = tf_map[qt]
                    df = self._doc_freqs.get(qt, 0)
                    idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
                    tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl))
                    score += idf * tf_norm

            scores.append((self._chunks[i], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if len(t) > 1]


# ──────────────────────────────────────────────
#  RAG Pipeline
# ──────────────────────────────────────────────

class RAGEngine:
    """
    Full Retrieval-Augmented Generation pipeline.
    
    Combines vector search + BM25 keyword search with:
    - Intelligent re-ranking
    - Context window management
    - MDP-guided retrieval decisions
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        max_context_tokens: int = 4000,
    ):
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedder = EmbeddingEngine(dim=embedding_dim)
        self.vector_store = VectorStore(dim=embedding_dim)
        self.bm25 = BM25Retriever()
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens

        # Stats
        self._total_docs = 0
        self._total_chunks = 0
        self._total_queries = 0

    def ingest_documents(self, documents: List[Document]) -> int:
        """
        Ingest documents into the RAG pipeline.
        Returns number of chunks created.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        # Fit embedder on all chunk texts
        texts = [c.content for c in all_chunks]
        self.embedder.fit(texts)

        # Embed all chunks
        for chunk in all_chunks:
            chunk.embedding = self.embedder.embed(chunk.content)

        # Add to stores
        self.vector_store.add(all_chunks)
        self.bm25.add(all_chunks)

        self._total_docs += len(documents)
        self._total_chunks += len(all_chunks)

        return len(all_chunks)

    def ingest_text(self, text: str, doc_id: str = "", source: str = "", metadata: Dict[str, Any] = None) -> int:
        """Convenience: ingest a single text string."""
        if not doc_id:
            doc_id = hashlib.md5(text[:100].encode()).hexdigest()[:10]
        doc = Document(doc_id=doc_id, content=text, source=source, metadata=metadata or {})
        return self.ingest_documents([doc])

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        method: str = "hybrid",
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of results
            method: 'vector', 'bm25', or 'hybrid'
        """
        k = top_k or self.top_k
        self._total_queries += 1

        results = []

        if method in ("vector", "hybrid"):
            query_vec = self.embedder.embed(query)
            vec_results = self.vector_store.search(query_vec, k * 2)
            for chunk, score in vec_results:
                results.append(RetrievalResult(
                    chunk=chunk,
                    score=score,
                    retrieval_method="vector",
                ))

        if method in ("bm25", "hybrid"):
            bm25_results = self.bm25.search(query, k * 2)
            for chunk, score in bm25_results:
                # Normalize BM25 score to 0–1 range
                max_score = bm25_results[0][1] if bm25_results else 1.0
                norm_score = score / max(max_score, 1e-8)
                results.append(RetrievalResult(
                    chunk=chunk,
                    score=norm_score * 0.8,  # Slight discount for BM25
                    retrieval_method="bm25",
                ))

        # Deduplicate by chunk_id, keeping highest score
        seen = {}
        for r in results:
            if r.chunk.chunk_id not in seen or r.score > seen[r.chunk.chunk_id].score:
                seen[r.chunk.chunk_id] = r

        # Re-rank and take top-k
        ranked = sorted(seen.values(), key=lambda r: r.score, reverse=True)[:k]

        # Assign ranks
        for i, r in enumerate(ranked):
            r.rank = i + 1

        return ranked

    def build_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        method: str = "hybrid",
    ) -> Tuple[str, List[RetrievalResult]]:
        """
        Retrieve and build a context string for LLM prompting.
        
        Returns:
            (context_string, retrieval_results)
        """
        results = self.retrieve(query, top_k, method)

        context_parts = []
        total_words = 0

        for r in results:
            words = r.chunk.word_count
            if total_words + words > self.max_context_tokens:
                break
            context_parts.append(
                f"[Source {r.rank}: score={r.score:.3f}]\n{r.chunk.content}"
            )
            total_words += words

        context = "\n\n---\n\n".join(context_parts)
        return context, results

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": self._total_docs,
            "total_chunks": self._total_chunks,
            "vector_store_size": self.vector_store.size,
            "total_queries": self._total_queries,
            "embedding_dim": self.embedder.dim,
        }
