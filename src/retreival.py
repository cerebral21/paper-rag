import json
import os
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# ============================================================
# BM25 INDEX — built once at ingestion, loaded at query time
# ============================================================

CHUNK_STORE_PATH = "./chunk_store.json"


def save_chunks_for_bm25(chunks):
    """
    Save chunks to a JSON file during ingestion so BM25 can use them at query time.
    Call this in ingest.py right after filtering.
    """
    chunk_data = []
    for chunk in chunks:
        chunk_data.append({
            "content": chunk.page_content,
            "metadata": chunk.metadata,
        })
    with open(CHUNK_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, ensure_ascii=False)
    print(f"Saved {len(chunk_data)} chunks to {CHUNK_STORE_PATH} for BM25")


def _load_chunks_from_store():
    """Load saved chunks from JSON."""
    if not os.path.exists(CHUNK_STORE_PATH):
        raise FileNotFoundError(
            f"{CHUNK_STORE_PATH} not found. Run ingest.py first."
        )
    with open(CHUNK_STORE_PATH, "r", encoding="utf-8") as f:
        chunk_data = json.load(f)
    return chunk_data


def _tokenize(text):
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


# ============================================================
# RETRIEVAL METHODS
# ============================================================

def retrieve_simple(vector_store, query: str, k: int = 5):
    """Basic semantic retrieval — just cosine similarity. (Original method)"""
    results = vector_store.similarity_search_with_score(query, k=k)
    return results


def retrieve_semantic(vector_store, query: str, k: int = 20):
    """Semantic search returning top-k results."""
    results = vector_store.similarity_search_with_score(query, k=k)
    return results


def retrieve_bm25(query: str, k: int = 20):
    """
    BM25 keyword search over saved chunks.
    Returns list of (Document, score) tuples.
    """
    chunk_data = _load_chunks_from_store()
    corpus = [_tokenize(c["content"]) for c in chunk_data]
    bm25 = BM25Okapi(corpus)

    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            doc = Document(
                page_content=chunk_data[idx]["content"],
                metadata=chunk_data[idx]["metadata"],
            )
            results.append((doc, float(scores[idx])))

    return results


def reciprocal_rank_fusion(result_lists, k_rrf=60):
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = sum over all lists of 1 / (k + rank(d))

    k_rrf is a constant (default 60) that prevents top-ranked documents
    from dominating too heavily.

    Returns a single merged list of (Document, fused_score) sorted by
    fused score descending.
    """
    doc_scores = {}
    doc_objects = {}

    for results in result_lists:
        for rank, (doc, _original_score) in enumerate(results, start=1):
            # Use content + source as a unique key to handle duplicates
            key = hash(doc.page_content[:200] + str(doc.metadata.get("source", "")))

            if key not in doc_scores:
                doc_scores[key] = 0.0
                doc_objects[key] = doc

            doc_scores[key] += 1.0 / (k_rrf + rank)

    # Sort by fused score descending
    sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)

    fused_results = []
    for key in sorted_keys:
        fused_results.append((doc_objects[key], doc_scores[key]))

    return fused_results



_cross_encoder = None

def _get_cross_encoder():
    """Lazy-load the cross-encoder model (loaded once, reused)."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        # Print only on first load
        print("Cross-encoder loaded.")
    return _cross_encoder


def rerank_with_cross_encoder(query: str, results, top_k: int = 5):
    """
    Re-rank results using a cross-encoder that scores query-passage pairs together.

    The cross-encoder reads query + passage as a single input and produces a
    relevance score, which is more accurate than the independent bi-encoder
    embeddings used in semantic search.

    Takes top candidates from retrieval, re-scores them, returns top_k.
    """
    if not results:
        return results

    cross_encoder = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc, _score in results]
    scores = cross_encoder.predict(pairs)
    scored_results = list(zip([doc for doc, _ in results], scores))

    scored_results.sort(key=lambda x: x[1], reverse=True)

    # Return top_k as (Document, score) tuples
    return [(doc, float(score)) for doc, score in scored_results[:top_k]]


# ============================================================
# MAIN RETRIEVAL FUNCTION — combines everything
# ============================================================

def retrieve_hybrid(vector_store, query: str, max_results: int = 10, min_score: float = -2.0):
    """
    Full hybrid retrieval pipeline:
    1. Semantic search (top 20 candidates)
    2. BM25 keyword search (top 20 candidates)
    3. Reciprocal Rank Fusion to merge both lists
    4. Cross-encoder re-ranking on top 20 fused results
    5. Return top k results
    """
    # Step 1: Semantic search
    semantic_results = retrieve_semantic(vector_store, query, k=20)

    # Step 2: BM25 search
    bm25_results = retrieve_bm25(query, k=20)

    # Step 3: Fuse with RRF
    fused_results = reciprocal_rank_fusion(
        [semantic_results, bm25_results],
        k_rrf=60,
    )

    # Step 4: Re-rank top 20 fused results with cross-encoder
    top_candidates = fused_results[:20]
    reranked = rerank_with_cross_encoder(query, top_candidates, top_k=20)
    
    # Step 5: Score threshold is the only filter
    filtered = [(doc, score) for doc, score in reranked if score >= min_score]
    
    # Step 6: Cap at max_results to prevent overwhelming the LLM
    return filtered[:max_results]