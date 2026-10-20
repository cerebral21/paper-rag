"""
Interactive single-query mode. For batch evaluation, use batch_query.py instead.
"""
import os
import warnings
import logging

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

from src.embeddings import load_vector_store
from src.retrieval import retrieve_hybrid
from src.generation import generate_answer


def query(question: str, vector_store):
    """Run a single query through the full hybrid retrieval + generation pipeline."""
    results = retrieve_hybrid(vector_store, question)

    print(f"\nQuery: {question}")
    print(f"Retrieved {len(results)} passages\n")

    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        print(f"--- Passage {i+1} (score: {score:.4f}) ---")
        print(f"Source: {source}, Page: {page}")
        print(doc.page_content[:300] + "...\n")

    answer = generate_answer(question, results)
    print(f"{'='*60}")
    print(f"ANSWER:\n{answer}")


if __name__ == "__main__":
    print("Loading vector store...")
    vector_store = load_vector_store()
    print("Ready.\n")

    while True:
        q = input("Enter your question (or 'quit' to exit): ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        query(q, vector_store)
