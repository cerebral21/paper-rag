import os
import warnings
import logging
import re
from datetime import datetime

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

from src.embeddings import load_vector_store
from src.retreival import retrieve_hybrid
from src.generation import generate_answer


QUESTIONS_FILE = "questions.txt"
OUTPUT_DIR = "results"


def slugify(text, max_len=50):
    """Convert question text to a safe filename."""
    slug = re.sub(r'[^\w\s-]', '', text.lower().strip())
    slug = re.sub(r'[\s]+', '_', slug)
    return slug[:max_len]


def format_result(query, results, answer, query_num):
    """Format the output exactly like the console output."""
    lines = []
    lines.append(f"Query: {query}")
    lines.append(f"Retrieved {len(results)} passages\n")

    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        lines.append(f"--- Passage {i+1} (score: {score:.4f}) ---")
        lines.append(f"Source: {source}, Page: {page}")
        lines.append(doc.page_content[:300] + "...\n")

    lines.append("=" * 60)
    lines.append(f"ANSWER:\n{answer}")

    return "\n".join(lines)


def run_batch():
    if not os.path.exists(QUESTIONS_FILE):
        print(f"ERROR: '{QUESTIONS_FILE}' not found.")
        print(f"Create a file called '{QUESTIONS_FILE}' with one question per line.")
        return

    # Read questions (skip empty lines and comments starting with #)
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = [
            line.strip() for line in f.readlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    if not questions:
        print("No questions found in questions.txt")
        return

    print(f"Found {len(questions)} questions")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load vector store once
    print("Loading vector store...")
    vector_store = load_vector_store()

    # Process each question
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question[:80]}...")

        # Retrieve
        results = retrieve_hybrid(vector_store, question, max_results=10)

        # Generate answer
        answer = generate_answer(question, results)

        # Format output
        output = format_result(question, results, answer, i)

        # Save to file
        filename = f"{i:02d}_{slugify(question)}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(output)

        print(f"Saved: {filepath}")

    # Also save a combined file with all results
    print(f"\nSaving combined results...")
    combined_path = os.path.join(OUTPUT_DIR, "all_results.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        for i, question in enumerate(questions, 1):
            filename = f"{i:02d}_{slugify(question)}.txt"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as qf:
                f.write(qf.read())
            f.write("\n\n" + "#" * 80 + "\n\n")

    print(f"Saved: {combined_path}")
    print(f"\nDone. {len(questions)} results saved to '{OUTPUT_DIR}/' folder.")


if __name__ == "__main__":
    run_batch()