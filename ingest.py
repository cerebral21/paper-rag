import os
import shutil
import warnings
import logging
logging.getLogger("safetensors").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
warnings.filterwarnings("ignore")

from src.chunking import load_documents, chunk_fixed_size, filter_reference_chunks
from src.embeddings import create_vector_store
from src.retreival import save_chunks_for_bm25

def ingest():
    if os.path.exists("./.chroma"):
        shutil.rmtree("./.chroma")
    docs = load_documents("data/documents")
    chunks = chunk_fixed_size(docs, chunk_size=512, chunk_overlap=50)
    chunks = filter_reference_chunks(chunks)
    create_vector_store(chunks)
    save_chunks_for_bm25(chunks)
    print("Ingestion complete. Vector store + BM25 index ready.")



if __name__ == "__main__":
    ingest()