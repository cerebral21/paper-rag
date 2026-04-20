from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def create_vector_store(chunks, collection_name="tech_docs"):
    # if os.path.exists("./.chroma"):
    #     shutil.rmtree("./.chroma")
    """Embed chunks and store in ChromaDB."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory="./.chroma",
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return vector_store


def load_vector_store(collection_name="tech_docs"):
    """Load an existing vector store."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory="./.chroma",
    )
    return vector_store