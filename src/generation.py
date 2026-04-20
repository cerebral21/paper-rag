from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def generate_answer(query: str, retrieved_docs):
    """Generate an answer from retrieved context."""
    if not retrieved_docs:
        return ("I don't have enough information in the provided documents "
                "to answer this question. No relevant passages were found.")
    # Format the context with source info
    context_parts = []
    for i, (doc, score) in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Passage {i+1}] (Source: {source}, Page: {page})\n{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a technical documentation assistant. Answer the user's
question based on the provided context passages below.

Rules:
- Base your answer on the information found in the provided passages
- Cite which passage number supports each claim, e.g. [Passage 1]
- If the passages contain relevant information, synthesize it into a clear answer
- If only part of the question can be answered from the passages, answer that
  part thoroughly and explicitly state which aspects the documents don't cover.
  For example: "Based on the documents, [answer the answerable part]. However,
  the provided documents do not discuss [unanswered aspect]."
- If the passages truly contain no relevant information at all, say:
  "I don't have enough information in the provided documents to answer
  this question."
- Do not add information that isn't supported by the passages, even if you
  know the answer from general knowledge
- If passages mention a concept briefly without explaining it, say what the
  documents state and note that they don't provide a detailed explanation

Context:
{context}

Question: {query}

Answer:"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(prompt)
    return response.content
