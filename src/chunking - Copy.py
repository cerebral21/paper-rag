from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import re


def load_documents(doc_dir: str):
    """Load all PDFs from a directory."""
    documents = []
    for filename in os.listdir(doc_dir):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(doc_dir, filename))
            documents.extend(loader.load())
    print(f"Loaded {len(documents)} pages from {doc_dir}")
    return documents


def chunk_fixed_size(documents, chunk_size=512, chunk_overlap=50):
    """Strategy 1: Fixed-size chunking with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # character count; switch to tiktoken for token count later
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Fixed-size chunking: {len(documents)} pages → {len(chunks)} chunks")
    return chunks


def chunk_semantic(documents, max_chunk_size=1000):
    """Strategy 2: Semantic chunking — split on paragraph boundaries."""
    # Start simple: split on double newlines (paragraphs), merge small chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=0,  # No overlap — boundaries are semantic
        length_function=len,
        separators=["\n\n", "\n"],  # Prefer paragraph breaks
    )
    chunks = splitter.split_documents(documents)
    print(f"Semantic chunking: {len(documents)} pages → {len(chunks)} chunks")
    return chunks

def is_reference_chunk(text):
    """
    Detect if a chunk is from a references/bibliography section,
    a dataset/benchmark table, or other non-content sections.
    
    Catches:
    - Traditional numbered references: [1] Author, Title...
    - Author-first references: Author et al. (2024). Title...
    - Dataset/benchmark tables: Name Language Format Size URL
    - Chunks dominated by URLs
    - Mathematical notation chunks (proofs, lemmas with minimal text)
    """
    text_lower = text.lower().strip()
 
    # ---- SECTION HEADERS ----
    # Matches "References", "7. REFERENCES", "Bibliography", etc.
    if re.search(r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:references|bibliography|works cited)\s*(?:\n|$)', text_lower):
        return True
 
    # ---- URL DENSITY ----
    urls = re.findall(r'https?://\S+', text)
    # Any chunk with 3+ URLs is almost certainly references or dataset tables
    if len(urls) >= 3:
        return True
 
    # ---- NUMBERED REFERENCE PATTERNS ----
    # [1] Author... [2] Author... style
    bracket_refs = re.findall(r'\[\d+\]\s*[A-Z]', text)
    if len(bracket_refs) >= 3:
        return True
 
    # ---- AUTHOR-YEAR REFERENCE PATTERNS ----
    # "Author et al., 2024" or "(Author, 2024)" or "Author (2024)"
    author_year = re.findall(r'(?:et\s+al\.?,?\s*(?:19|20)\d{2})|(?:\(\w+(?:\s+et\s+al\.?)?,?\s*(?:19|20)\d{2}\))', text)
    # "2024. URL https://" or "2024. doi:"
    year_url = re.findall(r'(?:19|20)\d{2}\.\s*(?:URL\s+https?://|doi:)', text)
    if len(author_year) + len(year_url) >= 4:
        return True
 
    # ---- DOI / ARXIV DENSITY ----
    dois = re.findall(r'doi[:\s]', text, re.IGNORECASE)
    arxivs = re.findall(r'arXiv', text, re.IGNORECASE)
    if len(dois) + len(arxivs) >= 3:
        return True
 
    # ---- DATASET TABLE PATTERNS ----
    # Rows like: "DatasetName (Author, 2022) en Retrieval 90,850 huggingface.co/..."
    # or: "DatasetName (Author et al., 2023) zh Clustering 100,000 github.com/..."
    dataset_rows = re.findall(
        r'\([\w\s.]+,?\s*(?:19|20)\d{2}[a-z]?\)\s*(?:en|zh|ru|es|fr|de|ja|ko|\d+|python|cpp|sql|java)\s*(?:Retrieval|Clustering|Classification|Reranking)',
        text
    )
    if len(dataset_rows) >= 2:
        return True
 
    # Simpler pattern: lines with "huggingface.co/datasets/" or "github.com/" repeated
    hf_github = re.findall(r'(?:huggingface\.co|github\.com)/', text)
    if len(hf_github) >= 3:
        return True
 
    # ---- COMBINED DENSITY SCORE ----
    # For chunks that have a mix of signals but don't trip any single threshold
    bracket_count = len(re.findall(r'\[\d+\]', text))
    url_count = len(urls)
    year_count = len(re.findall(r'(?:19|20)\d{2}[.,)\s]', text))
    doi_count = len(dois)
    arxiv_count = len(arxivs)
 
    density = bracket_count + url_count + arxiv_count + doi_count + (year_count * 0.3)
    if density > 5:
        return True
 
    # ---- CONFERENCE PROCEEDINGS BLOCKS ----
    # "In Proceedings of the..." or "Association for Computational Linguistics"
    proceedings = len(re.findall(r'(?:[Ii]n\s+Proceedings\s+of|Association\s+for\s+Computational|Conference\s+on\s+[A-Z])', text))
    if proceedings >= 2:
        return True
 
    # ---- AUTHOR-FIRST REFERENCES WITH URLS ----
    # Pattern: "Author Name. Title. Venue, year. URL https://..." or "doi: 10.xxx"
    # These have author names followed by periods, then venue info, then URL/doi
    author_then_url = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z]\.?)+'
                                  r'.+?(?:19|20)\d{2}'
                                  r'.+?(?:URL\s+https?://|doi:\s*10\.)', text, re.DOTALL)
    if len(author_then_url) >= 2:
        return True
 
    # ---- NUMBERED REFERENCES WITH QUOTED TITLES ----
    # [54] D. Author, "Title of paper," in Proceedings...
    quoted_refs = re.findall(r'\[\d+\]\s*[A-Z].*?"[^"]+', text, re.DOTALL)
    if len(quoted_refs) >= 2:
        return True
 
    # ---- BULLET-POINT BENCHMARK/DATASET LISTS ----
    # "• BenchmarkName(Author et al., 2023): Description..."
    bullet_benchmarks = re.findall(r'•\s*\w+.*?\([\w\s.,&]+(?:19|20)\d{2}\)', text)
    if len(bullet_benchmarks) >= 2:
        return True
 
    # ---- MIXED REFERENCE SIGNALS ----
    # Chunks with a bracket ref + quoted title + proceedings mention
    bracket_count = len(re.findall(r'\[\d+\]', text))
    quoted_titles = len(re.findall(r'"[^"]{10,},"', text))
    mixed = bracket_count + len(urls) + proceedings + quoted_titles
    if mixed >= 4:
        return True
 
    return False

def filter_reference_chunks(chunks):
    """Remove reference/bibliography chunks."""
    filtered = [c for c in chunks if not is_reference_chunk(c.page_content)]
    removed = len(chunks) - len(filtered)
    print(f"Filtered: {len(chunks)} → {len(filtered)} chunks ({removed} reference/noise chunks removed)")
    return filtered

