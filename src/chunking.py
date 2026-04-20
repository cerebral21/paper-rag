import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import re


# ============================================================
# DOCUMENT LOADING — Column-Aware PDF Extraction
# ============================================================

def extract_page_column_aware(page):
    """
    Extract text from a single PDF page, handling both single-column 
    and two-column layouts automatically.
    
    How it works:
    1. PyMuPDF extracts text as positioned blocks (rectangles of text 
       with x,y coordinates on the page)
    2. Each block is classified as LEFT column, RIGHT column, or 
       FULL-WIDTH based on its horizontal position and width
    3. Layout detection: if both left and right columns have substantial 
       text (>100 chars each), the page is two-column
    4. Two-column pages: full-width blocks are placed at their vertical 
       position, then left column is read top-to-bottom, then right 
       column top-to-bottom. Full-width blocks act as section dividers.
    5. Single-column pages: all blocks are read top-to-bottom normally
    """
    page_width = page.rect.width
    mid = page_width / 2
    
    # Get all text blocks with positions
    blocks = page.get_text("blocks")
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    
    if not text_blocks:
        return ""
    
    # Classify each block by horizontal position
    left_blocks = []
    right_blocks = []
    full_blocks = []
    
    for block in text_blocks:
        x0, y0, x1, y1, text, block_no, block_type = block
        block_width = x1 - x0
        block_center = (x0 + x1) / 2
        text = text.strip()
        if not text:
            continue
        
        # Full-width: spans more than 55% of page width
        if block_width > page_width * 0.55:
            full_blocks.append((y0, y1, text))
        # Left column: center of block is in the left half
        elif block_center < mid:
            left_blocks.append((y0, y1, text))
        # Right column: center of block is in the right half
        else:
            right_blocks.append((y0, y1, text))
    
    # Detect layout based on text volume in each column
    left_text_len = sum(len(t) for _, _, t in left_blocks)
    right_text_len = sum(len(t) for _, _, t in right_blocks)
    is_two_column = (left_text_len > 100 and right_text_len > 100)
    
    if not is_two_column:
        # Single column — sort all blocks by vertical position
        all_blocks = full_blocks + left_blocks + right_blocks
        all_blocks.sort(key=lambda b: b[0])
        return "\n\n".join(text for _, _, text in all_blocks)
    
    # Two-column — reconstruct proper reading order
    left_blocks.sort(key=lambda b: b[0])
    right_blocks.sort(key=lambda b: b[0])
    full_blocks.sort(key=lambda b: b[0])
    
    # Build a timeline of all blocks with their type
    events = []
    for y0, y1, text in full_blocks:
        events.append((y0, y1, "FULL", text))
    for y0, y1, text in left_blocks:
        events.append((y0, y1, "LEFT", text))
    for y0, y1, text in right_blocks:
        events.append((y0, y1, "RIGHT", text))
    events.sort(key=lambda e: e[0])
    
    # Reconstruct: when we hit a full-width block, flush accumulated 
    # column content (left then right), then add the full-width block
    result_parts = []
    current_left = []
    current_right = []
    
    for y0, y1, label, text in events:
        if label == "FULL":
            for _, _, t in sorted(current_left, key=lambda x: x[0]):
                result_parts.append(t)
            for _, _, t in sorted(current_right, key=lambda x: x[0]):
                result_parts.append(t)
            current_left = []
            current_right = []
            result_parts.append(text)
        elif label == "LEFT":
            current_left.append((y0, y1, text))
        elif label == "RIGHT":
            current_right.append((y0, y1, text))
    
    # Flush remaining column content
    for _, _, t in sorted(current_left, key=lambda x: x[0]):
        result_parts.append(t)
    for _, _, t in sorted(current_right, key=lambda x: x[0]):
        result_parts.append(t)
    
    return "\n\n".join(part for part in result_parts if part)


def load_documents(doc_dir: str):
    """
    Load all PDFs from a directory with column-aware text extraction.
    
    Uses PyMuPDF instead of PyPDFLoader for better handling of 
    two-column academic paper layouts. Automatically detects single 
    vs two-column pages and extracts text in proper reading order.
    
    Returns a list of Document objects, one per page, with metadata
    containing source filepath and page number.
    """
    documents = []
    for filename in sorted(os.listdir(doc_dir)):
        if not filename.endswith(".pdf"):
            continue
        filepath = os.path.join(doc_dir, filename)
        try:
            doc = fitz.open(filepath)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = extract_page_column_aware(page)
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": filepath, "page": page_num}
                    ))
            doc.close()
        except Exception as e:
            print(f"Warning: Could not process {filename}: {e}")
    print(f"Loaded {len(documents)} pages from {doc_dir}")
    return documents


# ============================================================
# CHUNKING STRATEGIES
# ============================================================

def chunk_fixed_size(documents, chunk_size=512, chunk_overlap=50):
    """Strategy 1: Fixed-size chunking with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Fixed-size chunking: {len(documents)} pages -> {len(chunks)} chunks")
    return chunks


def chunk_semantic(documents, max_chunk_size=1000):
    """Strategy 2: Semantic chunking — split on paragraph boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=0,
        length_function=len,
        separators=["\n\n", "\n"],
    )
    chunks = splitter.split_documents(documents)
    print(f"Semantic chunking: {len(documents)} pages -> {len(chunks)} chunks")
    return chunks


# ============================================================
# REFERENCE AND NOISE FILTERING
# ============================================================

def is_reference_chunk(text):
    """
    Detect if a chunk is from a references/bibliography section,
    a dataset/benchmark table, or other non-content sections.
    """
    text_lower = text.lower().strip()

    # Section headers
    if re.search(r'(?:^|\n)\s*(?:\d+\.?\s*)?(?:references|bibliography|works cited)\s*(?:\n|$)', text_lower):
        return True

    # URL density
    urls = re.findall(r'https?://\S+', text)
    if len(urls) >= 3:
        return True

    # Numbered references
    bracket_refs = re.findall(r'\[\d+\]\s*[A-Z]', text)
    if len(bracket_refs) >= 3:
        return True

    # Author-year patterns
    author_year = re.findall(r'(?:et\s+al\.?,?\s*(?:19|20)\d{2})|(?:\(\w+(?:\s+et\s+al\.?)?,?\s*(?:19|20)\d{2}\))', text)
    year_url = re.findall(r'(?:19|20)\d{2}\.\s*(?:URL\s+https?://|doi:)', text)
    if len(author_year) + len(year_url) >= 4:
        return True

    # DOI / arXiv density
    dois = re.findall(r'doi[:\s]', text, re.IGNORECASE)
    arxivs = re.findall(r'arXiv', text, re.IGNORECASE)
    if len(dois) + len(arxivs) >= 3:
        return True

    # Dataset table patterns
    dataset_rows = re.findall(
        r'\([\w\s.]+,?\s*(?:19|20)\d{2}[a-z]?\)\s*(?:en|zh|ru|es|fr|de|ja|ko|\d+|python|cpp|sql|java)\s*(?:Retrieval|Clustering|Classification|Reranking)',
        text
    )
    if len(dataset_rows) >= 2:
        return True

    # HuggingFace/GitHub URL rows
    hf_github = re.findall(r'(?:huggingface\.co|github\.com)/', text)
    if len(hf_github) >= 3:
        return True

    # Combined density score
    bracket_count = len(re.findall(r'\[\d+\]', text))
    url_count = len(urls)
    year_count = len(re.findall(r'(?:19|20)\d{2}[.,)\s]', text))
    doi_count = len(dois)
    arxiv_count = len(arxivs)
    density = bracket_count + url_count + arxiv_count + doi_count + (year_count * 0.3)
    if density > 5:
        return True

    # Conference proceedings
    proceedings = len(re.findall(r'(?:[Ii]n\s+Proceedings\s+of|Association\s+for\s+Computational|Conference\s+on\s+[A-Z])', text))
    if proceedings >= 2:
        return True

    # Author-first references with URLs
    author_then_url = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z]\.?)+'
                                  r'.+?(?:19|20)\d{2}'
                                  r'.+?(?:URL\s+https?://|doi:\s*10\.)', text, re.DOTALL)
    if len(author_then_url) >= 2:
        return True

    # Numbered references with quoted titles
    quoted_refs = re.findall(r'\[\d+\]\s*[A-Z].*?"[^"]+', text, re.DOTALL)
    if len(quoted_refs) >= 2:
        return True

    # Bullet-point benchmark lists
    bullet_benchmarks = re.findall(r'[•]\s*\w+.*?\([\w\s.,&]+(?:19|20)\d{2}\)', text)
    if len(bullet_benchmarks) >= 2:
        return True

    # Mixed reference signals
    bracket_count = len(re.findall(r'\[\d+\]', text))
    quoted_titles = len(re.findall(r'"[^"]{10,},"', text))
    mixed = bracket_count + len(urls) + proceedings + quoted_titles
    if mixed >= 4:
        return True

    return False


def filter_reference_chunks(chunks):
    """Remove reference/bibliography/table chunks from the chunk list."""
    filtered = [c for c in chunks if not is_reference_chunk(c.page_content)]
    removed = len(chunks) - len(filtered)
    print(f"Filtered: {len(chunks)} -> {len(filtered)} chunks ({removed} reference/noise chunks removed)")
    return filtered