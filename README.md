# Technical Document Q&A System with RAG

A Retrieval-Augmented Generation system for querying technical research papers. The system ingests PDF documents, retrieves relevant passages using a hybrid search pipeline, and generates grounded answers with source citations.

Built as a practical exploration of RAG architecture design decisions — PDF extraction, chunking strategies, retrieval methods, and answer quality evaluation — with quantified results across each iteration.

---

## Architecture

```
PDF Documents
      │
      ▼
┌──────────────────┐
│    Ingestion      │
│                  │
│  PyMuPDF         │  Column-aware extraction
│  (fitz)          │  (auto-detects single vs two-column layouts)
│       │          │
│       ▼          │
│  Fixed-size      │  512 chars, 50 overlap
│  Chunking        │  RecursiveCharacterTextSplitter
│       │          │
│       ▼          │
│  Reference       │  10-pattern heuristic filter
│  Filtering       │  Removes ~15-20% noise chunks
│       │          │
│       ▼          │
│  Dual Storage    │  ChromaDB (vectors) + JSON (BM25 index)
└──────┬───────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│              Hybrid Retrieval Pipeline            │
│                                                  │
│  ┌──────────────┐    ┌──────────────┐            │
│  │   Semantic    │    │    BM25      │            │
│  │   Search      │    │   Keyword    │            │
│  │  (top 20)     │    │   Search     │            │
│  │  ChromaDB +   │    │  (top 20)    │            │
│  │  MiniLM-L6-v2 │    │  rank_bm25   │            │
│  └──────┬────────┘    └──────┬───────┘            │
│         │                    │                    │
│         └────────┬───────────┘                    │
│                  ▼                                │
│     ┌────────────────────────┐                    │
│     │  Reciprocal Rank Fusion │                   │
│     │      (k=60)             │                   │
│     └───────────┬─────────────┘                   │
│                 ▼                                 │
│     ┌────────────────────────┐                    │
│     │  Cross-Encoder Reranking│                   │
│     │  ms-marco-MiniLM-L-6-v2│                    │
│     │  top 20 → scored       │                    │
│     └───────────┬─────────────┘                   │
│                 ▼                                 │
│     ┌────────────────────────┐                    │
│     │  Score Threshold Filter │                   │
│     │  (min_score = -2.0)     │                   │
│     │  Returns 0-10 passages  │                   │
│     └────────────────────────┘                    │
└──────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────┐
│    Generation     │
│                  │
│  GPT-4o-mini     │  Grounding prompt with citation rules
│                  │  → Full answers with [Passage N] citations
│                  │  → Partial answers when coverage is incomplete
│                  │  → Abstention when 0 passages retrieved
└──────────────────┘
```

---

## Design Decisions and Tradeoffs

### PDF Extraction: Column-Aware (PyMuPDF)

Standard PDF text extraction (PyPDFLoader) reads left-to-right, top-to-bottom across the entire page. For two-column academic papers (IEEE, ACL, NeurIPS formats), this interleaves text from the left and right columns, producing garbled chunks that embed poorly and retrieve incorrectly.

The system uses PyMuPDF (`fitz`) to extract text blocks with their x,y coordinates, then classifies each block as LEFT column, RIGHT column, or FULL-WIDTH based on horizontal position. Two-column pages are detected automatically (both columns must have >100 characters of text), and text is reconstructed in proper reading order: full-width content at its vertical position, then left column top-to-bottom, then right column top-to-bottom.

**Impact:** Fixed a specific chunking artifact where "Matryoshka Representation Learning" was being truncated to "tion Learning" at a column boundary. Also improved retrieval scores across all two-column papers by producing more coherent chunk text.

### Chunking: Fixed-Size (512 characters, 50 overlap)

Fixed-size chunking with `RecursiveCharacterTextSplitter` splits text into chunks of at most 512 characters, preferring natural boundaries (paragraph breaks → newlines → sentence endings → word boundaries). The 50-character overlap ensures concepts spanning a chunk boundary partially appear in both chunks.

**Tradeoff:** Uniform chunk sizes produce consistent embedding quality. Occasionally splits logical units when no natural boundary exists within the window. A semantic chunking function (`chunk_semantic`) is stubbed in the codebase for future comparison.

### Reference and Noise Filtering

A 10-pattern heuristic filter removes non-content chunks before embedding. Developed iteratively based on observed noise in retrieval results.

**Patterns detected:**
- Section headers (References, Bibliography)
- URL density (3+ URLs per chunk)
- Numbered references ([1] Author... style, 3+ per chunk)
- Author-year citation patterns (4+ per chunk)
- DOI/arXiv density (3+ per chunk)
- Dataset/benchmark tables (HuggingFace/GitHub URL rows)
- Conference proceedings blocks
- Author-first references with DOI/URL
- Quoted paper titles in numbered references
- Bullet-point benchmark/dataset lists

**Impact:** Removes ~15-20% of chunks. Eliminates dataset tables from F2LLM-v2 and reference sections from all papers that were polluting retrieval results.

### Retrieval: Hybrid Search + Re-ranking

The system evolved through three iterations, each addressing failures observed in evaluation:

**Iteration 1 — Semantic search only:** Cosine similarity between query and chunk embeddings (all-MiniLM-L6-v2). Failed when different papers shared vocabulary but discussed different topics — "graph" in knowledge graph clustering matched "graph" in SQL query optimization. Retrieval precision: ~55%.

**Iteration 2 — Added BM25 + Reciprocal Rank Fusion:** BM25 keyword search provides a second ranking signal based on exact term matching. RRF (k=60) merges both ranked lists, penalizing passages that only one method considers relevant. Improved precision on exact-terminology queries but didn't fully resolve the vocabulary overlap problem.

**Iteration 3 — Added cross-encoder re-ranking:** A cross-encoder (ms-marco-MiniLM-L-6-v2) reads query and passage together as a single input, producing relevance scores that distinguish semantic similarity from topical relevance. This was the single highest-impact improvement, resolving the core vocabulary overlap problem. Retrieval precision: ~85%.

**Score threshold filtering (min_score = -2.0):** The cross-encoder produces positive scores for relevant passages and negative scores for irrelevant ones. Filtering at -2.0 means queries return 0-10 passages based on actual corpus coverage rather than a fixed count. Well-covered topics return up to 10 passages; topics absent from the corpus correctly return 0.

### Generation: Grounding and Partial Answers

The prompt is designed around three behaviors:

1. **Full answer** when passages contain sufficient information — synthesize and cite with [Passage N] tags.
2. **Partial answer** when passages partially cover the question — answer what's covered, explicitly state what's missing ("However, the provided documents do not discuss...").
3. **Abstention** when no relevant passages are retrieved — a hard-coded check skips the LLM call entirely and returns "No relevant passages were found." This prevents the LLM from generating answers from its own knowledge when the corpus has no coverage.

---

## Evaluation

### Corpus

9 research papers (~200 pages) covering Learning to Rank, multilingual embeddings, LLM safety, multi-agent memory systems, and cross-model query optimization. The corpus deliberately includes topically adjacent but semantically different papers to test retrieval precision under vocabulary overlap.

### Evaluation Dataset

25 manually constructed questions in three categories:

- **Factual (15):** Answer is in the corpus. Tests retrieval accuracy and answer grounding.
- **Partial coverage (5):** Answer is partially in the corpus. Tests honest partial answers vs hallucination.
- **Unanswerable (5):** Answer is not in the corpus. Tests abstention behavior.

### Results Across Iterations

| Metric | Semantic Only | + Ref Filter | + Hybrid + Reranking | + Score Threshold | Final (max 10) |
|---|---|---|---|---|---|
| Retrieval precision | ~55% | ~55% | ~85% | ~85% | ~85% |
| Correct full answers | 4/12 | 4/12 | 5/12 | 12/25 | 10/25 |
| Correct partial answers | 0/12 | 3/12 | 5/12 | 7/25 | 11/25 |
| Correct refusals | 3/12 | 3/12 | 2/12 | 3/25 | 4/25 |
| Incorrect refusals | 3/12 | 2/12 | 0/12 | 0/25 | 0/25 |
| Hallucinated answers | 2/12 | 0/12 | 0/12 | 0/25 | 0/25 |
| Avg passages/query | 5.0 | 5.0 | 5.0 | 3.7 | 5.5 |

### Key Findings

**Cross-encoder re-ranking was the single highest-impact improvement.** It resolved the vocabulary overlap problem that neither chunking improvements nor BM25 could fix. Reading query and passage jointly as a single input is fundamentally more powerful than comparing independent embeddings.

**Column-aware PDF extraction fixed text quality at the source.** Two-column papers that were producing garbled chunks now extract cleanly, improving both embedding quality and answer coherence.

**Score threshold filtering trades recall for precision.** Some queries return 1-2 passages instead of 10, but those passages are genuinely relevant. The LLM produces better answers from 2 relevant passages than from 2 relevant + 8 noisy ones.

**Variable passage count reflects corpus coverage honestly.** Well-covered topics return 10 passages with rich multi-citation answers. Absent topics return 0 passages with clean abstention. This is more informative than always returning a fixed count.

**The partial answer prompt is critical for honest behavior.** Without explicit instructions to handle partial coverage, the LLM either refuses entirely (losing useful information) or fills gaps from its own knowledge (hallucinating). The instruction "answer what you can, state what's missing" produces the most trustworthy outputs.

**Reference filtering prevents garbage at ingestion time** but doesn't fix retrieval ranking. It's a necessary preprocessing step, not a retrieval solution.

---

## Project Structure

```
├── ingest.py                # Document ingestion pipeline (run once)
├── batch_query.py           # Batch evaluation over questions.txt
├── main.py                  # Interactive single-query mode
├── questions.txt            # Evaluation questions (one per line)
├── src/
│   ├── chunking.py          # PDF loading (column-aware), chunking, reference filtering
│   ├── embeddings.py        # ChromaDB vector store (create + load)
│   ├── retreival.py         # Hybrid retrieval pipeline (semantic + BM25 + RRF + reranking)
│   └── generation.py        # LLM answer generation with grounding prompt
├── data/
│   └── documents/           # PDF corpus (not committed)
├── results/                 # Generated evaluation results
├── .chroma/                 # ChromaDB persistent storage (generated)
├── chunk_store.json         # BM25 chunk index (generated)
├── .env                     # API keys (not committed)
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API key (~₹300-500 total for development and evaluation)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/rag-technical-qa.git
cd rag-technical-qa

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install langchain langchain-openai langchain-community
pip install langchain-chroma langchain-text-splitters langchain-huggingface
pip install chromadb sentence-transformers pymupdf
pip install rank_bm25 python-dotenv numpy
```

### Configuration

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

Add PDFs to `data/documents/`.

### Usage

```bash
# Step 1: Ingest documents (run once, or when documents change)
python ingest.py

# Step 2a: Run evaluation batch
python batch_query.py

# Step 2b: Or query interactively
python main.py
```

---

## Tech Stack

| Component | Technology | Rationale |
|---|---|---|
| Language | Python 3.10+ | Standard for ML/NLP |
| PDF Extraction | PyMuPDF (fitz) | Column-aware text extraction for academic papers |
| Chunking | LangChain RecursiveCharacterTextSplitter | Respects natural text boundaries |
| Vector Store | ChromaDB | Simple persistence, HNSW indexing |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Best speed-quality tradeoff for local embedding |
| Keyword Search | rank_bm25 | Lightweight BM25 for hybrid search |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Joint query-passage relevance scoring |
| Generation | OpenAI GPT-4o-mini | Cost-efficient, replaceable with local model |

---

## Limitations and Future Work

- **Semantic chunking comparison:** A `chunk_semantic` function is stubbed for paragraph-boundary splitting. Evaluating both strategies on the same 25-question set would quantify chunking impact on answer quality.
- **Single-turn only:** No conversation history or follow-up question support.
- **Cross-encoder latency:** Re-ranking adds ~2-5 seconds per query. A distilled model or pre-computed cache could reduce this for production use.
- **Corpus-dependent:** The system correctly abstains when coverage is absent but cannot direct users to external resources.
- **BM25 tokenization:** Simple whitespace splitting doesn't handle punctuation or stemming. Production systems would benefit from a more sophisticated tokenizer.
