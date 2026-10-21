# Technical Document Q&A with RAG

A retrieval-augmented generation (RAG) system for answering questions over technical research papers. It ingests PDFs, retrieves passages with a hybrid search pipeline (semantic + BM25, fused and re-ranked), and generates answers that cite their source passages, give partial answers when coverage is incomplete, and abstain when the corpus has no relevant passage.

I built it to study RAG design decisions one at a time (PDF extraction, chunking, retrieval, grounding) and to measure each change on a fixed question set instead of judging by feel.

**Headline results** (25-question evaluation, manually judged):

| Metric | Result |
|---|---|
| Retrieval precision, semantic search only | ~55% |
| Retrieval precision, hybrid retrieval + cross-encoder re-ranking | ~85% |
| Answers with unsupported claims (final configuration) | 0 of 25 (manually checked against the retrieved passages) |

Read [Evaluation](#evaluation) for what these numbers do and do not show.

---

## How it works

```
PDFs --> column-aware extraction (PyMuPDF)
     --> 512-char chunks, 50 overlap
     --> reference/noise filter
     --> ChromaDB (vectors) + JSON store (for BM25)

Query --> semantic top 20 (all-MiniLM-L6-v2) --\
                                                 >-- RRF (k=60) --> cross-encoder re-rank (top 20)
      --> BM25 top 20 (rank_bm25) --------------/                       |
                                                                         v
                                              score cutoff (>= -2.0), at most 10 passages
                                                                         |
                                                                         v
                                        GPT-4o-mini: grounded answer with [Passage N] citations,
                                        partial answer, or abstention (no LLM call if 0 passages)
```

## Design decisions and what each one fixed

### PDF extraction: column-aware (PyMuPDF)

Standard PDF loaders read straight across a two-column page, interleaving the left and right columns and producing garbled chunks. This system reads text blocks with their x/y coordinates, classifies each as left, right or full-width, detects two-column pages (both columns need more than 100 characters), and rebuilds reading order: full-width blocks at their vertical position, then the left column, then the right.

Effect: it fixed chunks like "Matryoshka Representation Learning" being cut to "tion Learning" at a column boundary, and gave cleaner chunk text on every two-column paper.

### Chunking: fixed size

`RecursiveCharacterTextSplitter` with 512-character chunks and 50 characters of overlap, preferring paragraph, line, sentence and word boundaries in that order. Uniform chunks give consistent embeddings, at the cost of sometimes splitting a logical unit. A paragraph-based `chunk_semantic` function exists in the code but has not been evaluated against it.

### Reference and noise filtering

A set of regex heuristics removes bibliography and table chunks before embedding: reference headers, URL density, numbered and author-year citations, DOI/arXiv density, dataset tables, conference-proceedings blocks and similar patterns. It removes roughly 15-20% of chunks and stops reference lists and benchmark tables from polluting retrieval. It is a preprocessing step, not a retrieval fix, and its false-positive rate (genuine content dropped) has not been measured.

### Retrieval: three iterations

1. **Semantic search only.** Cosine similarity over MiniLM embeddings. It failed when different papers shared vocabulary but not topic: "graph" in knowledge-graph clustering matched "graph" in SQL query optimization. Precision about 55%.
2. **Add BM25 and Reciprocal Rank Fusion (k=60).** A second, keyword-based ranking helped exact-terminology queries but did not resolve the vocabulary-overlap problem. I did not measure precision for this stage on its own; it was evaluated together with iteration 3.
3. **Add cross-encoder re-ranking** (`ms-marco-MiniLM-L-6-v2`). Reading the query and passage together separates topical relevance from surface similarity. Together with iteration 2 this raised precision to about 85%. Because the two were measured together, the numbers do not show how much each contributed; the cross-encoder is what fixed the "graph" mismatch above.

A final cutoff (cross-encoder score of at least -2.0) lets the number of passages vary from 0 to 10 with how well the corpus covers the question, instead of always returning a fixed count. In the final run, 3 of the 25 questions returned no passages and the system abstained; the average was 5.5 passages per query.

### Generation: grounding, partial answers, abstention

The prompt requires citations as `[Passage N]`, tells the model to answer only what the passages support and to state what is missing, and forbids filling gaps from general knowledge. In early runs, without the partial-coverage instruction, the model tended either to refuse entirely or to fill gaps from its own knowledge. If retrieval returns no passages, the code skips the LLM call and returns a fixed "no relevant passages" message, so the model cannot answer from its own knowledge when the corpus has no coverage.

---

## Evaluation

### Corpus

8 research papers (171 pages, committed in `data/documents/`) on learning to rank, multilingual embeddings, LLM refusal behavior, agent memory systems and cross-store query optimization. They are arXiv papers from February and March 2026, which post-date GPT-4o-mini's training data, so a correct answer has to come from retrieval rather than memorization. The corpus deliberately mixes adjacent topics to test retrieval under vocabulary overlap.

### Question set

25 hand-written questions (`questions.txt`): 15 answerable from the corpus, 5 partially answerable and 5 not answerable.

### Retrieval precision

Precision is an informal, manual measure: for each question I read the retrieved results and judged whether they were relevant to it. I did not record per-passage labels or script the calculation, so the figures are approximate (hence "~") and cannot be recomputed from this repo.

| Run | Retrieval | Precision |
|---|---|---|
| 1 | Semantic search only | ~55% |
| 2 | + Reference filter | ~55% |
| 3 | + BM25 and RRF + cross-encoder re-ranking | ~85% |

The reference filter cleaned the text but did not change precision. BM25 + RRF and the cross-encoder were evaluated together in run 3, so there is no separate number for BM25 + RRF alone. Adding the score cutoff and the 10-passage limit afterwards left the figure at ~85%.

### Final configuration: answer outcomes (25 questions, manually judged)

| Outcome | Count |
|---|---|
| Correct full answer | 10 |
| Correct partial answer (states what the corpus does not cover) | 11 |
| Correct refusal (question not answerable from the corpus) | 4 |
| Incorrect refusal | 0 |
| Unsupported or hallucinated claim | 0 |
| Average passages per query | 5.5 |

Raw outputs for every question are in `results_25_final/`.

### Groundedness check

I read each of the 25 answers against its retrieved passages and flagged any claim the passages did not support; none were found. This measures **groundedness**, meaning the model stayed within the context it was given. It does not measure **correctness**: an answer can be faithful to the retrieved text and still be wrong if the retrieved text was only partly relevant (precision is about 85%, not 100%).

### What the evaluation does and does not show

- The 25 questions were also used to guide development, so they are a development set, not a held-out test. The numbers show the effect of each change on this corpus, not performance on unseen questions.
- 25 questions is a sanity check, not statistical evidence. A single question moves a percentage by 4 points.
- Judgments were made by one person (me), with no second annotator.
- Precision was judged by hand without saved labels (see above), and the outputs of the earlier stages were not kept, so only the final configuration can be inspected in this repo.

---

## Project structure

```
ingest.py            Build the vector store and BM25 chunk store (run once)
batch_query.py       Run every question in questions.txt, save to results/
main.py              Interactive single-question mode
questions.txt        The 25 evaluation questions
results_25_final/    Outputs of the final configuration (one file per question)
data/documents/      The PDF corpus
src/
  chunking.py        Column-aware loading, chunking, reference filtering
  embeddings.py      ChromaDB vector store
  retreival.py       Hybrid retrieval (semantic + BM25 + RRF + cross-encoder)
  generation.py      Grounded answer generation
```

## Setup

Requires Python 3.10+ and an OpenAI API key (evaluation of the 25 questions costs a few hundred rupees at GPT-4o-mini prices).

```bash
git clone https://github.com/cerebral21/paper-rag.git
cd paper-rag
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

pip install langchain langchain-openai langchain-community langchain-chroma \
            langchain-text-splitters langchain-huggingface chromadb \
            sentence-transformers pymupdf rank_bm25 python-dotenv numpy

echo "OPENAI_API_KEY=your-key-here" > .env     # never commit this file

python ingest.py          # step 1: build the indexes
python batch_query.py     # step 2a: run the 25-question evaluation (writes to results/)
python main.py            # step 2b: or ask questions interactively
```

## Tech stack

PyMuPDF, LangChain text splitters, ChromaDB, `all-MiniLM-L6-v2` embeddings, `rank_bm25`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, GPT-4o-mini.

## Limitations and next steps

- **No held-out question set.** Write a fresh set of questions, fixed before any further tuning, and report results on it separately.
- **Make precision reproducible.** Save per-question, per-passage relevance judgments for each configuration in a CSV and compute precision from it with a script.
- **BM25 index is rebuilt on every query** from `chunk_store.json`; build it once and cache it.
- **BM25 tokenization** is plain lowercase whitespace splitting, with no punctuation handling or stemming.
- **Latency:** cross-encoder re-ranking adds a few seconds per query.
- **Chunking comparison:** evaluate semantic (paragraph) chunking against fixed-size on the same question set.
- **Single-turn only:** no conversation history or follow-up handling.
- **Corpus-dependent:** the system abstains when coverage is absent but cannot point users to external sources.
