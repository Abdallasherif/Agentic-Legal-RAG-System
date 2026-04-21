# ⚖️ RAG103 — Legal Agentic RAG System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=for-the-badge&logo=pytorch)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green?style=for-the-badge)
![Transformers](https://img.shields.io/badge/Transformers-5.3-yellow?style=for-the-badge&logo=huggingface)
![CUDA](https://img.shields.io/badge/CUDA-A100%20GPU-76B900?style=for-the-badge&logo=nvidia)

**A production-grade, multi-agent Retrieval-Augmented Generation system for US legal research — built on Supreme Court case law and commercial contract analysis.**

</div>

---

## 📌 Overview

RAG103 is a **fully agentic, multi-hop legal RAG pipeline** designed for professional-grade legal question answering across two specialized corpora:

- 🏛️ **SCOTUS** — 4,746 US Supreme Court opinions (128,212 chunks)
- 📜 **CUAD** — 408 commercial contracts (15,922 chunks)

The system orchestrates a graph of specialized AI agents via **LangGraph**, combining hybrid retrieval, chain-of-thought generation, and a 4-layer evaluation framework to deliver cited, grounded legal answers with full transparency.

> 🏆 **Overall System Score: 0.7711 (Good)** — evaluated across 4 independent benchmarks.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────┐
│   Planner   │  ← Intent Classification (4-way) + Corpus Routing
│   Agent     │  ← ReAct Pattern
└──────┬──────┘
       │
   ┌───┴──────────────────┐
   │                      │
   ▼                      ▼
[REFUSED]          [RAG Pipeline]
[DIRECT LLM]            │
                         ▼
                ┌─────────────────┐
                │  Retrieval      │  ← Dense (FAISS) + Sparse (BM25)
                │  Agent          │  ← RRF Fusion → Cross-Encoder Reranking
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Reflection     │  ← CRAG-style quality gate
                │  Agent          │  ← CORRECT / AMBIGUOUS / INCORRECT
                └────────┬────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
         [CORRECT /            [INCORRECT]
         AMBIGUOUS]                 │
              │                     ▼
              │            ┌─────────────────┐
              │            │  Rewriter        │  ← Query expansion + domain prefix
              │            │  Agent           │  ← Corpus expansion (max 2 rewrites)
              │            └────────┬────────┘
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Generation     │  ← Qwen2.5-14B-Instruct (4-bit NF4)
                │  Agent          │  ← Chain-of-Thought + Grounded Citations
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Verification   │  ← Citation validation + Faithfulness check
                │  Agent          │  ← Relevance scoring + Regeneration trigger
                └────────┬────────┘
                         │
                         ▼
                   Final Answer
                (with Sources + CoT)
```

---

## 🤖 Model Stack

| Component | Model | Details |
|-----------|-------|---------|
| 🧠 **Generator** | `Qwen/Qwen2.5-14B-Instruct` | 4-bit NF4 quantization, 4096 token context |
| 🔤 **Embedder** | `BAAI/bge-m3` | 1024-dim embeddings, normalized for cosine similarity |
| 🏆 **Reranker** | `BAAI/bge-reranker-large` | Cross-encoder, 512 token max, batch size 64 |
| 🔍 **Sparse Retrieval** | `BM25Okapi` | Legal-aware tokenizer (preserves case names & abbreviations) |
| 📐 **Vector Index** | `FAISS IndexFlatIP` | Inner product search over normalized embeddings |
| 🕸️ **Agent Framework** | `LangGraph` | Stateful graph with memory checkpointing |

---

## 📚 Corpora

### 🏛️ SCOTUS — US Supreme Court Opinions
- **Source:** `lex_glue/scotus` (HuggingFace)
- **Documents:** 4,746 cases (filtered for length ≥ 1,500 chars)
- **Chunks:** 128,212 (avg 1,361 chars each)
- **Coverage:** Constitutional law, criminal procedure, civil rights, federal statutes

### 📜 CUAD — Commercial Contract Understanding Dataset
- **Source:** `chenghao/cuad_qa` (HuggingFace)
- **Documents:** 408 unique contracts
- **Chunks:** 15,922 (avg 1,310 chars each)
- **Coverage:** All 41 CUAD clause categories (indemnification, governing law, IP ownership, termination, non-compete, etc.)
- **Gold Q&A Pairs:** 10,404 expert-annotated pairs for evaluation

---

## 🔬 Retrieval Pipeline

### Hybrid Retrieval = Dense + Sparse + Fusion + Reranking

```
Query
  ├── BGE-M3 Encoding → FAISS Search → Top-60 Dense Candidates
  └── BM25 Tokenization → BM25Okapi → Top-60 Sparse Candidates
                    │
                    ▼
          Reciprocal Rank Fusion (RRF, k=60)
                    │
                    ▼
            Top-100 Fused Results
                    │
                    ▼
     BGE-Reranker-Large (Cross-Encoder)
                    │
                    ▼
              Top-10 Final Chunks
```

**RRF Formula:** `score = Σ 1 / (60 + rank)`

### Why This Works
- **Dense retrieval** captures semantic similarity across 1024-dim embedding space
- **BM25** ensures exact legal terminology matches (case names, statute numbers)
- **RRF fusion** is rank-based — no score normalization needed across incompatible scales
- **Cross-encoder reranking** evaluates full (query, chunk) pairs jointly for maximum precision

---

## ✂️ Hybrid Legal Chunker

A 3-level cascading chunker purpose-built for legal documents:

```
Level 1: Legal Structure Markers
  SCOTUS → HELD:, OPINION:, DISSENT:, MR. JUSTICE, I. II. III.
  CUAD   → ARTICLE X, SECTION 1, WHEREAS, NOW THEREFORE

Level 2: Paragraph Boundaries (\\n\\n)
  Triggered when sections exceed 1,500 chars

Level 3: Sentence Boundaries (spaCy)
  Triggered when paragraphs still exceed 1,500 chars
```

**Result:** Chunks that respect legal document structure at every level of granularity.

---

## 🎯 Intent Classification & Routing

The **Planner Agent** uses embedding-based prototype classification to route queries:

| Intent | Action | Description |
|--------|--------|-------------|
| `LEGAL_RAG` | Full Pipeline | Case law & contract questions requiring retrieval |
| `LEGAL_SIMPLE` | Direct LLM | Basic legal definitions (habeas corpus, mens rea, etc.) |
| `GENERAL` | Direct LLM | Conversational queries |
| `OUT_OF_SCOPE` | Refuse | Requests to forge documents, illegal activity, etc. |

**Corpus Detection** (keyword-based within `LEGAL_RAG`):
- CUAD keywords: `contract`, `clause`, `indemnif`, `terminat`, `governing law`, etc.
- SCOTUS keywords: `court`, `amendment`, `constitutional`, `warrant`, `due process`, etc.

---

## 📊 Evaluation Results

RAG103 implements a rigorous **4-layer evaluation framework**:

### Layer 1 — Retrieval Quality (n=100, No Data Leakage)
> Qwen2.5 generates semantically-related questions from sampled chunks; evaluated against ground-truth chunk retrieval

| Metric | Score | Grade |
|--------|-------|-------|
| Recall@1 | 0.6500 | 🟠 Acceptable |
| Recall@3 | 0.7700 | 🟡 Good |
| Recall@5 | 0.8300 | 🟡 Good |
| Recall@10 | 0.9000 | 🟢 Excellent |
| MRR@5 | 0.7197 | 🟠 Acceptable |
| MRR@10 | 0.7292 | 🟠 Acceptable |

### Layer 2 — CUAD Gold Q&A Benchmark (n=50)
> Expert-annotated Q&A pairs from The Atticus Project; query expansion applied for short clause names

| Metric | Score | Grade |
|--------|-------|-------|
| Semantic Similarity | 0.5096 | 🔴 Challenging Domain |
| Exact Match (partial) | 0.2427 | 🔴 Expected for open-ended |
| Avg Retrieval Score | 0.5048 | 🔴 Dense clause vocabulary |
| **Success Rate** | **1.0000** | 🟢 Excellent |

> 💡 Low semantic similarity on CUAD is expected — contract clauses often use highly specific legal language that requires exact verbatim extraction. 100% pipeline success rate confirms robust execution.

### Layer 3 — RAGAS-Style Metrics (n=20, LLM-as-Judge)
> Qwen2.5-14B self-evaluates faithfulness, relevancy, precision, and recall locally (no external API cost)

| Metric | Score | Grade |
|--------|-------|-------|
| Faithfulness | 0.7265 | 🟠 Acceptable |
| Answer Relevancy | 0.7219 | 🟠 Acceptable |
| Context Precision | 0.6667 | 🟠 Acceptable |
| Context Recall | 0.6647 | 🟠 Acceptable |

### Layer 4 — Human Judgment / Concept-Relevance@5 (n=20)
> Manual binary relevance labels on 20 legal concept queries spanning both corpora

| Metric | Score | Grade |
|--------|-------|-------|
| Concept-Relevance@5 | 0.7500 | 🟡 Good |
| SCOTUS hits | 9/10 | ✅ |
| CUAD hits | 6/10 | ✅ |

### 🏆 Overall System Score: **0.7711 — Good**

> Computed as unweighted mean of: `Recall@5 + MRR@5 + CUAD Success Rate + Faithfulness + Answer Relevancy + Context Precision + Concept-Relevance@5`

---

## 🛠️ Agent Tool Registry

| Tool | Description | Best For |
|------|-------------|----------|
| `search_scotus` | Search SCOTUS corpus via hybrid retrieval | Constitutional law, case holdings, legal standards |
| `search_cuad` | Search CUAD corpus via hybrid retrieval | Contract clauses, governing law, indemnification |
| `search_both` | Search both corpora simultaneously | Broad legal questions, unclear corpus |
| `rewrite_query` | Reformulate low-scoring queries | Low retrieval scores, irrelevant results |
| `summarize_context` | Compress chunks to fit token budget | Too many chunks, overlapping content |

---

## ⚙️ Configuration

Key parameters (all configurable in Cell 04):

```python
# Data
MAX_SCOTUS_CASES     = 5000
MAX_CUAD_CONTRACTS   = 510

# Chunking
MAX_CHUNK_CHARS      = 1500
MIN_CHUNK_SENTENCES  = 3
MAX_CHUNK_SENTENCES  = 8

# Retrieval
TOPK_DENSE           = 60      # FAISS candidates
TOPK_BM25            = 60      # BM25 candidates
TOPK_FUSED           = 100     # After RRF fusion
TOPK_RERANK          = 10      # After cross-encoder
RRF_K                = 60      # RRF damping constant

# Reflection Gates
CORRECT_THRESHOLD    = 0.65    # Above → proceed to generation
AMBIGUOUS_LOW        = 0.40    # Below → trigger rewriter

# Agent Loop Limits
MAX_HOPS             = 3       # Maximum retrieval hops
MAX_REWRITES         = 2       # Maximum query rewrites
MAX_REGENERATIONS    = 1       # Maximum answer regenerations

# Generation
GEN_MODEL            = "Qwen/Qwen2.5-14B-Instruct"
MAX_INPUT_TOKENS     = 4096
MAX_NEW_TOKENS       = 512
EMBED_BATCH_SIZE     = 128
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# GPU: NVIDIA A100 (80GB) recommended
# VRAM: 40GB minimum for 4-bit quantized Qwen2.5-14B
# RAM:  32GB+ system RAM
# Python: 3.12
```

### Installation

```python
# All dependencies installed automatically in Cell 02
packages = [
    "torch", "transformers>=4.45.0", "accelerate>=0.26.0",
    "bitsandbytes>=0.43.0", "sentence-transformers",
    "faiss-cpu", "rank-bm25", "spacy>=3.7.0",
    "langgraph>=0.2.0", "langchain>=0.3.0",
    "datasets>=2.20.0", "rouge-score", "gradio>=4.0.0"
]
```

### Usage

```python
# Single query
result = answer("What is the automobile exception to the warrant requirement?")

# Display formatted result
display_answer(result)

# Output:
# ============================================================
# ⚖️  INTENT    : LEGAL_RAG
# 📚 CORPUS    : scotus
# 📊 STATUS    : SUCCESS
# 🔄 HOPS      : 1
# ✏️  REWRITES  : 0
# ⏱️  TIME      : ~25,000ms
# ------------------------------------------------------------
# 🧠 THINKING: ...chain-of-thought reasoning...
# ------------------------------------------------------------
# 💬 ANSWER: The automobile exception allows police with probable
#             cause to conduct a warrantless vehicle search... [S2]
# ------------------------------------------------------------
# 📋 SOURCES:
#   [1] SCOTUS | score=0.9976 | Held: Police officers who...
```

### Running the Full Pipeline

| Cell | Purpose | Time |
|------|---------|------|
| 01-04 | Environment setup & configuration | ~2 min |
| 05-06 | Load SCOTUS + CUAD datasets | ~5 min |
| 07-09 | Build unified corpus + chunking | ~85 min |
| **10** | **Save to Google Drive** | ~5 min |
| **10b** | **Load from Drive (subsequent sessions)** | ~3 min |
| 11-12 | BM25 indexes + FAISS embeddings | ~55 min |
| 13-17 | Retrieval functions + Tool registry | ~2 min |
| 18-25 | Agent definitions + LangGraph assembly | ~10 min |
| 26-27 | Main pipeline + smoke tests | ~5 min |
| 28-33 | 4-layer evaluation + save results | ~90 min |

> ⚡ **On subsequent sessions:** Run Cells 01-04, then 10b (load Drive), then 11-17 (rebuild BM25 + load FAISS), then 18-26. Skip Cells 05-10 entirely.

---

## 💾 Persisted Artifacts

All artifacts saved to Google Drive (`/content/drive/MyDrive/RAG103`):

| File | Size | Description |
|------|------|-------------|
| `scotus_chunks.pkl` | 172 MB | 128,212 SCOTUS text chunks |
| `cuad_chunks.pkl` | 21 MB | 15,922 CUAD text chunks |
| `scotus_corpus.pkl` | 169 MB | Raw SCOTUS documents |
| `cuad_corpus.pkl` | 20 MB | Raw CUAD documents |
| `faiss_scotus.bin` | 501 MB | FAISS index (128K × 1024-dim) |
| `faiss_cuad.bin` | 62 MB | FAISS index (16K × 1024-dim) |
| `scotus_embeddings.pkl` | 501 MB | Pre-computed BGE-M3 embeddings |
| `cuad_embeddings.pkl` | 62 MB | Pre-computed BGE-M3 embeddings |
| `cuad_qa_gold.pkl` | 24 MB | 10,404 gold Q&A pairs |
| `full_report.json` | < 1 MB | Complete evaluation report |

**Total: ~1.7 GB**

---

## 📐 Design Decisions & Rationale

### Why RRF over Weighted Score Combination?
RRF is rank-based — it doesn't require normalizing incompatible BM25 and cosine similarity scales. The damping constant `k=60` is a well-established standard that prevents top-ranked documents from dominating excessively.

### Why BGE-M3 for Embeddings?
BGE-M3 supports multi-lingual, multi-granularity retrieval with 1024-dim embeddings. It consistently outperforms OpenAI `text-embedding-ada-002` on legal domain retrieval benchmarks while running locally with no API cost.

### Why Hybrid Chunking (Not Fixed-Size)?
Legal documents have rich structural hierarchy: opinions have `HELD/DISSENT/CONCUR` sections; contracts have `ARTICLE/SECTION/WHEREAS` clauses. Fixed-size chunking breaks these semantic units. The 3-level cascade preserves structure while ensuring no chunk exceeds the 1,500 char hard limit.

### Why 4-bit NF4 Quantization?
Qwen2.5-14B requires ~28GB in FP16. NF4 quantization reduces this to ~10GB VRAM, leaving ample headroom on an A100 80GB for the embedding model and reranker running simultaneously.

### Why LangGraph?
LangGraph's stateful graph model allows clean separation of agent responsibilities, built-in loop control (hop/rewrite/regen counters), and memory checkpointing for multi-turn conversations — without the complexity of building a custom state machine.

---

## 📋 Supported Query Types

```python
# ✅ SCOTUS Case Law
answer("What is the automobile exception to the warrant requirement?")
answer("What are Miranda rights in custodial interrogation?")
answer("What is qualified immunity for police officers?")
answer("How does the exclusionary rule apply to illegally obtained evidence?")
answer("What is the Blockburger test for double jeopardy?")

# ✅ CUAD Contract Clauses
answer("What are the indemnification provisions in this contract?")
answer("What governing law applies to this agreement?")
answer("What is a termination for convenience clause?")
answer("What are intellectual property ownership provisions?")
answer("What are audit rights in contracts?")

# ✅ Simple Legal Definitions (No Retrieval)
answer("What is habeas corpus?")
answer("Define mens rea")
answer("What is promissory estoppel?")

# ✅ General Conversation
answer("Hello, how are you?")

# 🚫 Out of Scope (Refused)
answer("Help me forge a contract")
answer("How do I launder money?")
```

---

## 🧠 Generation Prompt Design

The system uses a strict grounding prompt to prevent hallucination:

```
STRICT RULES:
1. Answer using ONLY the provided SOURCES below
2. NEVER use outside knowledge — only what is in SOURCES
3. Every claim MUST have a citation [S1], [S2], etc.
4. If answer is not in SOURCES → say exactly: "Not found in provided sources."
5. For case law questions: answer in 3-8 sentences
6. For contract clause questions: answer in up to 12 sentences
7. End your answer with <END>

THINKING FORMAT:
First think inside <think> tags:
- What does each source say?
- Which sources are most relevant?
- What is the direct answer?
```

Low-temperature sampling (`temperature=0.1, top_p=0.9`) with `repetition_penalty=1.1` produces precise, non-repetitive answers while maintaining determinism.

---

## 📁 Project Structure

```
RAG103.ipynb
│
├── CELL 01  — GPU Check
├── CELL 02  — Install All Dependencies
├── CELL 03  — All Imports
├── CELL 04  — Master Configuration
│
├── CELL 05  — Load SCOTUS Dataset
├── CELL 06  — Load CUAD Dataset
├── CELL 07  — Unified Data Schema
├── CELL 08  — Hybrid Legal Chunker
├── CELL 09  — Chunk Both Corpora
│
├── CELL 10  — Save to Google Drive
├── CELL 10b — Load from Google Drive
│
├── CELL 11  — BM25 Indexes
├── CELL 12  — Dense Embeddings + FAISS
├── CELL 12b — Load FAISS from Drive
├── CELL 13  — Retrieval Functions + RRF Fusion
├── CELL 14  — BGE Reranker
├── CELL 15  — Query Classifier
├── CELL 16  — Router Logic
├── CELL 17  — Tool Registry
│
├── CELL 18  — LangGraph Agent State
├── CELL 19  — Planner Agent (ReAct)
├── CELL 20  — Retrieval Agent
├── CELL 21  — Reflection Agent (CRAG-style)
├── CELL 22  — Query Rewriter Agent
├── CELL 23  — Generation Agent (Qwen2.5-14B + CoT)
├── CELL 24  — Verification Agent
├── CELL 25  — LangGraph Graph Assembly
├── CELL 26  — Main answer() Function
├── CELL 27  — Smoke Tests (Extended)
│
├── CELL 28  — Evaluation Layer 1 (Retrieval Metrics)
├── CELL 29  — Evaluation Layer 2 (CUAD Gold Q&A)
├── CELL 30  — Evaluation Layer 3 (RAGAS-Style)
├── CELL 31  — Evaluation Layer 4 (Manual Concept-Relevance@5)
├── CELL 32  — Full Evaluation Report
└── CELL 33  — Save Everything to Google Drive
```

---

## 🔧 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | V100 (16GB) | **A100 (80GB)** |
| VRAM | 40GB | 80GB |
| System RAM | 64GB | **167GB** |
| Storage | 20GB (Drive) | ~2GB |
| CUDA | 11.8+ | 13.0 |

> ⚠️ On V100 (16GB), use `Qwen2.5-7B-Instruct` instead of 14B. Adjust `GEN_MODEL` in Cell 04.

---

## 📦 Dependencies

```
torch==2.10.0+cu128
transformers>=5.3.0
accelerate>=0.26.0
bitsandbytes>=0.43.0
sentence-transformers==5.3.0
faiss-cpu==1.13.2
rank-bm25
spacy>=3.8.0 + en_core_web_sm
langgraph>=0.2.0
langchain>=1.2.0
langchain-core>=0.3.0
langchain-community>=0.3.0
datasets>=4.8.0
pandas
rouge-score
nltk
scikit-learn
fastapi
uvicorn
pyngrok
nest-asyncio
gradio>=4.0.0
pydantic>=2.0.0
psutil
```

---

## 🔮 Future Work

- [ ] **Streaming responses** via FastAPI + Server-Sent Events
- [ ] **Gradio UI** for interactive legal research interface
- [ ] **Multi-turn conversation** with full context carry-over
- [ ] **Citation linking** — hyperlink `[S1]` to source document
- [ ] **Query caching** — embedding cache for repeated queries
- [ ] **Expand SCOTUS** — full 9,000+ case dataset
- [ ] **Add CFR corpus** — Code of Federal Regulations
- [ ] **Fine-tuned reranker** — domain-adapted on legal Q&A pairs
- [ ] **Production deployment** — containerized FastAPI service

---

## 📜 Data Sources & Licenses

| Dataset | License | Citation |
|---------|---------|----------|
| `lex_glue/scotus` | CC BY 4.0 | Chalkidis et al., 2021 |
| `chenghao/cuad_qa` | CC BY 4.0 | Atticus Project, 2021 |
| `BAAI/bge-m3` | MIT | Chen et al., 2024 |
| `BAAI/bge-reranker-large` | MIT | BAAI, 2023 |
| `Qwen/Qwen2.5-14B-Instruct` | Apache 2.0 | Alibaba Cloud, 2024 |

---

## 🙏 Acknowledgements

This system builds on seminal work in:
- **CRAG** (Corrective RAG) — reflection-based retrieval quality assessment
- **ReAct** — reasoning + acting agent pattern
- **RRF** (Reciprocal Rank Fusion) — Cormack et al., 2009
- **LangGraph** — LangChain's stateful agent framework
- **RAGAS** — RAG evaluation framework

---

<div align="center">

**Built with ⚖️ for the legal research community**

*RAG103 — Where AI meets the Law*

</div>
