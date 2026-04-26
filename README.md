# Multi-Agent Research Assistant

A Streamlit web app that answers research queries using two approaches side-by-side: a single ReAct agent and a multi-agent pipeline built with LangGraph. Includes a real LLM-as-judge benchmark evaluated across 50 queries on 5 metrics.

---

## Architecture

```
User Query
    │
    ├── Single Agent (ReAct)
    │       └── Groq llama-3.3-70b  +  Tavily web search  +  FAISS retrieval
    │
    └── Multi-Agent Pipeline (LangGraph)
            ├── Planner   → breaks query into subtasks + search queries
            ├── Researcher → runs web searches per subtask
            ├── Analyst   → synthesizes findings, flags confidence
            ├── Writer    → produces structured markdown report
            └── Memory    → persists session to SQLite
```

**Stack:** Python · Streamlit · LangGraph · LangChain · Groq (llama-3.3-70b-versatile, llama-3.1-8b-instant) · Tavily Search · FAISS · HuggingFace Sentence Transformers · SQLiteDict

---

## Features

- Live animated UI with real-time pipeline progress bar and node status
- Side-by-side comparison: single agent vs multi-agent pipeline
- Benchmark panel with real evaluation metrics (not hardcoded)
- Session memory via SQLite — agents remember prior conversations
- RAG-ready: plug in a FAISS vector store for domain-specific retrieval
- Resumable benchmark runner — saves results incrementally to JSON

---

## Benchmark Results

Evaluated on **50 queries** across GenAI, ML fundamentals, agentic AI, retrieval, and applied AI topics. Judged by an LLM-as-judge (`llama-3.1-8b-instant`) on 5 metrics.

| Metric | Single Agent | Multi-Agent |
|---|---|---|
| Avg Relevance | 0.877 | 0.850 |
| Avg Coherence | **0.912** | 0.900 |
| Avg Completeness | **0.798** | 0.720 |
| Avg Depth | **0.658** | 0.590 |
| Hallucination Rate | 24% | **4%** |
| Success Rate (rel >= 0.70) | 78% | 78% |
| Avg Latency | **5.5s** | 307s |

**Key finding:** The multi-agent pipeline reduces hallucination by 6× (24% → 4%) at the cost of higher latency. The planner→researcher→analyst chain forces grounding before writing, making it significantly more trustworthy for production use cases. Single agent wins on speed and raw quality metrics for straightforward queries.

---

## Project Structure

```
multi-agent-research/
├── app.py                  # Streamlit UI (Live Query + Benchmark panels)
├── graph.py                # LangGraph pipeline definition
├── evaluator.py            # LLM-as-judge evaluator (5 metrics)
├── bench_runner.py         # Benchmark runner (resumable, 50 queries)
├── bench_results.json      # Saved benchmark results
├── agents/
│   ├── planner.py          # Structured query planning (Pydantic + llama-3.3-70b)
│   ├── researcher.py       # Web search + FAISS retrieval (llama-3.1-8b)
│   ├── analyst.py          # Research synthesis and confidence scoring
│   ├── writer.py           # Final report generation
│   └── single_agent.py     # ReAct single agent baseline
├── tools/
│   ├── web_search.py       # Tavily search wrapper
│   ├── vector_store.py     # FAISS store (all-MiniLM-L6-v2 embeddings)
│   └── python_repl.py      # Python REPL tool for code queries
├── memory/
│   └── store.py            # SQLiteDict session memory
└── PROBLEMS_FACED.md       # 15 real engineering problems faced while building
```

---

## Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Fill in: GROQ_API_KEY, TAVILY_API_KEY

# 3. Run the app
streamlit run app.py

# 4. (Optional) Run the benchmark
python bench_runner.py
```

**.env variables:**
```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

---

## Evaluation Metrics

The LLM-as-judge evaluator scores each response on:

| Metric | Description |
|---|---|
| **Relevance** | How directly the response answers the query (0–1) |
| **Hallucination** | No / Possible / Yes — whether claims are grounded |
| **Coherence** | Logical structure, flow, and readability (0–1) |
| **Completeness** | Coverage of all key aspects of the query (0–1) |
| **Depth** | Technical depth — explains how/why, not just what (0–1) |

---

## Engineering Notes

See [PROBLEMS_FACED.md](PROBLEMS_FACED.md) for 15 real engineering problems encountered during development — including LangGraph state serialization, Groq rate limit handling, Streamlit session state resets, and more. Useful for interview prep.
