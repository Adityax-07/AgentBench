"""
Benchmark runner — evaluates both agents on 50 queries and saves results.

Usage (from inside multi-agent-research/):
    python bench_runner.py

Results are saved incrementally to bench_results.json so the run is
resumable if it is interrupted (e.g. due to rate limits).
"""
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.single_agent import run_single_agent
from graph import build_graph
from evaluator import evaluate

# ── 50 benchmark queries ─────────────────────────────────────────────────────
BENCH_QUERIES = [
    # GenAI concepts
    "What is retrieval-augmented generation (RAG) and how does it work?",
    "Explain the attention mechanism in transformer models",
    "What are the key differences between GPT-3 and GPT-4?",
    "How does chain-of-thought prompting improve LLM reasoning?",
    "What is prompt engineering and what are the best practices?",
    # Agentic AI
    "When should I use LangGraph over LangChain for building agents?",
    "What is the ReAct agent framework and how does it work?",
    "How do autonomous AI agents differ from traditional chatbots?",
    "Explain the concept of memory in LLM-based agents",
    "What are function calling and tool use in large language models?",
    # Fine-tuning & training
    "How does LoRA reduce fine-tuning costs for large language models?",
    "What is RLHF and how is it used in LLM alignment?",
    "Explain parameter-efficient fine-tuning (PEFT) techniques",
    "What is instruction tuning and why does it matter?",
    "How does knowledge distillation work in machine learning?",
    # Architectures
    "Explain multi-head self-attention in transformers",
    "How does positional encoding work in transformer models?",
    "What is the encoder-decoder architecture in NLP?",
    "How does the BERT architecture differ from GPT?",
    "What is Mixture of Experts (MoE) in language models?",
    # Retrieval & search
    "How does FAISS work for approximate nearest-neighbour search?",
    "What is the difference between sparse and dense retrieval?",
    "How do vector databases differ from traditional relational databases?",
    "What is semantic chunking and why does it matter for RAG?",
    "Explain the role of embeddings in semantic search",
    # Evaluation & safety
    "How do you evaluate the quality of LLM-generated responses?",
    "What is hallucination in LLMs and how can it be reduced?",
    "What is model alignment and why is it important?",
    "How do you measure relevance and factuality in AI outputs?",
    "What is adversarial prompting and how can it be defended against?",
    # ML fundamentals
    "Explain gradient descent and backpropagation",
    "What is overfitting and how do you prevent it?",
    "What is the softmax function and where is it used?",
    "Explain cross-entropy loss in classification tasks",
    "What is the difference between precision and recall?",
    # Efficient ML
    "What is model quantization and what tradeoffs does it introduce?",
    "How does speculative decoding speed up LLM inference?",
    "What is the context window in LLMs and why does it matter?",
    "How does HuggingFace simplify machine learning workflows?",
    "What is contrastive learning and how is it used in representation learning?",
    # Applied AI
    "How should I decide between RAG and fine-tuning for my use case?",
    "What is data drift in machine learning and how do you detect it?",
    "Explain few-shot and zero-shot learning in modern LLMs",
    "What is semantic similarity and how is it measured?",
    "How do knowledge graphs enhance AI reasoning?",
    # Trends & future
    "What are the main generative AI trends in 2024?",
    "How is agentic AI changing software development workflows?",
    "What is multimodal AI and what are its main applications?",
    "What are the main safety risks in deploying LLMs in production?",
    "What is the future of open-source vs closed-source LLMs?",
]

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results.json")

EVAL_DEFAULTS = {
    "rel": 0.0, "halu": "Possible",
    "coherence": 0.0, "completeness": 0.0, "depth": 0.0,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load() -> dict:
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {"queries": [], "done_set": []}


def _save(data: dict):
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _run_multi(pipeline, query: str) -> tuple[str, float]:
    initial = {
        "query": query,
        "session_id": str(uuid.uuid4()),
        "plan": None,
        "research_text": None,
        "analysis": None,
        "report": None,
    }
    final_state: dict = {}
    for update in pipeline.stream(initial, stream_mode="updates"):
        node_name = next(iter(update))
        if update[node_name]:
            final_state.update(update[node_name])
    rpt = final_state.get("report")
    text = (rpt.body or rpt.title) if rpt else ""
    return text, final_state


def _sleep(sec: float, reason: str = ""):
    label = f" ({reason})" if reason else ""
    print(f"  [wait] Waiting {sec}s{label}...")
    time.sleep(sec)


def _score_to_dict(score) -> dict:
    return {
        "rel":         round(score.relevance, 2),
        "halu":        score.hallucination,
        "coherence":   round(score.coherence, 2),
        "completeness": round(score.completeness, 2),
        "depth":       round(score.depth, 2),
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def run():
    data     = _load()
    done     = set(data.get("done_set", []))
    pipeline = build_graph()

    total   = len(BENCH_QUERIES)
    pending = [q for q in BENCH_QUERIES if q not in done]
    print(f"Benchmark runner: {len(pending)} / {total} queries remaining\n")

    for idx, query in enumerate(BENCH_QUERIES):
        q_num = idx + 1
        if query in done:
            print(f"[{q_num:02d}/{total}] OK skip  {query[:60]}")
            continue

        print(f"\n[{q_num:02d}/{total}] >> {query[:70]}")

        # ── Single agent ──────────────────────────────────────────────────────
        s_text, s_lat = "", 0.0
        try:
            t0 = time.time()
            single_rpt, _ = run_single_agent(query)
            s_lat  = round(time.time() - t0, 1)
            s_text = (single_rpt.body or single_rpt.title) if single_rpt else ""
            print(f"  Single OK  {s_lat}s  {len(s_text.split())} words")
        except Exception as exc:
            print(f"  Single ERR  {exc}")

        _sleep(4, "rate limit")

        # ── Multi-agent pipeline ──────────────────────────────────────────────
        m_text, m_lat = "", 0.0
        try:
            t1 = time.time()
            m_text, _ = _run_multi(pipeline, query)
            m_lat = round(time.time() - t1, 1)
            print(f"  Multi  OK  {m_lat}s  {len(m_text.split())} words")
        except Exception as exc:
            print(f"  Multi  ERR  {exc}")

        _sleep(5, "rate limit")

        # ── Evaluate ──────────────────────────────────────────────────────────
        s_scores = dict(EVAL_DEFAULTS)
        m_scores = {"rel": 0.0, "halu": "No", "coherence": 0.0, "completeness": 0.0, "depth": 0.0}

        if s_text:
            try:
                s_ev = evaluate(query, s_text)
                s_scores = _score_to_dict(s_ev)
                print(f"  Eval S  rel={s_scores['rel']}  halu={s_scores['halu']}  "
                      f"coh={s_scores['coherence']}  comp={s_scores['completeness']}  depth={s_scores['depth']}")
            except Exception as exc:
                print(f"  Eval S ERR  {exc}")
            _sleep(3, "eval rate limit")

        if m_text:
            try:
                m_ev = evaluate(query, m_text)
                m_scores = _score_to_dict(m_ev)
                print(f"  Eval M  rel={m_scores['rel']}  halu={m_scores['halu']}  "
                      f"coh={m_scores['coherence']}  comp={m_scores['completeness']}  depth={m_scores['depth']}")
            except Exception as exc:
                print(f"  Eval M ERR  {exc}")

        # ── Save incrementally ────────────────────────────────────────────────
        data["queries"].append({
            "query":  query,
            "single": {"text": s_text[:600], "lat": s_lat, **s_scores},
            "multi":  {"text": m_text[:600], "lat": m_lat, **m_scores},
        })
        done.add(query)
        data["done_set"] = list(done)
        _save(data)

        _sleep(6, "between queries")

    # ── Compute aggregate summary ─────────────────────────────────────────────
    qs = data["queries"]
    if qs:
        def avg(key, agent):
            vals = [q[agent].get(key, 0) for q in qs if q[agent].get(key, 0)]
            return round(sum(vals) / len(vals), 3) if vals else 0

        def halu_rate(agent):
            bad = sum(1 for q in qs if q[agent].get("halu", "No") != "No")
            return round(bad / len(qs), 3)

        def success_rate(agent):
            ok = sum(1 for q in qs if q[agent].get("rel", 0) >= 0.70)
            return round(ok / len(qs), 3)

        data["summary"] = {
            "total":          len(qs),
            "s_avg_rel":      avg("rel", "single"),
            "m_avg_rel":      avg("rel", "multi"),
            "s_avg_lat":      avg("lat", "single"),
            "m_avg_lat":      avg("lat", "multi"),
            "s_halu_rate":    halu_rate("single"),
            "m_halu_rate":    halu_rate("multi"),
            "s_success":      success_rate("single"),
            "m_success":      success_rate("multi"),
            "s_avg_coherence":    avg("coherence", "single"),
            "m_avg_coherence":    avg("coherence", "multi"),
            "s_avg_completeness": avg("completeness", "single"),
            "m_avg_completeness": avg("completeness", "multi"),
            "s_avg_depth":        avg("depth", "single"),
            "m_avg_depth":        avg("depth", "multi"),
        }
        _save(data)

    print("\n\nBenchmark complete!")
    print(json.dumps(data.get("summary", {}), indent=2))


if __name__ == "__main__":
    run()
