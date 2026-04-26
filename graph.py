"""
RESEARCH PIPELINE GRAPH
========================

Purpose:
This file wires all four agents into a single executable pipeline
using LangGraph's StateGraph.

Why LangGraph?
LangGraph is built specifically for multi-agent workflows.
It gives us:
• typed shared state that flows between nodes
• a clear visual graph of the pipeline
• easy debugging (each node is isolated and testable)
• future support for branching, loops, and parallel nodes

Pipeline order:
User Query
    ↓
[Planner]    → breaks query into subtasks + search queries
    ↓
[Researcher] → executes searches, retrieves documents
    ↓
[Analyst]    → extracts key insights, optionally runs code
    ↓
[Writer]     → produces final polished markdown report
    ↓
[Memory]     → saves session to SQLite for future recall
    ↓
Final Report
"""

# =========================
# Imports
# =========================
import time
import uuid
from functools import lru_cache
from typing import Optional
from typing_extensions import TypedDict

from groq import RateLimitError

from langgraph.graph import StateGraph, END

# All four agents
from agents.planner import run_planner, ResearchPlan
from agents.researcher import build_researcher
from agents.analyst import run_analyst, AnalysisResult
from agents.writer import run_writer, FinalReport

# Persistent memory
from memory.store import MemoryStore

# Vector store loader
# WHY import here and not inside researcher?
# The FAISS store must be loaded once at startup and passed into
# the researcher. Loading it inside a node would rebuild it on
# every single run — slow and wasteful.
from tools.vector_store import load_store


# =========================
# 1️⃣ Shared Pipeline State
# =========================
# WHY TypedDict?
# LangGraph requires a typed state schema.
# Each node reads from and writes to this shared dict.
# TypedDict gives us type hints without the overhead of Pydantic.
#
# WHY Optional on most fields?
# At the start of the pipeline, only `query` and `session_id` exist.
# Each subsequent node fills in its field.
# Optional prevents KeyError crashes on partial state.

class PipelineState(TypedDict):
    # Set by the caller before the graph runs
    query: str
    session_id: str

    # Filled by planner node
    plan: Optional[ResearchPlan]

    # Filled by researcher node
    research_text: Optional[str]

    # Filled by analyst node
    analysis: Optional[AnalysisResult]

    # Filled by writer node
    report: Optional[FinalReport]


# =========================
# 2️⃣ Node Definitions
# =========================
# WHY wrap each agent in a node function?
# LangGraph nodes must:
# • accept the full state dict
# • return only the fields they changed
#
# This keeps each node focused on its own output
# and avoids overwriting other nodes' work.


# ---- Node 1: Planner ----
def planner_node(state: PipelineState) -> dict:
    """
    Convert user query → structured research plan.

    Reads:  state["query"]
    Writes: state["plan"]
    """

    print(f"\n[Planner] Planning for: {state['query']}")

    plan = run_planner(state["query"])

    print(f"[Planner] Subtasks: {plan.subtasks}")
    print(f"[Planner] Search queries: {plan.search_queries}")
    print(f"[Planner] Requires code: {plan.requires_code}")

    # WHY return only {"plan": ...}?
    # LangGraph merges the returned dict into the existing state.
    # Returning the full state would overwrite everything.
    return {"plan": plan}


# ---- Node 2: Researcher ----
def make_researcher_node(researcher_agent):
    """
    Factory function that closes over the pre-built researcher agent.

    WHY a factory instead of a plain function?
    The researcher agent requires a vector store injected at startup.
    We can't pass it as state (not serializable).
    A closure captures it once at graph-build time — clean and efficient.
    """

    def researcher_node(state: PipelineState) -> dict:
        """
        Execute each search query and collect raw research text.

        Reads:  state["plan"]
        Writes: state["research_text"]
        """

        plan = state["plan"]

        # WHY sequential instead of parallel?
        # Groq's free tier has a 6000 TPM limit. Running all queries in
        # parallel bursts them simultaneously and hits the rate limit.
        # Sequential execution spreads token usage over time.
        def _search(query: str) -> str:
            print(f"\n[Researcher] Searching: {query}")
            for attempt in range(3):
                try:
                    result = researcher_agent.invoke({"messages": [("user", query)]})
                    return f"Query: {query}\n{result['messages'][-1].content}"
                except RateLimitError:
                    if attempt == 2:
                        return f"Query: {query}\nRate limit exceeded after retries."
                    wait = 25
                    print(f"[Researcher] Rate limit hit, waiting {wait}s...")
                    time.sleep(wait)
                except Exception as e:
                    return f"Query: {query}\nSearch failed: {e}"

        results = []
        for q in plan.search_queries:
            results.append(_search(q))

        combined = "\n\n---\n\n".join(results)

        return {"research_text": combined}

    return researcher_node


# ---- Node 3: Analyst ----
def analyst_node(state: PipelineState) -> dict:
    """
    Extract structured insights from raw research.

    Reads:  state["research_text"], state["plan"], state["query"]
    Writes: state["analysis"]
    """

    print("\n[Analyst] Analyzing research...")

    plan = state["plan"]

    analysis = run_analyst(
        research_text=state["research_text"],
        original_query=state["query"],
        requires_code=plan.requires_code,

        # WHY pass an empty code_task here?
        # The analyst generates its own code if requires_code=True.
        # We don't pre-write code — the LLM decides what to compute
        # based on the research it receives.
        code_task=""
    )

    print(f"[Analyst] Confidence: {analysis.confidence}")
    print(f"[Analyst] Insights found: {len(analysis.key_insights)}")

    return {"analysis": analysis}


# ---- Node 4: Writer ----
def writer_node(state: PipelineState) -> dict:
    """
    Write the final polished report from structured analysis.

    Reads:  state["analysis"], state["plan"], state["query"]
    Writes: state["report"]
    """

    print("\n[Writer] Writing final report...")

    report = run_writer(
        analysis=state["analysis"],
        original_query=state["query"],
        subtasks=state["plan"].subtasks
    )

    print(f"[Writer] Report title: {report.title}")
    print(f"[Writer] Word count: {report.word_count}")

    return {"report": report}


# ---- Node 5: Memory ----
def make_memory_node(memory_store: MemoryStore):
    """
    Factory that closes over the MemoryStore instance.

    WHY a factory (same reason as researcher)?
    MemoryStore holds a database connection — not serializable into state.
    We capture it once at graph-build time via closure.
    """

    def memory_node(state: PipelineState) -> dict:
        """
        Persist the completed session to SQLite.

        Reads:  state["session_id"], state["query"], state["report"]
        Writes: nothing (side-effect only — saves to disk)
        """

        print("\n[Memory] Saving session...")

        memory_store.save_session(
            session_id=state["session_id"],
            query=state["query"],
            report=state["report"].body
        )

        print(f"[Memory] Session saved: {state['session_id']}")

        # WHY return empty dict?
        # This node only persists data — it doesn't change pipeline state.
        # Returning {} tells LangGraph: state is unchanged.
        return {}

    return memory_node


# =========================
# 3️⃣ Graph Builder
# =========================

@lru_cache(maxsize=1)
def build_graph():
    """
    Assemble and compile the full research pipeline graph.

    WHY a function instead of module-level code?
    Wrapping in a function means:
    • graph is only built when first called (lazy init)
    • easier to test (call build_graph() in tests)
    • avoids side effects (file I/O, API calls) on import
    """

    # ---- Load shared resources ----
    # WHY load these here, not inside nodes?
    # Both are expensive to initialize (disk I/O, model loading).
    # Loading once at graph-build time means every pipeline run
    # reuses the same instances — fast and efficient.

    print("[Graph] Loading vector store...")
    vector_store = load_store()

    print("[Graph] Initializing memory store...")
    memory = MemoryStore()

    # ---- Build agents ----
    researcher_agent = build_researcher(vector_store)

    # ---- Create graph ----
    # WHY StateGraph(PipelineState)?
    # This tells LangGraph what schema the shared state follows.
    # It validates state transitions and enables type checking.
    graph = StateGraph(PipelineState)

    # ---- Register nodes ----
    # WHY string names?
    # LangGraph uses string names to identify nodes in edges.
    # These names also appear in debug output and graph visualizations.
    graph.add_node("planner",    planner_node)
    graph.add_node("researcher", make_researcher_node(researcher_agent))
    graph.add_node("analyst",    analyst_node)
    graph.add_node("writer",     writer_node)
    graph.add_node("memory",     make_memory_node(memory))

    # ---- Define edges (execution order) ----
    # WHY linear edges here?
    # Each stage depends on the previous stage's output.
    # The pipeline is sequential by design:
    # you cannot analyze before you research.
    graph.set_entry_point("planner")
    graph.add_edge("planner",    "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst",    "writer")
    graph.add_edge("writer",     "memory")
    graph.add_edge("memory",     END)

    # ---- Compile ----
    # WHY compile()?
    # compile() validates the graph (no missing nodes, no dead edges)
    # and returns an executable runnable object.
    return graph.compile()


# =========================
# 4️⃣ Public Run Function
# =========================

def run_pipeline(query: str, session_id: str = None) -> FinalReport:
    """
    Execute the full research pipeline for a user query.

    Parameters
    ----------
    query : str
        The user's research question.

    session_id : str, optional
        Unique ID for this session. Auto-generated if not provided.
        WHY optional?
        Callers (Streamlit, API, CLI) may want to supply their own
        session IDs (e.g. user account ID). But standalone runs
        should still work without needing to manage IDs.

    Returns
    -------
    FinalReport
        The completed report with title, body, sources, word count.
    """

    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    print(f"\n{'='*50}")
    print(f"Starting research pipeline")
    print(f"Session: {session_id}")
    print(f"Query: {query}")
    print(f"{'='*50}")

    # Build and compile the graph
    pipeline = build_graph()

    # Run the pipeline with initial state
    # WHY invoke() instead of stream()?
    # invoke() waits for the full result — simpler for synchronous callers.
    # stream() can be used later for real-time progress in Streamlit.
    final_state = pipeline.invoke({
        "query": query,
        "session_id": session_id,
        "plan": None,
        "research_text": None,
        "analysis": None,
        "report": None,
    })

    print(f"\n{'='*50}")
    print("Pipeline complete.")
    print(f"{'='*50}\n")

    return final_state["report"]


# =========================
# Example test (CLI)
# =========================
if __name__ == "__main__":
    report = run_pipeline(
        query="Analyze AI job trends and salary ranges in 2024"
    )

    print("\n=== FINAL REPORT ===")
    print(f"Title: {report.title}")
    print(f"Word Count: {report.word_count}")
    print(f"Sources: {report.sources_cited}")
    print(f"\n{report.body}")
