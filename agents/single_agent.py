"""
SINGLE AGENT (Baseline)
=======================

Purpose:
A naive single-agent baseline that handles the full research task
in one LLM call — no planning, no specialised roles, no tool loops.

Why this exists:
To compare against the multi-agent pipeline. The single agent shows
what a plain LLM produces when given only the user's question and
access to one web search. The delta between the two results reveals
the concrete value of decomposition, specialisation, and iteration.
"""

import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from agents.writer import FinalReport
from tools.web_search import web_search


def run_single_agent(query: str) -> tuple[FinalReport, float]:
    """
    Research and write a report in a single LLM call.

    Performs one web search, then asks the LLM to write a complete
    report from that result — no planner, no analyst, no iteration.

    Returns
    -------
    (FinalReport, elapsed_seconds)
    """

    start = time.time()

    # ---- One web search ----
    # The single agent gets one search pass — no parallelism,
    # no query decomposition. This mirrors what a user would do
    # manually: search once, write from what comes back.
    print(f"[SingleAgent] Searching: {query}")
    raw_results = web_search.invoke(query)

    # ---- One LLM call: research + write ----
    # The LLM must do everything the pipeline does across four
    # specialised agents — in a single prompt.
    llm = ChatGroq(
        model="llama-3.3-70b-versatile"
    ).with_structured_output(FinalReport)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a research assistant. Given a user question and raw web "
            "search results, write a complete, well-structured markdown report. "
            "Include: an introduction, key findings, analysis, conclusion. "
            "Cite only sources present in the search results. "
            "Do not invent facts or statistics not found in the results."
        ),
        (
            "human",
            "User question:\n{query}\n\n"
            "Web search results:\n{results}\n\n"
            "Write a complete research report now."
        )
    ])

    report = (prompt | llm).invoke({
        "query":   query,
        "results": raw_results,
    })

    elapsed = round(time.time() - start, 1)
    return report, elapsed
