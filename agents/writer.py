"""
WRITER AGENT
============

Purpose:
This agent takes structured analysis and writes a polished final report.

Why this agent exists:
The analyst extracts WHAT is true.
The writer decides HOW to communicate it clearly.

These are genuinely different cognitive tasks — merging them into one
agent produces mediocre output at both. Keeping them separate lets each
LLM call focus on one thing and do it well.

Why the writer is the LAST agent in the pipeline:
By this stage, all facts have been verified (researcher) and insights
have been structured (analyst). The writer has everything it needs and
cannot introduce hallucinations — it only reformats real data.
"""

# =========================
# Imports
# =========================
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List

# We import AnalysisResult so the writer receives typed data,
# not raw strings. This enforces the pipeline contract.
# WHY typed input instead of a plain string?
# If the analyst changes its output schema, Python will catch
# the mismatch immediately rather than silently producing bad reports.
from agents.analyst import AnalysisResult


# =========================
# 1️⃣ Structured Output Schema
# =========================
# WHY:
# Even though the writer's job is to produce prose, we still
# wrap the output in a Pydantic model.
#
# This gives us:
# • a machine-readable title (useful for saving/displaying)
# • the report body as a clean string
# • a list of sources (important for credibility)
# • word count for quality checks downstream

class FinalReport(BaseModel):
    # Short descriptive title for the report
    title: str

    # Full markdown-formatted report body
    body: str

    # Key sources cited (extracted from analyst insights)
    sources_cited: List[str]

    # Approximate word count — useful for trimming or expanding later
    word_count: int


# =========================
# 2️⃣ Writer Function
# =========================
def run_writer(
    analysis: AnalysisResult,
    original_query: str,
    subtasks: List[str]
) -> FinalReport:
    """
    Convert structured analysis → polished final report.

    Parameters
    ----------
    analysis : AnalysisResult
        Output from the analyst agent (insights, summary, code, confidence).

    original_query : str
        The user's original question.
        WHY pass this again?
        The writer needs the exact phrasing to write an opening that
        directly answers the user — not just summarizes the research.

    subtasks : List[str]
        The subtasks from the planner's ResearchPlan.
        WHY include these?
        They act as section headers — the writer structures the report
        around the plan, ensuring every subtask is addressed.

    Returns
    -------
    FinalReport
        A fully structured, human-readable research report.
    """

    # =========================
    # Step A: Build Context String
    # =========================
    # WHY pre-format before the prompt?
    # Injecting raw Python objects into prompts produces messy output.
    # We convert the analysis into clean, readable text sections
    # so the LLM receives well-structured context — not dict reprs.

    # Format key insights as a numbered list for the LLM
    insights_text = "\n".join(
        f"{i+1}. {insight}"
        for i, insight in enumerate(analysis.key_insights)
    )

    # Format the planner's subtasks so the writer knows what to cover
    subtasks_text = "\n".join(
        f"- {task}" for task in subtasks
    )

    # Only include code section if something was actually computed
    # WHY conditional inclusion?
    # Injecting an empty "No code output" section wastes tokens
    # and can confuse the LLM into inventing data tables.
    code_section = (
        f"Data / computation results:\n{analysis.code_output}"
        if analysis.code_output
        else ""
    )


    # =========================
    # Step B: LLM Setup
    # =========================
    # WHY claude-sonnet-4-6 again (same as analyst)?
    # Writing requires the same reasoning depth as analysis.
    # We need coherent structure, good prose, and accurate citations.
    # A cheaper model (Haiku) would produce noticeably worse writing.

    llm = ChatGroq(
        model="llama-3.3-70b-versatile"
    ).with_structured_output(FinalReport)


    # =========================
    # Step C: Prompt Construction
    # =========================
    # WHY such a detailed system message?
    # The system message defines the persona and output contract.
    # "Research report writer" → formal, structured, sourced prose.
    # Without this framing, the LLM defaults to a generic chatbot tone.
    #
    # We also explicitly tell it to use markdown — because the report
    # will be rendered in Streamlit / frontend, not printed as plain text.

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a professional research report writer. "
            "Write clear, well-structured markdown reports. "
            "Use the analyst's insights as your source of truth — do not invent facts. "
            "Structure the report with: an introduction, one section per subtask, "
            "a conclusion, and a sources section. "
            "Confidence level from the analyst should be noted in the conclusion."
        ),
        (
            "human",
            "User's original question:\n{query}\n\n"
            "Research subtasks to cover:\n{subtasks}\n\n"
            "Key insights from analyst:\n{insights}\n\n"
            "Overall summary from analyst:\n{summary}\n\n"
            "{code_section}"
            "\nAnalyst confidence: {confidence}\n\n"
            "Write a complete research report now."
        )
    ])


    # =========================
    # Step D: Run the Chain
    # =========================
    # WHY same LCEL pattern as planner and analyst?
    # Consistency across agents makes the codebase easy to maintain.
    # Anyone reading this file can immediately understand the pattern.

    report = (prompt | llm).invoke({
        "query": original_query,
        "subtasks": subtasks_text,
        "insights": insights_text,
        "summary": analysis.summary,
        "code_section": code_section,
        "confidence": analysis.confidence
    })

    return report


# =========================
# Example test
# =========================
if __name__ == "__main__":

    # Simulate analyst output for offline testing
    # WHY not call the real analyst here?
    # We want this file to be testable in isolation,
    # without needing API keys for the full pipeline.
    mock_analysis = AnalysisResult(
        key_insights=[
            "AI engineer salaries range from $120k–$300k in the US.",
            "Demand for ML roles grew 35% year-over-year.",
            "Python and PyTorch are the most sought-after skills.",
            "Google, OpenAI, and Microsoft are the top employers.",
        ],
        summary=(
            "The AI job market is rapidly expanding with strong salary growth "
            "and increasing demand across major tech companies."
        ),
        code_output="Mean salary: $205,000",
        confidence="high"
    )

    mock_subtasks = [
        "Research current AI job market trends",
        "Find salary ranges for AI/ML roles",
        "Identify top employers hiring AI talent",
        "Compute average salary from data"
    ]

    report = run_writer(
        analysis=mock_analysis,
        original_query="Analyze AI job trends and create a salary chart",
        subtasks=mock_subtasks
    )

    print("\n=== Final Report ===")
    print(f"Title: {report.title}")
    print(f"Word Count: {report.word_count}")
    print(f"Sources: {report.sources_cited}")
    print(f"\n{report.body}")
