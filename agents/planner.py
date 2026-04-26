"""
PLANNER AGENT (Orchestrator)
============================

Purpose:
This agent converts a messy user question into a structured research plan
that other tools (search, vector DB, python REPL) can execute.

Why this agent exists:
LLMs perform MUCH better when large problems are broken into smaller steps.
This file is the "brain" that decides WHAT to do before tools decide HOW.
"""

# =========================
# Imports
# =========================
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List


# =========================
# 1️⃣ Structured Output Schema
# =========================
# WHY:
# Instead of free-text planning, we force the LLM to return structured JSON.
# This makes downstream automation reliable and removes parsing errors.

class ResearchPlan(BaseModel):
    subtasks: List[str] = []
    requires_code: bool = False
    search_queries: List[str] = []


# =========================
# 2️⃣ Planner Function
# =========================
def run_planner(user_query: str) -> ResearchPlan:
    """
    Convert user query → structured research plan.

    This is the FIRST step in the agent pipeline.
    """

    # ---- LLM Setup ----
    # WHY:
    # We use a reasoning-optimized model to think and plan.
    # with_structured_output() forces JSON output matching ResearchPlan schema.
    llm = ChatGroq(
        model="llama-3.3-70b-versatile"
    ).with_structured_output(ResearchPlan)

    # ---- Prompt Template ----
    # WHY:
    # The system message defines the LLM's role as a planner.
    # This dramatically improves decomposition quality.
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a research planner. "
            "Break the user request into 3–5 clear subtasks. "
            "Also decide if web search or coding is needed."
        ),
        ("human", "{query}")
    ])

    # ---- LCEL Chain ----
    # WHY:
    # prompt | llm  → creates a simple LangChain pipeline
    # invoke() executes the chain and returns structured output.
    plan = (prompt | llm).invoke({"query": user_query})

    return plan


# =========================
# Example test
# =========================
if __name__ == "__main__":
    result = run_planner("Analyze AI job trends and create a salary chart")
    print(result)