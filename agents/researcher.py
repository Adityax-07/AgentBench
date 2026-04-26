"""
RESEARCHER AGENT (ReAct Loop)
=============================

Purpose:
This agent executes the research plan created by the planner.

It can:
• search the internet (Tavily tool)
• search private knowledge (FAISS vector store)

Design philosophy:
Planner decides WHAT to do.
Researcher decides HOW to gather information.
"""

# =========================
# Imports
# =========================
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

# These come from your earlier tools
from tools.web_search import web_search
from tools.vector_store import retrieve


# =========================
# Build Research Agent
# =========================
def build_researcher(vector_store):
    """
    Creates a ReAct-style research agent.

    ReAct = Reason + Act loop
    The LLM can think, choose tools, observe results, repeat.
    """

    # ---- Tool list ----
    # WHY:
    # Agent needs two knowledge sources:
    # 1) Web search → fresh public info
    # 2) Vector store → private/local knowledge (RAG)

    tools = [web_search]

    # Only add RAG retrieval if a vector store was built.
    # Without one the agent falls back to web search only.
    if vector_store is not None:
        tools.append(lambda query: retrieve(query, vector_store))


    # ---- Choose cheaper model ----
    # WHY:
    # Research phase calls tools MANY times.
    # Using a smaller model dramatically reduces cost.
    llm = ChatGroq(model="llama-3.1-8b-instant")


    # ---- Create ReAct agent ----
    # WHY:
    # ReAct agents follow loop:
    # Thought → Action → Observation → repeat → Final Answer
    #
    # This makes tool usage dynamic and autonomous.
    researcher_agent = create_react_agent(
        llm,
        tools=tools
    )

    return researcher_agent


# =========================
# Example test
# =========================
if __name__ == "__main__":
    from tools.vector_store import load_store

    store = load_store()
    researcher = build_researcher(store)

    result = researcher.invoke(
        {"messages": [("user", "What is FAISS and why is it used in RAG?")]}
    )

    print(result)