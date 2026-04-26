"""
LLM-as-judge evaluator.
Scores a (query, response) pair on 5 metrics:
  relevance, hallucination, coherence, completeness, depth
"""
import time
from functools import lru_cache
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


class EvalScore(BaseModel):
    relevance: float = Field(..., ge=0.0, le=1.0,
        description="0-1 how well the response answers the query")
    hallucination: str = Field(...,
        description="Exactly one of: No, Possible, Yes")
    coherence: float = Field(..., ge=0.0, le=1.0,
        description="0-1 how logically structured and easy to follow the response is")
    completeness: float = Field(..., ge=0.0, le=1.0,
        description="0-1 how thoroughly the response covers all key aspects of the query")
    depth: float = Field(..., ge=0.0, le=1.0,
        description="0-1 technical depth and quality of explanation")


_SYSTEM = """You are a strict, neutral evaluator for AI-generated research responses.

Score the response on five criteria:

RELEVANCE (float 0.0-1.0):
  0.90-1.00  Comprehensive, accurate, directly addresses every aspect of the query
  0.70-0.89  Good coverage; one or two aspects missing or underdeveloped
  0.50-0.69  Partial answer; significant gaps or tangential content
  0.30-0.49  Mostly off-topic, very incomplete, or significant factual issues
  0.00-0.29  Irrelevant or almost entirely wrong

HALLUCINATION (string, exactly one of: No / Possible / Yes):
  No        All claims are grounded; nothing fabricated or confidently wrong
  Possible  A few uncertain or unverifiable statements but no clear fabrications
  Yes       Clear fabrications, invented statistics, or confident wrong facts

COHERENCE (float 0.0-1.0):
  0.90-1.00  Logically structured, flows naturally, headers/sections used well
  0.70-0.89  Mostly clear; minor structural or flow issues
  0.50-0.69  Disorganized or hard to follow in places
  0.00-0.49  Chaotic, repetitive, or incoherent

COMPLETENESS (float 0.0-1.0):
  0.90-1.00  All key sub-topics and nuances addressed
  0.70-0.89  Most aspects covered; 1-2 important points missing
  0.50-0.69  Covers the basics but misses significant aspects
  0.00-0.49  Major gaps; leaves core parts of the query unanswered

DEPTH (float 0.0-1.0):
  0.90-1.00  Expert-level explanation with mechanisms, tradeoffs, and examples
  0.70-0.89  Good technical detail; could go deeper in 1-2 areas
  0.50-0.69  Surface-level; explains what but not how or why
  0.00-0.49  Shallow, generic, or lacking any technical substance

Return ONLY valid JSON with keys: relevance, hallucination, coherence, completeness, depth. No commentary."""

_HUMAN = """Query: {query}

Response (truncated to 1800 chars):
{response}"""


@lru_cache(maxsize=1)
def _get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.0).with_structured_output(EvalScore)


def evaluate(query: str, response: str, retries: int = 3) -> EvalScore:
    """Score a single (query, response) pair. Retries on rate-limit errors."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human",  _HUMAN),
    ])
    chain = prompt | _get_llm()
    for attempt in range(retries):
        try:
            return chain.invoke({"query": query, "response": response[:1800]})
        except Exception as exc:
            if attempt < retries - 1 and ("rate" in str(exc).lower() or "429" in str(exc)):
                wait = 20 * (attempt + 1)
                print(f"  [eval] Rate limit hit, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("evaluate() failed after retries")
