
#TavilyClient → performs the actual web search.
#tool → converts this function into a LangChain tool usable by agents.
#os → securely read API key from environment (never hardcode keys).
#typing → helps readability and maintainability (production standard).
from tavily import TavilyClient
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
from typing import List, Dict

# WHY load_dotenv() here?
# os.getenv() only reads variables already in the OS environment.
# load_dotenv() reads the .env file and injects them into os.environ
# so the getenv() call below can find them.
# This must happen BEFORE any os.getenv() call.
load_dotenv()

#Why this section exists?
#Fail fast principle -> If API key missing → crash immediately at startup instead of failing randomly later.
# Security best practice -> Keys must come from environment variables, not source code.
#Debuggability ->Clear error instead of mysterious tool failure.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in environment variables")

#Why global client?
#Creating the client once:
         # avoids recreating object on every tool call
         # improves speed and reduces latency
         # prevents rate-limit overhead
         # important because agents may call tools many times
# This pattern is called a singleton client.
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

#Why a separate helper function?
#Separation of concerns:
       # searching logic ≠ formatting logic
       # easier to test independently
       # easier to upgrade later (ex: switch to JSON output)
#This is a production design pattern.
def _format_results(results: List[Dict]) -> str:
    """Clean and compress Tavily results for LLM consumption."""
    
    # We create a new list instead of modifying raw results to avoid side effects 
    # and maintain immutability, which is a good practice in production code.
    cleaned = []
    

#Why this block?
        #Key reliability + hallucination control decisions:
        # .get() prevents crashes if API response changes.
        #Default fallbacks avoid KeyError.
        #Content truncation (VERY important)
        #LLMs:
        #have token limits
        #get confused by long noisy text
        #Short summaries → better answers + lower cost.
    for r in results:
        title = r.get("title", "No title")
        content = r.get("content", "")[:500]   # truncate long content
        url = r.get("url", "")
        

        #Why structured formatting?
        #LLMs reason better with predictable structure:
             # Title
             # Summary
             # Source
        #This reduces hallucinations dramatically.
        cleaned.append(
            f"Title: {title}\nSummary: {content}\nSource: {url}"
        )
    
    #Why join results?
    #Agents perform better when:
    #each result separated clearly
    #easier for LLM to cite sources
    return "\n\n".join(cleaned)

# Tool definition :
@tool
def web_search(query: str) -> str:

    #Why such a detailed docstring?
    #This is not for humans — it's for the LLM agent.
    #Agents use this text to decide:
    #Should I call the tool?
    #What is it good for?
    #Better docstring = smarter agent behavior.
    """
    Search the web for recent and factual information.

    Use this tool when:
    - The question needs up-to-date info
    - The question involves current events
    - The LLM lacks knowledge

    Input:
        query (str): short search query (max 200 chars)

    Output:
        Formatted search results with title, summary and source URL.
    """
    
    # Why Guardrails?
    #Agents sometimes send:
         #empty strings
         #garbage prompts
         #broken inputs
    #Never trust tool input.
    #This prevents crashes.
    if not query or len(query.strip()) == 0:
        return "Error: Empty search query."
    
    #Why limit query length?
    #Security + cost control:
    #prevents prompt injection attacks
    #prevents extremely long inputs
    #keeps search focused
    #reduces API cost
    query = query.strip()[:200]  # prevent prompt injection / long inputs
    

    #Why try/except?
    #External APIs can fail:
          #network errors
          #rate limits
          #service outages
    #Production tools must never crash the agent.
    #Why these parameters?
         #max_results=5 → balance between context and noise.
         #search_depth="advanced" → better quality results.
    try:
        response = tavily_client.search(
            query=query,
            max_results=5,
            search_depth="advanced"
        )
        

        #Again defensive coding — never assume API shape.
        results = response.get("results", [])

        
        #Agents need explicit feedback instead of empty output.
        if not results:
            return "No relevant results found."
        

        #Separation of concerns again:
        #search → format → return
        return _format_results(results)

    #Why return error instead of raising?
        #Agents cannot handle Python exceptions.
        #They can handle text.
    #So we convert crashes → readable tool output.
    except Exception as e:
        return f"Search tool error: {str(e)}"