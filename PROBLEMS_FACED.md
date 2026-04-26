# Problems Faced While Building This Project
### Multi-Agent Research Pipeline — Interview Reference

---

## 1. GROQ RATE LIMIT (6,000 TPM on Free Tier)

**Problem:**  
The Groq free tier allows only 6,000 tokens per minute. The multi-agent pipeline runs 3–5 search queries back-to-back inside the researcher node. When queries were fired in parallel (or even sequentially without delay), the API returned `RateLimitError` mid-pipeline, silently killing the run.

**What happened:**  
The first run would succeed, but the second or third query would hit the rate wall. The error surfaced deep inside a LangGraph node with no graceful recovery.

**Fix:**  
- Made the researcher process queries **sequentially** (not in parallel), giving time between calls.
- Added a **retry loop with `time.sleep(25)`** — 25 seconds is enough for the token bucket to refill.
- Wrapped each search in `try/except RateLimitError` with 3 max attempts before giving up.

**Interview talking point:**  
Rate limiting taught me the difference between throughput and latency trade-offs in agentic systems. Parallel calls maximize throughput in paid tiers but destroy reliability on free tiers. Sequential with retry is slower but resilient.

---

## 2. LANGGRAPH STATE — NON-SERIALIZABLE OBJECTS

**Problem:**  
LangGraph passes state between nodes as a serializable dictionary (TypedDict). I initially tried to put the FAISS vector store and the SQLite MemoryStore directly into the pipeline state so each node could access them. Both are live objects (open file handles, in-memory indexes) — they cannot be serialized into JSON/dict.

**What happened:**  
LangGraph threw serialization errors when trying to pass the objects through state transitions.

**Fix:**  
Used the **factory/closure pattern**:
```python
def make_researcher_node(researcher_agent):
    def researcher_node(state):
        # researcher_agent is captured at graph-build time
        ...
    return researcher_node
```
The expensive objects are loaded once at `build_graph()` time and captured via closure. State only carries serializable data (strings, Pydantic models).

**Interview talking point:**  
This taught me the architectural principle: *shared resources that can't be serialized should be injected at initialization, not passed through runtime state.* Same concept as dependency injection in backend frameworks.

---

## 3. FAISS VECTOR STORE — COLD START COST

**Problem:**  
Loading the HuggingFace embedding model (`all-MiniLM-L6-v2`) and rebuilding the FAISS index takes 5–15 seconds. If `load_store()` was called inside the researcher node, it would rebuild the index **on every single pipeline run** — unacceptable latency.

**Fix:**  
- `load_store()` is called once inside `build_graph()`.
- `build_graph()` itself is wrapped with `@lru_cache(maxsize=1)` and decorated with `@st.cache_resource` in the Streamlit layer.
- The FAISS object is built once, held in memory, and reused across all pipeline runs.
- If no FAISS store exists on disk, the agent **gracefully degrades** to web-search-only mode instead of crashing.

**Interview talking point:**  
This is the lazy initialization + singleton pattern applied to ML models. Critical for production RAG systems — you never want embedding model initialization in the hot path.

---

## 4. LANGGRAPH `stream_mode="updates"` — FORMAT CONFUSION

**Problem:**  
LangGraph's `.stream()` method returns different shapes depending on the `stream_mode`. With `stream_mode="updates"`, each yielded item is a dict shaped as `{node_name: node_output_dict}` — not just the output directly.

**What happened:**  
Initially treated the streamed object as raw state, causing `KeyError` when trying to read `update["report"]` — the key was actually `update["writer"]["report"]`.

**Fix:**  
```python
for update in pipeline.stream(initial, stream_mode="updates"):
    node_name = next(iter(update))   # e.g. "planner"
    node_data = update[node_name]    # the actual output dict
    if node_data:
        final_state.update(node_data)
```

**Interview talking point:**  
LangGraph's streaming API is powerful but requires understanding which `stream_mode` gives you what shape. `"values"` gives full state snapshots; `"updates"` gives only the delta from each node — much more efficient for large states.

---

## 5. STREAMLIT — SILENT QUERY LOSS ON RERUN

**Problem:**  
When a user clicked a suggestion pill button, the query was stored in a local variable `chosen_sug`. When the user then clicked "Run", Streamlit triggered a full page rerun — resetting all local variables to their defaults. So `chosen_sug` was `""` again, `query.strip()` was falsy, and the pipeline silently never ran. No error. No output. Nothing.

**What happened:**  
The `if run_clicked and query.strip():` condition was never True because the query value didn't survive the rerun.

**Fix:**  
Replaced the local variable with `st.session_state` + widget `key`:
```python
query = st.text_area("...", key="query_input")
# suggestion pill:
st.session_state.query_input = sug  # persists across reruns
```
Session state is the only memory that survives a Streamlit rerun.

**Interview talking point:**  
Streamlit's execution model reruns the entire script top-to-bottom on every interaction. Any value that needs to persist across reruns must live in `st.session_state` — this is Streamlit's fundamental design constraint.

---

## 6. STREAMLIT — `StreamlitAPIException` WHEN SETTING WIDGET STATE

**Problem:**  
The Clear button handler tried to reset the text area by doing:
```python
if clear_clicked:
    st.session_state["query_input"] = ""  # ← crash
```
But the `st.text_area(key="query_input")` widget had **already been rendered** above this point in the script. Streamlit forbids setting a widget's session state key after the widget has rendered in the same rerun.

**What happened:**  
`StreamlitAPIException: Values for the widget with key "query_input" can't be set using st.session_state...`

**Fix:**  
Introduced a **flag-based deferred reset** pattern:
```python
# On clear click — don't touch widget state directly
st.session_state["_reset_query"] = True
st.rerun()

# At the VERY TOP of panel_live(), before any widget renders:
if st.session_state.pop("_reset_query", False):
    st.session_state.query_input = ""
```
The actual state mutation happens on the *next* rerun, *before* the widget is drawn — which is allowed.

**Interview talking point:**  
Streamlit processes state mutations for widget keys only before the widget renders. This is a known footgun. The fix requires thinking in terms of *which rerun* a state change belongs to, not which line of code it's on.

---

## 7. STALE CODE BEING SERVED (PORT CONFLICT)

**Problem:**  
After fixing the pipeline bugs, the app appeared unchanged in the browser. All fixes seemed to have no effect.

**Root cause:**  
An old Streamlit server process was still running on port 8501, serving the pre-fix version of the code. The new server started successfully on a different port, but the browser was hitting the old one.

**Fix:**  
- Used `netstat -ano | findstr "8501"` to identify the stale PID.
- Killed it with `cmd /c "taskkill /PID <pid> /F"` (bash `kill` alone didn't work on Windows).
- Started fresh: `python -m streamlit run app.py --server.port 8505`.

**Interview talking point:**  
Always verify *which* server process is handling your requests, especially during debugging. Hot-reload doesn't help if the process itself is the stale one.

---

## 8. REACT AGENT — LAMBDA TOOL NOT RECOGNIZED

**Problem:**  
To pass the FAISS retrieval function as a tool to the researcher agent, I initially wrote:
```python
tools.append(lambda query: retrieve(query, vector_store))
```
LangChain's `create_react_agent` requires tools to have a proper `.name` and `.description` attribute. The raw lambda had neither, causing the agent to either ignore the tool or throw a tool-parsing error.

**Fix:**  
Wrapped the lambda properly using `@tool` decorator or ensured the tool was a fully formed `StructuredTool`. The fallback approach used was to only append the tool when `vector_store is not None` and let the named `retrieve` function carry the metadata.

**Interview talking point:**  
LangChain tools must be properly typed and named — the LLM uses tool names and descriptions to decide which one to call. A nameless lambda breaks the tool-selection loop.

---

## 9. OPTIONAL STATE FIELDS — `KeyError` IN DOWNSTREAM NODES

**Problem:**  
LangGraph nodes run sequentially. Early nodes (planner) don't populate fields that later nodes (writer) need. If `Optional` wasn't used on those fields, accessing `state["analysis"]` in the writer node before the analyst ran would raise a `KeyError`.

**Fix:**  
```python
class PipelineState(TypedDict):
    query: str
    session_id: str
    plan: Optional[ResearchPlan]       # None until planner runs
    research_text: Optional[str]       # None until researcher runs
    analysis: Optional[AnalysisResult] # None until analyst runs
    report: Optional[FinalReport]      # None until writer runs
```
Initialize all `Optional` fields to `None` in the initial state dict passed to `pipeline.invoke()`.

**Interview talking point:**  
In sequential multi-agent pipelines, you must design state to be valid at *every stage*, not just at the end. `Optional` fields model partial pipeline state cleanly without using sentinel values or try/except blocks.

---

## 10. BUTTON TEXT TRUNCATION IN STREAMLIT

**Problem:**  
Suggestion pills with long text (e.g. "Explain the attention mechanism in transformers") were being cut off inside Streamlit buttons. Only the first ~20 characters were visible, with no ellipsis — text just disappeared.

**Root cause:**  
Streamlit's default button CSS has `white-space: nowrap` and `overflow: hidden`, which clips text to a single line.

**Fix:**  
CSS injection via `st.markdown(unsafe_allow_html=True)`:
```css
div[data-testid="stButton"] > button {
    white-space: normal !important;
    word-break: break-word !important;
    height: auto !important;
    line-height: 1.45 !important;
}
```

**Interview talking point:**  
Streamlit's component CSS is set at the framework level and can only be overridden with `!important` injection. This works but is fragile across Streamlit version upgrades — a real trade-off between control and maintainability.

---

## 11. STRUCTURED OUTPUT SCHEMA — PYDANTIC VS TYPEDDICT

**Problem:**  
When using `.with_structured_output()` on the Groq LLM, the schema needed to be either a Pydantic model or a TypedDict. Initially used plain Python `dataclass` — Groq's structured output call silently returned unstructured text instead of the expected object.

**Fix:**  
Switched all output schemas to Pydantic `BaseModel`:
```python
class FinalReport(BaseModel):
    title: str
    body: str
    sources_cited: list[str]
    word_count: int
```
Pydantic models give LangChain/Groq the JSON schema they need to constrain generation.

**Interview talking point:**  
Structured output with LLMs works by injecting a JSON schema into the system prompt and then parsing the response. If your schema type isn't supported, the LLM generates free text and the parser silently fails or throws. Always use Pydantic or TypedDict — not dataclasses.

---

## 12. HALLUCINATION IN SINGLE AGENT (BASELINE)

**Problem:**  
The single-agent baseline, given only one web search result, frequently invented citations, fabricated statistics, and produced confident-sounding false claims. There was no mechanism to verify claims against sources.

**Observation:**  
Multi-agent pipeline reduced hallucination because:
1. The researcher fetches multiple independent sources.
2. The analyst explicitly cross-references claims against retrieved text.
3. The writer is instructed to cite only sources present in the analyst's output.

**Interview talking point:**  
This is the concrete demonstration of why multi-agent decomposition matters: specialization + verification reduces hallucination. A single LLM call cannot simultaneously retrieve, verify, and synthesize — it shortcuts to hallucination under information pressure.

---

## 13. `@lru_cache` ON A FUNCTION WITH SIDE EFFECTS

**Problem:**  
`build_graph()` does three expensive things: loads a HuggingFace model, loads a FAISS index, and opens a SQLite connection. Without caching, every Streamlit rerun (triggered by any button click) would rebuild the entire graph.

**Fix:**  
```python
@lru_cache(maxsize=1)
def build_graph():
    ...
```
`maxsize=1` means only one result is cached. Since `build_graph()` takes no arguments, this effectively makes it a singleton.

**Gotcha:**  
`@lru_cache` doesn't work with mutable arguments. If any argument to the cached function is unhashable (like a list or dict), it throws `TypeError`. The fix is to only cache argument-free or hashable-argument functions.

**Interview talking point:**  
`@lru_cache` is Python's built-in memoization. For expensive initialization code (model loading, DB connections), it's the simplest way to implement a singleton without a class. But it hides side effects — if the underlying resource changes (FAISS store updated), the cache returns the stale object.

---

## 14. FILE CLEANUP — FINDING DEAD CODE SAFELY

**Problem:**  
The project accumulated files from multiple iterations: `api.py` (FastAPI), `agentbench.py` (old dashboard), `app_backup.py`, `eval/`, `evaluators/`, `Dockerfile`. Deleting them without checking for imports would break the app silently.

**Process:**  
Used grep to verify nothing active imported them before deletion:
```bash
grep -r "from api\|import api\|agentbench\|app_backup\|evaluators" --include="*.py" .
```
Only deleted files that returned zero matches.

**Interview talking point:**  
Dead code removal requires static dependency analysis, not just intuition. In Python, `grep` for import statements is the fastest way to verify a file is truly orphaned. More robust approaches use tools like `vulture` or `importlab`.

---

## 15. PROGRESS BAR — STATIC HTML VS LIVE PLACEHOLDER

**Problem:**  
The initial progress bar was a static HTML div rendered once. It showed 0% for the entire pipeline duration and then disappeared. Updating it required either JavaScript or a Streamlit-native approach.

**Fix:**  
Used `st.empty()` — a Streamlit placeholder that can be overwritten:
```python
prog_ph = st.empty()
# Inside each node callback:
prog_ph.markdown(f'<div class="ab-pbar" style="width:{pct}%"></div>', unsafe_allow_html=True)
# After completion:
prog_ph.empty()
```
Node-to-progress mapping: `planner→22%`, `researcher→46%`, `analyst→68%`, `writer→88%`, `memory→100%`.

**Interview talking point:**  
`st.empty()` is Streamlit's escape hatch for dynamic content. It reserves a slot in the layout that can be overwritten at any time during script execution — the foundation for live streaming UIs in Streamlit.

---

## Summary Table

| # | Problem | Category | Root Cause | Fix |
|---|---------|----------|------------|-----|
| 1 | Groq rate limit | LLM API | 6000 TPM cap | Sequential + retry backoff |
| 2 | Non-serializable state | LangGraph | FAISS/SQLite in state dict | Factory/closure injection |
| 3 | FAISS cold start | Performance | Rebuild on every run | `@lru_cache` + disk persistence |
| 4 | `stream_mode` format | LangGraph | Wrong update shape assumed | `next(iter(update))` pattern |
| 5 | Silent query loss | Streamlit | Local var reset on rerun | `st.session_state` + widget key |
| 6 | Widget state exception | Streamlit | Setting key after render | `_reset_query` flag + deferred reset |
| 7 | Stale server | DevOps | Port conflict, old process | `taskkill` + fresh port |
| 8 | Lambda tool failure | LangChain | Missing `.name`/`.description` | Named tool / `@tool` decorator |
| 9 | `KeyError` in nodes | LangGraph | Missing Optional fields | `Optional[T]` + `None` initialization |
| 10 | Button text clipping | Streamlit CSS | `white-space: nowrap` default | CSS injection with `!important` |
| 11 | Structured output failure | LLM | dataclass not supported | Switch to Pydantic BaseModel |
| 12 | Hallucination (baseline) | LLM quality | Single LLM, no verification | Multi-agent decomposition |
| 13 | Graph rebuilt on every rerun | Performance | No caching | `@lru_cache(maxsize=1)` |
| 14 | Unsafe file deletion | Code hygiene | Unknown dependencies | grep import analysis first |
| 15 | Static progress bar | Streamlit | HTML rendered once | `st.empty()` placeholder |
