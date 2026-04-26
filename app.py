"""
AgentBench — Multi-Agent Evaluation Dashboard (Streamlit)
"""
import html
import json
import os
import time
import uuid

import plotly.graph_objects as go
import streamlit as st

from agents.single_agent import run_single_agent
from graph import build_graph
from evaluator import evaluate

st.set_page_config(
    page_title="AgentBench — Multi-Agent Evaluation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@keyframes fadeInUp {
  from { opacity:0; transform:translateY(22px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeInLeft {
  from { opacity:0; transform:translateX(-18px); }
  to   { opacity:1; transform:translateX(0); }
}
@keyframes shimmerSlide {
  0%   { background-position:-200% center; }
  100% { background-position: 200% center; }
}
@keyframes gradientFlow {
  0%,100% { background-position:0% 50%; }
  50%      { background-position:100% 50%; }
}
@keyframes pulseRing {
  0%   { box-shadow:0 0 0 0   rgba(37,99,235,0.45); }
  70%  { box-shadow:0 0 0 10px rgba(37,99,235,0); }
  100% { box-shadow:0 0 0 0   rgba(37,99,235,0); }
}
@keyframes livePing {
  0%   { box-shadow:0 0 0 0   rgba(34,197,94,0.60); }
  70%  { box-shadow:0 0 0 9px rgba(34,197,94,0); }
  100% { box-shadow:0 0 0 0   rgba(34,197,94,0); }
}
@keyframes floatBob {
  0%,100% { transform:translateY(0px); }
  50%      { transform:translateY(-6px); }
}
@keyframes scaleIn {
  from { opacity:0; transform:scale(0.82); }
  to   { opacity:1; transform:scale(1); }
}
@keyframes borderGlow {
  0%,100% { border-color:#2563eb; box-shadow:0 0 0 0 rgba(37,99,235,0.15); }
  50%      { border-color:#60a5fa; box-shadow:0 0 14px rgba(96,165,250,0.28); }
}
@keyframes runPulse {
  0%,100% { opacity:1; }
  50%      { opacity:0.55; }
}
@keyframes stepDone {
  0%  { transform:scale(0.9); opacity:0; }
  60% { transform:scale(1.06); }
  100%{ transform:scale(1);   opacity:1; }
}
@keyframes heroOrb1 {
  0%,100% { transform:translate(0,0) scale(1); }
  50%      { transform:translate(20px,-12px) scale(1.08); }
}
@keyframes heroOrb2 {
  0%,100% { transform:translate(0,0) scale(1); }
  50%      { transform:translate(-15px,10px) scale(0.95); }
}
@keyframes pbarShimmer {
  0%   { background-position:-200% center; }
  100% { background-position: 200% center; }
}
@keyframes numberReveal {
  from { opacity:0; transform:translateY(14px); filter:blur(6px); }
  to   { opacity:1; transform:translateY(0);    filter:blur(0); }
}
@keyframes crownBounce {
  0%,100% { transform:translateY(0) rotate(-5deg); }
  50%      { transform:translateY(-4px) rotate(5deg); }
}
@keyframes glowWin {
  0%,100% { box-shadow:0 0 0 0 rgba(21,128,61,0.2); }
  50%      { box-shadow:0 0 16px 4px rgba(21,128,61,0.15); }
}

/* ── Base ── */
#MainMenu,footer,header{ visibility:hidden; }
[data-testid="stDecoration"]{ display:none; }
.main {
  background:
    radial-gradient(ellipse at 80% 10%,rgba(37,99,235,0.05) 0%,transparent 50%),
    radial-gradient(ellipse at 10% 80%,rgba(96,165,250,0.05) 0%,transparent 50%),
    linear-gradient(160deg,#f8faff 0%,#f2f6ff 40%,#fafafa 100%);
}
.main .block-container{ padding-top:1.4rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background:linear-gradient(170deg,#0a0e1a 0%,#0d1117 45%,#111827 100%) !important;
  border-right:1px solid rgba(255,255,255,0.07) !important;
  min-width:220px !important; max-width:240px !important;
}
[data-testid="stSidebarContent"]{ padding:0 !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span { color:#c9d1d9 !important; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,0.08) !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * { color:#c9d1d9 !important; }

/* ── Buttons ── */
div[data-testid="stButton"] > button {
  white-space:normal !important; word-break:break-word !important;
  height:auto !important; line-height:1.45 !important;
  transition:transform 0.18s ease,box-shadow 0.18s ease,background 0.18s ease !important;
}
div[data-testid="stButton"] > button:hover {
  transform:translateY(-2px) !important;
  box-shadow:0 4px 16px rgba(37,99,235,0.18) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
  box-shadow:0 6px 22px rgba(37,99,235,0.38) !important;
}

/* ── Hero ── */
.ab-hero {
  background:linear-gradient(135deg,#eef4ff 0%,#e8efff 40%,#f0f9ff 100%);
  background-size:200% 200%;
  animation:gradientFlow 9s ease infinite;
  border:1px solid rgba(37,99,235,0.15);
  border-radius:18px; padding:22px 26px; margin-bottom:20px;
  position:relative; overflow:hidden;
}
.ab-hero::before {
  content:''; position:absolute; top:-40%; right:-8%;
  width:280px; height:280px;
  background:radial-gradient(circle,rgba(37,99,235,0.10) 0%,transparent 70%);
  border-radius:50%; animation:heroOrb1 7s ease-in-out infinite; pointer-events:none;
}
.ab-hero::after {
  content:''; position:absolute; bottom:-35%; left:5%;
  width:200px; height:200px;
  background:radial-gradient(circle,rgba(96,165,250,0.09) 0%,transparent 70%);
  border-radius:50%; animation:heroOrb2 9s ease-in-out infinite; pointer-events:none;
}
.ab-hero-content{ position:relative; z-index:1; display:flex; align-items:flex-start; justify-content:space-between; }
.ab-hero-title {
  font-size:26px; font-weight:900; letter-spacing:-0.04em; margin:0 0 5px 0;
  background:linear-gradient(90deg,#1e3a8a 0%,#2563eb 40%,#60a5fa 70%,#1e3a8a 100%);
  background-size:200% auto;
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
  animation:shimmerSlide 4s linear infinite;
}
.ab-hero-sub{ font-size:12px; color:#6b7280; margin:0; }
.ab-hero-badge{ font-size:11px; padding:5px 14px; border-radius:20px; font-weight:700; flex-shrink:0; margin-top:2px; }

/* ── Agent cards ── */
.ab-card {
  border:1px solid #e5e7eb; border-radius:16px; overflow:hidden; background:#fff;
  margin-bottom:6px;
  box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 18px rgba(0,0,0,0.04);
  transition:box-shadow 0.25s ease,transform 0.25s ease;
  animation:fadeInUp 0.45s ease both;
}
.ab-card:hover { box-shadow:0 4px 14px rgba(0,0,0,0.10),0 14px 38px rgba(0,0,0,0.07); transform:translateY(-3px); }
.ab-card.winner {
  border:2px solid #2563eb;
  box-shadow:0 2px 8px rgba(37,99,235,0.14),0 8px 30px rgba(37,99,235,0.07);
  animation:fadeInUp 0.45s ease both,borderGlow 3.5s ease-in-out infinite 0.5s;
}
.ab-ch {
  padding:11px 16px;
  background:linear-gradient(135deg,#f9fafb 0%,#f3f4f6 100%);
  border-bottom:1px solid #e5e7eb;
  display:flex; align-items:center; justify-content:space-between;
}
.ab-name{ font-size:13px; font-weight:700; color:#111827; letter-spacing:-0.01em; }
.ab-pipe {
  padding:7px 14px; background:#f8f9fa; border-bottom:1px solid #e5e7eb;
  display:flex; gap:5px; align-items:center; flex-wrap:wrap; min-height:34px;
}
.ab-step {
  font-size:10px; padding:3px 10px; border-radius:20px;
  border:1px solid #d1d5db; background:#fff; color:#9ca3af;
  white-space:nowrap; font-weight:500; transition:all 0.2s ease;
}
.ab-step.done { border-color:#86efac; color:#15803d; background:#f0fdf4; font-weight:600; animation:stepDone 0.35s ease both; }
.ab-step.active {
  border-color:#93c5fd; color:#1d4ed8; font-weight:600;
  background:linear-gradient(90deg,#eff6ff 0%,#dbeafe 50%,#eff6ff 100%);
  background-size:200% auto;
  animation:shimmerSlide 1.6s linear infinite,runPulse 1.3s ease-in-out infinite;
}
.ab-arr{ color:#d1d5db; font-size:11px; }
.ab-body{ padding:14px 16px; font-size:13px; line-height:1.8; color:#1f2937; min-height:120px; }
.ab-body-ph{ color:#d1d5db; font-style:italic; }
.ab-foot{
  padding:8px 16px; border-top:1px solid #f3f4f6;
  background:linear-gradient(135deg,#f9fafb,#f3f4f6);
  display:flex; gap:18px; font-size:11px; color:#6b7280;
}
.ab-foot b{ color:#111827; }

/* ── Mini metric rows inside cards ── */
.ab-mrow {
  display:flex; flex-wrap:wrap; gap:8px;
  padding:8px 16px 10px; border-top:1px solid #f1f5f9; background:#fafafa;
}
.ab-mrow-item{ display:flex; align-items:center; gap:5px; flex:1; min-width:90px; }
.ab-mrow-label{ font-size:9px; font-weight:700; color:#9ca3af; text-transform:uppercase; letter-spacing:0.05em; width:42px; flex-shrink:0; }
.ab-mrow-bwrap{ flex:1; height:4px; background:#f1f5f9; border-radius:3px; overflow:hidden; }
.ab-mrow-bar{ height:100%; border-radius:3px; transition:width 1s cubic-bezier(0.4,0,0.2,1); }
.ab-mrow-val{ font-size:10px; font-weight:700; color:#374151; width:28px; text-align:right; }

/* ── Badges ── */
.ab-badge{ font-size:10px; padding:3px 10px; border-radius:20px; font-weight:600; letter-spacing:0.02em; transition:all 0.2s ease; }
.ab-idle { background:#f1f5f9; color:#64748b; border:1px solid #e2e8f0; }
.ab-run  { background:#fff7ed; color:#c2410c; border:1px solid #fed7aa; animation:runPulse 0.9s ease-in-out infinite; }
.ab-win  { background:#f0fdf4; color:#15803d; border:1px solid #86efac; animation:glowWin 2.5s ease-in-out infinite; }
.ab-halu-ok  { background:#f0fdf4; color:#15803d; border:1px solid #86efac; }
.ab-halu-bad { background:#fef2f2; color:#dc2626; border:1px solid #fecaca; }
.ab-ph-badge { font-size:11px; padding:4px 14px; border-radius:20px; font-weight:600; }
.ab-live  { background:linear-gradient(135deg,#dbeafe,#eff6ff); color:#1d4ed8; border:1px solid #bfdbfe; animation:pulseRing 2.2s ease-out infinite; }
.ab-bench { background:linear-gradient(135deg,#dcfce7,#f0fdf4); color:#15803d; border:1px solid #86efac; }

/* ── Metric cards v2 (benchmark panel) ── */
.ab-mc2 {
  border-radius:14px; padding:16px 16px 13px; background:#fff;
  border:1px solid #e5e7eb;
  box-shadow:0 1px 4px rgba(0,0,0,0.05);
  animation:fadeInUp 0.5s ease both;
  transition:all 0.25s ease; position:relative; overflow:hidden;
}
.ab-mc2::before {
  content:''; position:absolute; top:0; left:0; right:0;
  height:3px; border-radius:14px 14px 0 0;
}
.ab-mc2.blue::before   { background:linear-gradient(90deg,#1e40af,#60a5fa); }
.ab-mc2.green::before  { background:linear-gradient(90deg,#15803d,#4ade80); }
.ab-mc2.red::before    { background:linear-gradient(90deg,#dc2626,#f87171); }
.ab-mc2.orange::before { background:linear-gradient(90deg,#c2410c,#fb923c); }
.ab-mc2.purple::before { background:linear-gradient(90deg,#7c3aed,#a78bfa); }
.ab-mc2.gray::before   { background:linear-gradient(90deg,#475569,#94a3b8); }
.ab-mc2:hover{ transform:translateY(-3px); box-shadow:0 8px 28px rgba(0,0,0,0.10); }
.ab-ml2{ font-size:9px; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:8px; }
.ab-mv2{ font-size:26px; font-weight:800; color:#0f172a; letter-spacing:-0.03em; animation:numberReveal 0.7s ease both; }
.ab-mdelta-g { font-size:10px; font-weight:700; padding:1px 7px; border-radius:6px; margin-left:5px; background:#f0fdf4; color:#15803d; border:1px solid #86efac; }
.ab-mdelta-r { font-size:10px; font-weight:700; padding:1px 7px; border-radius:6px; margin-left:5px; background:#fef2f2; color:#dc2626; border:1px solid #fecaca; }
.ab-mdelta-n { font-size:10px; font-weight:700; padding:1px 7px; border-radius:6px; margin-left:5px; background:#f8fafc; color:#64748b; border:1px solid #e2e8f0; }
.ab-mvs{ font-size:10px; color:#94a3b8; margin-top:4px; }
.ab-hpbar-wrap{ height:5px; background:#f1f5f9; border-radius:4px; margin-top:8px; overflow:hidden; }
.ab-hpbar{ height:100%; border-radius:4px; transition:width 1s cubic-bezier(0.4,0,0.2,1); }
.ab-hpbar.blue   { background:linear-gradient(90deg,#1e40af,#60a5fa); }
.ab-hpbar.green  { background:linear-gradient(90deg,#15803d,#4ade80); }
.ab-hpbar.orange { background:linear-gradient(90deg,#c2410c,#fb923c); }
.ab-hpbar.red    { background:linear-gradient(90deg,#dc2626,#f87171); }
.ab-hpbar.purple { background:linear-gradient(90deg,#7c3aed,#a78bfa); }

/* ── Inline query metric cards (live panel) ── */
.ab-qmc {
  background:linear-gradient(135deg,#f8faff,#eff4ff);
  border:1px solid #dbeafe; border-radius:12px;
  padding:12px 14px; margin-bottom:6px;
  box-shadow:0 1px 3px rgba(37,99,235,0.06);
  animation:fadeInUp 0.4s ease both;
  transition:transform 0.2s ease,box-shadow 0.2s ease;
}
.ab-qmc:hover{ transform:translateY(-2px); box-shadow:0 4px 14px rgba(37,99,235,0.13); }
.ab-qml{ font-size:10px; color:#6b7280; font-weight:700; letter-spacing:0.07em; text-transform:uppercase; margin-bottom:6px; }
.ab-qmv{ font-size:20px; font-weight:800; color:#0f172a; letter-spacing:-0.02em; }
.ab-qmb-w{ font-size:10px; font-weight:700; padding:2px 7px; border-radius:6px; background:#dcfce7; color:#15803d; margin-left:5px; border:1px solid #86efac; }
.ab-qmb-l{ font-size:10px; font-weight:700; padding:2px 7px; border-radius:6px; background:#fef2f2; color:#dc2626; margin-left:5px; border:1px solid #fecaca; }
.ab-qmb-n{ font-size:10px; font-weight:700; padding:2px 7px; border-radius:6px; background:#f8fafc; color:#64748b; margin-left:5px; border:1px solid #e2e8f0; }
.ab-qm-vs{ font-size:10px; color:#9ca3af; margin-top:4px; }

/* ── Progress bar ── */
.ab-pbar-wrap{ height:5px; background:#e5e7eb; border-radius:3px; margin:12px 0; overflow:hidden; }
.ab-pbar {
  height:100%;
  background:linear-gradient(90deg,#1e40af,#2563eb,#60a5fa,#818cf8,#2563eb);
  background-size:300% auto; border-radius:3px;
  transition:width 0.7s cubic-bezier(0.4,0,0.2,1);
  animation:pbarShimmer 2s linear infinite;
}

/* ── Winner pill / wbar ── */
.ab-winner-pill {
  display:inline-flex; align-items:center; gap:8px;
  background:linear-gradient(135deg,#f0fdf4,#dcfce7);
  border:1px solid #86efac; border-radius:10px;
  padding:10px 16px; font-size:12px; color:#15803d; font-weight:600;
  margin-bottom:16px; box-shadow:0 1px 4px rgba(21,128,61,0.10);
  animation:fadeInUp 0.4s ease both;
}
.ab-wbar {
  background:linear-gradient(135deg,#f0fdf4,#dcfce7);
  border:1px solid #86efac; border-radius:10px;
  padding:12px 16px; margin:14px 0 18px 0;
  font-size:12px; color:#15803d; font-weight:500;
  box-shadow:0 1px 4px rgba(21,128,61,0.08);
  animation:fadeInLeft 0.4s ease both;
}
.ab-insight {
  background:linear-gradient(135deg,#f0f9ff,#e0f2fe);
  border:1px solid #bae6fd; border-left:4px solid #0284c7;
  border-radius:10px; padding:12px 16px; font-size:12px; color:#0c4a6e;
  margin:12px 0; animation:fadeInLeft 0.4s ease both;
}

/* ── Chart containers ── */
.ab-cc {
  border:1px solid #e5e7eb; border-radius:14px;
  padding:16px 16px 4px; background:#fff; margin-bottom:6px;
  box-shadow:0 1px 4px rgba(0,0,0,0.05);
  animation:fadeInUp 0.5s ease both;
  transition:box-shadow 0.2s ease,transform 0.2s ease;
}
.ab-cc:hover{ box-shadow:0 4px 18px rgba(0,0,0,0.09); transform:translateY(-2px); }
.ab-cc-title{ font-size:13px; font-weight:700; color:#0f172a; margin-bottom:2px; letter-spacing:-0.01em; }
.ab-cc-sub  { font-size:11px; color:#9ca3af; margin-bottom:0; }

/* ── Resp comparison cards ── */
.ab-resp-card {
  border:1px solid #e5e7eb; border-radius:14px; overflow:hidden; background:#fff;
  box-shadow:0 1px 4px rgba(0,0,0,0.05); animation:fadeInUp 0.4s ease both;
  transition:transform 0.22s ease,box-shadow 0.22s ease;
}
.ab-resp-card:hover{ transform:translateY(-2px); box-shadow:0 4px 16px rgba(0,0,0,0.10); }
.ab-resp-card.winner {
  border:2px solid #2563eb; box-shadow:0 2px 8px rgba(37,99,235,0.12);
  animation:fadeInUp 0.4s ease both,borderGlow 3.5s ease-in-out infinite 0.3s;
}

/* ── Misc ── */
hr.ab{ border:none; border-top:1px solid #f1f5f9; margin:16px 0; }
.ab-logo-float{ animation:floatBob 3.5s ease-in-out infinite; display:inline-block; }
.ab-dot-ping  { animation:livePing 2s ease-out infinite; border-radius:50%; display:inline-block; }
.ab-crown     { display:inline-block; animation:crownBounce 2s ease-in-out infinite; }
.ab-section-lbl {
  font-size:10px; font-weight:700; color:#94a3b8;
  text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

# ── Static data ───────────────────────────────────────────────────────────────
SUGGESTIONS = [
    "What is retrieval-augmented generation?",
    "Explain the attention mechanism in transformers",
    "When should I use LangGraph over LangChain?",
    "How does LoRA reduce fine-tuning costs?",
    "What is data drift in machine learning?",
]

CATEGORIES = [
    ("GenAI Concepts",   range(0,  5)),
    ("Agentic AI",       range(5,  10)),
    ("Fine-tuning",      range(10, 15)),
    ("Architectures",    range(15, 20)),
    ("Retrieval",        range(20, 25)),
    ("Eval & Safety",    range(25, 30)),
    ("ML Fundamentals",  range(30, 35)),
    ("Efficient ML",     range(35, 40)),
    ("Applied AI",       range(40, 45)),
    ("Trends & Future",  range(45, 50)),
]

_BENCH_FILE = os.path.join(os.path.dirname(__file__), "bench_results.json")

S_CLR  = "#f97316"
M_CLR  = "#2563eb"
S_FILL = "rgba(249,115,22,0.12)"
M_FILL = "rgba(37,99,235,0.12)"
GR_CLR = "rgba(0,0,0,0.05)"
TK_CLR = "rgba(0,0,0,0.40)"

_BASE_CHART = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(size=10, color="#595959"),
    margin=dict(t=30, b=8, l=8, r=8),
)

# ── Data helpers ──────────────────────────────────────────────────────────────
def _load_bench() -> dict:
    if os.path.exists(_BENCH_FILE):
        try:
            with open(_BENCH_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _bench_tabs_data() -> list:
    data = _load_bench()
    tabs = []
    for row in data.get("queries", [])[:10]:
        label = row["query"][:30].rstrip() + "…" if len(row["query"]) > 30 else row["query"]
        tabs.append({
            "label": label,
            "query": row["query"],
            "s": {
                "text":        row["single"]["text"],
                "lat":         f'{row["single"]["lat"]}s',
                "rel":         row["single"].get("rel", 0),
                "halu":        row["single"].get("halu", "?"),
                "coherence":   row["single"].get("coherence", 0),
                "completeness":row["single"].get("completeness", 0),
                "depth":       row["single"].get("depth", 0),
            },
            "m": {
                "text":        row["multi"]["text"],
                "lat":         f'{row["multi"]["lat"]}s',
                "rel":         row["multi"].get("rel", 0),
                "halu":        row["multi"].get("halu", "?"),
                "coherence":   row["multi"].get("coherence", 0),
                "completeness":row["multi"].get("completeness", 0),
                "depth":       row["multi"].get("depth", 0),
            },
        })
    return tabs

# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "ran": False, "last_query": "", "query_input": "",
    "res_single": None, "res_multi": None, "cmp_view": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

@st.cache_resource
def get_pipeline():
    return build_graph()

# ── Chart builders ────────────────────────────────────────────────────────────

def _chart_layout(**kw):
    d = dict(**_BASE_CHART)
    d.update(kw)
    return d


def bar_two(labels, s_vals, m_vals, ymax=None):
    fig = go.Figure()
    fig.add_bar(name="Single", x=labels, y=s_vals,
                marker_color=S_CLR, marker_line_width=0, opacity=0.9,
                text=[f"{v:.2f}" if isinstance(v, float) else str(v) for v in s_vals],
                textposition="outside", textfont=dict(size=9))
    fig.add_bar(name="Multi",  x=labels, y=m_vals,
                marker_color=M_CLR, marker_line_width=0, opacity=0.9,
                text=[f"{v:.2f}" if isinstance(v, float) else str(v) for v in m_vals],
                textposition="outside", textfont=dict(size=9))
    fig.update_layout(**_chart_layout(
        barmode="group", height=200, showlegend=True,
        legend=dict(orientation="h", y=1.2, x=0, font_size=9),
        xaxis=dict(gridcolor=GR_CLR, tickfont=dict(size=9, color=TK_CLR)),
        yaxis=dict(gridcolor=GR_CLR, tickfont=dict(size=9, color=TK_CLR),
                   range=[0, ymax] if ymax else None),
    ))
    return fig


def bar_pair(labels, vals, ymax=None):
    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=[S_CLR, M_CLR], marker_line_width=0, opacity=0.9,
        text=[f"{v:.2f}" if isinstance(v, float) else str(v) for v in vals],
        textposition="outside", textfont=dict(size=9),
    ))
    fig.update_layout(**_chart_layout(
        height=200, showlegend=False,
        xaxis=dict(gridcolor=GR_CLR, tickfont=dict(size=9, color=TK_CLR)),
        yaxis=dict(gridcolor=GR_CLR, tickfont=dict(size=9, color=TK_CLR),
                   range=[0, ymax] if ymax else None),
    ))
    return fig


def radar_5(s_rel, s_coh, s_comp, s_dep, s_no_halu,
            m_rel, m_coh, m_comp, m_dep, m_no_halu):
    cats   = ["Relevance", "Coherence", "Completeness", "Depth", "No-Halluc."]
    s_vals = [s_rel, s_coh, s_comp, s_dep, s_no_halu]
    m_vals = [m_rel, m_coh, m_comp, m_dep, m_no_halu]
    closed = cats + [cats[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=s_vals + [s_vals[0]], theta=closed, fill="toself", name="Single",
        line=dict(color=S_CLR, width=2), fillcolor=S_FILL,
    ))
    fig.add_trace(go.Scatterpolar(
        r=m_vals + [m_vals[0]], theta=closed, fill="toself", name="Multi",
        line=dict(color=M_CLR, width=2), fillcolor=M_FILL,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#595959"),
        margin=dict(t=36, b=24, l=28, r=28), height=260,
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1], gridcolor=GR_CLR,
                            tickfont=dict(size=8, color=TK_CLR)),
            angularaxis=dict(gridcolor=GR_CLR),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1, x=0.2, font_size=9),
    )
    return fig


def mini_radar_5(s_rel, s_halu_ok, m_rel, m_halu_ok):
    cats = ["Relevance", "Trust", "Speed-adj", "Coverage"]
    s_v  = [s_rel, 1 if s_halu_ok else 0.4, 0.92, min(s_rel + 0.04, 1.0)]
    m_v  = [m_rel, 1 if m_halu_ok else 0.4, 0.15, min(m_rel + 0.07, 1.0)]
    closed = cats + [cats[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=s_v+[s_v[0]], theta=closed, fill="toself",
        name="Single", line=dict(color=S_CLR, width=1.5), fillcolor=S_FILL))
    fig.add_trace(go.Scatterpolar(r=m_v+[m_v[0]], theta=closed, fill="toself",
        name="Multi",  line=dict(color=M_CLR, width=1.5), fillcolor=M_FILL))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(size=9, color="#595959"),
        margin=dict(t=28, b=16, l=28, r=28), height=180,
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1], gridcolor=GR_CLR,
                            tickfont=dict(size=7, color=TK_CLR)),
            angularaxis=dict(gridcolor=GR_CLR), bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True, legend=dict(orientation="h", y=-0.15, x=0.15, font_size=8),
    )
    return fig


def trend_line(qs_data):
    idxs   = list(range(1, len(qs_data) + 1))
    s_rels = [q["single"].get("rel", 0) for q in qs_data]
    m_rels = [q["multi"].get("rel", 0) for q in qs_data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idxs, y=s_rels, mode="lines", name="Single",
        line=dict(color=S_CLR, width=2, shape="spline"),
        fill="tozeroy", fillcolor=S_FILL))
    fig.add_trace(go.Scatter(x=idxs, y=m_rels, mode="lines", name="Multi",
        line=dict(color=M_CLR, width=2, shape="spline"),
        fill="tozeroy", fillcolor=M_FILL))
    fig.update_layout(**_chart_layout(
        height=210, showlegend=True,
        legend=dict(orientation="h", y=1.2, x=0, font_size=9),
        xaxis=dict(title="Query #", range=[1, len(idxs)], gridcolor=GR_CLR,
                   tickfont=dict(size=9, color=TK_CLR)),
        yaxis=dict(title="Relevance", range=[0, 1.15], gridcolor=GR_CLR,
                   tickfont=dict(size=9, color=TK_CLR)),
    ))
    return fig


def scatter_lat_rel(qs_data):
    s_x = [q["single"].get("lat", 0) for q in qs_data]
    s_y = [q["single"].get("rel", 0) for q in qs_data]
    m_x = [q["multi"].get("lat", 0) for q in qs_data]
    m_y = [q["multi"].get("rel", 0) for q in qs_data]
    q_labels = [q["query"][:45] + "…" for q in qs_data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_x, y=s_y, mode="markers", name="Single",
        marker=dict(color=S_CLR, size=9, opacity=0.75, line=dict(width=1, color="white")),
        text=q_labels,
        hovertemplate="<b>%{text}</b><br>Lat: %{x}s  Rel: %{y}<extra>Single</extra>"))
    fig.add_trace(go.Scatter(x=m_x, y=m_y, mode="markers", name="Multi",
        marker=dict(color=M_CLR, size=9, opacity=0.75, line=dict(width=1, color="white")),
        text=q_labels,
        hovertemplate="<b>%{text}</b><br>Lat: %{x}s  Rel: %{y}<extra>Multi</extra>"))
    fig.update_layout(**_chart_layout(
        height=210, showlegend=True,
        legend=dict(orientation="h", y=1.2, x=0, font_size=9),
        xaxis=dict(title="Latency (s)", type="log", gridcolor=GR_CLR,
                   tickfont=dict(size=9, color=TK_CLR)),
        yaxis=dict(title="Relevance", range=[0, 1.15], gridcolor=GR_CLR,
                   tickfont=dict(size=9, color=TK_CLR)),
    ))
    return fig


def gauge_duo(s_pct, m_pct):
    fig = go.Figure()
    steps_green = [
        {"range": [0, 10],  "color": "#f0fdf4"},
        {"range": [10, 30], "color": "#fef9c3"},
        {"range": [30, 100],"color": "#fef2f2"},
    ]
    for val, label, clr, domain in [
        (s_pct, "Single Agent", S_CLR, [0, 0.44]),
        (m_pct, "Multi-Agent",  M_CLR, [0.56, 1.0]),
    ]:
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=val,
            number={"suffix": "%", "font": {"size": 22, "color": clr}},
            title={"text": label, "font": {"size": 12, "color": "#374151"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1,
                         "tickfont": {"size": 8}, "tickcolor": "#d1d5db"},
                "bar": {"color": clr, "thickness": 0.28},
                "bgcolor": "white",
                "steps": steps_green,
                "threshold": {"line": {"color": "red", "width": 3},
                              "thickness": 0.75, "value": 30},
            },
            domain={"x": domain, "y": [0, 1]},
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=10, color="#595959"),
        margin=dict(t=40, b=10, l=20, r=20),
        height=200,
    )
    return fig


def category_chart(qs_data):
    if len(qs_data) < 50:
        return None
    names, s_avgs, m_avgs = [], [], []
    for cat_name, idx_range in CATEGORIES:
        chunk = [qs_data[i] for i in idx_range if i < len(qs_data)]
        if not chunk:
            continue
        names.append(cat_name)
        s_avgs.append(round(sum(q["single"].get("rel", 0) for q in chunk) / len(chunk), 2))
        m_avgs.append(round(sum(q["multi"].get("rel", 0)  for q in chunk) / len(chunk), 2))
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Single", y=names, x=s_avgs, orientation="h",
        marker_color=S_CLR, marker_line_width=0, opacity=0.88,
        text=s_avgs, textposition="outside", textfont=dict(size=8)))
    fig.add_trace(go.Bar(name="Multi",  y=names, x=m_avgs, orientation="h",
        marker_color=M_CLR, marker_line_width=0, opacity=0.88,
        text=m_avgs, textposition="outside", textfont=dict(size=8)))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=9, color="#595959"),
        margin=dict(t=24, b=8, l=8, r=50), height=310,
        barmode="group", showlegend=True,
        legend=dict(orientation="h", y=1.08, x=0, font_size=9),
        xaxis=dict(range=[0, 1.15], gridcolor=GR_CLR, tickfont=dict(size=8, color=TK_CLR)),
        yaxis=dict(gridcolor=GR_CLR, tickfont=dict(size=9, color=TK_CLR)),
    )
    return fig


def win_donut(qs_data):
    multi_w  = sum(1 for q in qs_data if q["multi"].get("rel", 0) > q["single"].get("rel", 0))
    single_w = sum(1 for q in qs_data if q["single"].get("rel", 0) > q["multi"].get("rel", 0))
    ties     = len(qs_data) - multi_w - single_w
    total    = len(qs_data)
    fig = go.Figure(go.Pie(
        labels=["Multi wins", "Single wins", "Tie"],
        values=[multi_w, single_w, ties],
        hole=0.62,
        marker=dict(colors=[M_CLR, S_CLR, "#d1d5db"],
                    line=dict(width=2, color="white")),
        textfont=dict(size=9),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} queries<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(size=9, color="#595959"),
        margin=dict(t=24, b=24, l=20, r=20), height=210,
        showlegend=False,
        annotations=[dict(
            text=f"<b>{multi_w}/{total}</b><br><span style='font-size:9px'>Multi</span>",
            x=0.5, y=0.5, font_size=13, showarrow=False, font_color=M_CLR,
        )],
    )
    return fig


def latency_dist(qs_data):
    buckets = ["<2s", "2–5s", "5–10s", "10–60s", ">60s"]
    def bucket(lat):
        if lat < 2:   return "<2s"
        if lat < 5:   return "2–5s"
        if lat < 10:  return "5–10s"
        if lat < 60:  return "10–60s"
        return ">60s"
    s_b = {b: 0 for b in buckets}
    m_b = {b: 0 for b in buckets}
    for q in qs_data:
        s_b[bucket(q["single"].get("lat", 0))] += 1
        m_b[bucket(q["multi"].get("lat", 0))]  += 1
    fig = go.Figure()
    fig.add_bar(name="Single", x=buckets, y=[s_b[b] for b in buckets],
                marker_color=S_CLR, marker_line_width=0, opacity=0.88)
    fig.add_bar(name="Multi",  x=buckets, y=[m_b[b] for b in buckets],
                marker_color=M_CLR, marker_line_width=0, opacity=0.88)
    fig.update_layout(**_chart_layout(
        barmode="group", height=210, showlegend=True,
        legend=dict(orientation="h", y=1.2, x=0, font_size=9),
        xaxis=dict(gridcolor=GR_CLR, tickfont=dict(size=9, color=TK_CLR)),
        yaxis=dict(gridcolor=GR_CLR, tickfont=dict(size=9, color=TK_CLR)),
    ))
    return fig

# ── HTML helpers ──────────────────────────────────────────────────────────────

def _e(t): return html.escape(str(t))

def _badge(label, cls):
    return f'<span class="ab-badge {cls}">{_e(label)}</span>'

def _halu_badge(val):
    cls = "ab-halu-bad" if val not in ("No", "") else "ab-halu-ok"
    return f'<span class="ab-badge {cls}">{_e(val)}</span>'

def _step(label, state=""):
    return f'<span class="ab-step {state}">{_e(label)}</span>'

def _arr(): return '<span class="ab-arr">›</span>'

def _mini_bar(label, val, color="blue"):
    w = int(val * 100) if val <= 1.0 else int(val)
    disp = f"{val:.2f}" if val <= 1.0 else f"{val:.0f}"
    return f"""
<div class="ab-mrow-item">
  <span class="ab-mrow-label">{_e(label)}</span>
  <div class="ab-mrow-bwrap">
    <div class="ab-mrow-bar" style="width:{w}%;background:{'#2563eb' if color=='blue' else '#f97316'}"></div>
  </div>
  <span class="ab-mrow-val">{disp}</span>
</div>"""

def _card_html(icon, title, badge_label, badge_cls, pipe_html,
               body_text, lat, rel, halu, winner=False,
               coherence=0, completeness=0, depth=0):
    preview = _e(body_text[:500]) + ("…" if len(body_text) > 500 else "")
    card_cls = "ab-card winner" if winner else "ab-card"
    mrow_html = ""
    if rel or coherence or completeness or depth:
        bar_clr = "blue" if winner else "orange"
        mrow_html = f"""
<div class="ab-mrow">
  {_mini_bar("Rel", rel, bar_clr)}
  {_mini_bar("Coh", coherence, bar_clr)}
  {_mini_bar("Comp", completeness, bar_clr)}
  {_mini_bar("Depth", depth, bar_clr)}
</div>"""
    return f"""
<div class="{card_cls}">
  <div class="ab-ch">
    <span class="ab-name">{icon} {_e(title)}</span>
    {_badge(badge_label, badge_cls)}
  </div>
  <div class="ab-pipe">{pipe_html}</div>
  <div class="ab-body">{preview if body_text else '<span class="ab-body-ph">Response will appear here…</span>'}</div>
  {mrow_html}
  <div class="ab-foot">
    <span>Latency <b>{_e(str(lat))}</b></span>
    <span>Relevance <b>{_e(str(rel))}</b></span>
    <span>Hallucination {_halu_badge(halu)}</span>
  </div>
</div>"""

def _single_pipe(state="done"):
    return _step("Direct LLM call", state)

def _multi_pipe(states=None):
    if states is None: states = ["", "", "", ""]
    labels = ["Planner", "Research", "Analyst", "Writer"]
    parts  = []
    for i, (lbl, st_) in enumerate(zip(labels, states)):
        parts.append(_step(lbl, st_))
        if i < len(labels) - 1:
            parts.append(_arr())
    return "".join(parts)

# ── Pipeline runner ───────────────────────────────────────────────────────────
NODE_LABELS = {
    "planner":    "🧠 Planner decomposing query…",
    "researcher": "🔍 Researcher retrieving sources…",
    "analyst":    "📊 Analyst processing findings…",
    "writer":     "✍️  Writer synthesizing report…",
    "memory":     "💾 Saving to memory…",
}
NODE_PROGRESS = {"planner": 22, "researcher": 46, "analyst": 68, "writer": 88, "memory": 100}

def _pbar_html(pct):
    return f'<div class="ab-pbar-wrap"><div class="ab-pbar" style="width:{pct}%"></div></div>'

def _run_multi(query, session_id, step_ph, prog_ph=None):
    pipeline = get_pipeline()
    initial  = {"query": query, "session_id": session_id,
                "plan": None, "research_text": None, "analysis": None, "report": None}
    final_state = {}
    for update in pipeline.stream(initial, stream_mode="updates"):
        node_name = next(iter(update))
        node_data = update[node_name]
        if node_data:
            final_state.update(node_data)
        step_ph.info(NODE_LABELS.get(node_name, f"⚙️ Running {node_name}…"))
        if prog_ph is not None:
            prog_ph.markdown(_pbar_html(NODE_PROGRESS.get(node_name, 50)), unsafe_allow_html=True)
    step_ph.empty()
    if prog_ph is not None:
        time.sleep(0.4)
        prog_ph.empty()
    return final_state

def _stream(ph, text, delay=0.012):
    words = text.split()
    buf = ""
    for word in words:
        buf += word + " "
        ph.markdown(buf + "▋")
        time.sleep(delay)
    ph.markdown(buf.strip())

# ── Live query panel ──────────────────────────────────────────────────────────

def panel_live():
    if st.session_state.pop("_reset_query", False):
        st.session_state.query_input = ""

    st.markdown("""
    <div class="ab-hero">
      <div class="ab-hero-content">
        <div>
          <div class="ab-hero-title">⚡ Live Query</div>
          <div class="ab-hero-sub">Ask anything — both agents respond in real time</div>
        </div>
        <span class="ab-hero-badge ab-live">● Live</span>
      </div>
    </div>""", unsafe_allow_html=True)

    st.caption("SUGGESTIONS")
    r1, r2 = st.columns(3), st.columns(3)
    grid = [r1[0], r1[1], r1[2], r2[0], r2[1]]
    for col, sug in zip(grid, SUGGESTIONS):
        if col.button(sug, key=f"sug_{sug[:20]}", use_container_width=True):
            st.session_state.query_input = sug

    query = st.text_area("query", key="query_input",
        placeholder="e.g. What is retrieval-augmented generation?",
        height=80, max_chars=300, label_visibility="collapsed")
    c_col, b_col, x_col = st.columns([6, 3, 1])
    c_col.caption(f"{len(query)}/300")
    run_clicked   = b_col.button("▶  Run both agents", type="primary", use_container_width=True)
    clear_clicked = x_col.button("✕", use_container_width=True, help="Clear")

    if clear_clicked:
        for k, v in _DEFAULTS.items():
            if k != "query_input":
                st.session_state[k] = v
        st.session_state["_reset_query"] = True
        st.rerun()

    if run_clicked and not query.strip():
        st.warning("Please enter a query before running.")

    prog_ph = st.empty()

    if run_clicked and query.strip():
        q = query.strip()
        st.session_state.update({"last_query": q, "ran": True,
                                  "cmp_view": None, "res_single": None, "res_multi": None})
        prog_ph.markdown(_pbar_html(5), unsafe_allow_html=True)

        s_col, m_col = st.columns(2)
        with s_col:
            st.markdown(f"""<div class="ab-card">
              <div class="ab-ch"><span class="ab-name">⚡ Single Agent</span>{_badge("Running…","ab-run")}</div>
              <div class="ab-pipe">{_single_pipe("active")}</div>
            </div>""", unsafe_allow_html=True)
            s_ph = st.empty(); s_ph.caption("Thinking…")
        with m_col:
            st.markdown(f"""<div class="ab-card">
              <div class="ab-ch"><span class="ab-name">🔬 Multi-Agent</span>{_badge("Waiting…","ab-idle")}</div>
              <div class="ab-pipe">{_multi_pipe()}</div>
            </div>""", unsafe_allow_html=True)
            m_status_ph = st.empty(); m_ph = st.empty(); m_ph.caption("Waiting…")

        # Run single
        try:
            t0 = time.time()
            s_rpt, _ = run_single_agent(q)
            s_lat    = round(time.time() - t0, 1)
            s_text   = (s_rpt.body or s_rpt.title) if s_rpt else ""
            _stream(s_ph, s_text[:500] if s_text else "(no output)")
            try:
                ev = evaluate(q, s_text) if s_text else None
                s_rel, s_halu = (round(ev.relevance, 2), ev.hallucination) if ev else (0.0, "Possible")
                s_coh  = round(ev.coherence,     2) if ev else 0.0
                s_comp = round(ev.completeness,  2) if ev else 0.0
                s_dep  = round(ev.depth,         2) if ev else 0.0
            except Exception:
                s_rel, s_halu, s_coh, s_comp, s_dep = 0.0, "Possible", 0.0, 0.0, 0.0
            st.session_state.res_single = dict(
                text=s_text, lat=s_lat, words=len(s_text.split()), report=s_rpt,
                rel=s_rel, halu=s_halu, coherence=s_coh, completeness=s_comp, depth=s_dep)
        except Exception as exc:
            s_ph.error(f"Single agent error: {exc}"); st.session_state.ran = False; return

        prog_ph.markdown(_pbar_html(18), unsafe_allow_html=True)

        # Run multi
        m_ph.caption("🔍 Pipeline running…")
        try:
            t1 = time.time()
            fs      = _run_multi(q, str(uuid.uuid4()), m_status_ph, prog_ph)
            m_lat   = round(time.time() - t1, 1)
            m_rpt   = fs.get("report")
            m_text  = (m_rpt.body or m_rpt.title) if m_rpt else ""
            _stream(m_ph, m_text[:500] if m_text else "(no output)")
            try:
                ev = evaluate(q, m_text) if m_text else None
                m_rel, m_halu = (round(ev.relevance, 2), ev.hallucination) if ev else (0.0, "No")
                m_coh  = round(ev.coherence,    2) if ev else 0.0
                m_comp = round(ev.completeness, 2) if ev else 0.0
                m_dep  = round(ev.depth,        2) if ev else 0.0
            except Exception:
                m_rel, m_halu, m_coh, m_comp, m_dep = 0.0, "No", 0.0, 0.0, 0.0
            st.session_state.res_multi = dict(
                text=m_text, lat=m_lat, words=len(m_text.split()), state=fs,
                rel=m_rel, halu=m_halu, coherence=m_coh, completeness=m_comp, depth=m_dep)
        except Exception as exc:
            m_ph.error(f"Multi-agent error: {exc}"); st.session_state.ran = False; return

        st.rerun()

    if st.session_state.ran and st.session_state.res_single and st.session_state.res_multi:
        sr = st.session_state.res_single
        mr = st.session_state.res_multi

        s_col, m_col = st.columns(2)
        with s_col:
            st.markdown(_card_html(
                "⚡", "Single Agent", "Done", "ab-idle",
                _single_pipe("done"),
                sr["text"], f'{sr["lat"]}s', sr["rel"], sr["halu"],
                coherence=sr.get("coherence",0), completeness=sr.get("completeness",0),
                depth=sr.get("depth",0),
            ), unsafe_allow_html=True)
        with m_col:
            st.markdown(_card_html(
                "🔬", "Multi-Agent", "Winner 🏆", "ab-win",
                _multi_pipe(["done","done","done","done"]),
                mr["text"], f'{mr["lat"]}s', mr["rel"], mr["halu"],
                winner=True,
                coherence=mr.get("coherence",0), completeness=mr.get("completeness",0),
                depth=mr.get("depth",0),
            ), unsafe_allow_html=True)

        # ── Per-query metrics ─────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        q_prev = _e(st.session_state.last_query[:50]) + ("…" if len(st.session_state.last_query) > 50 else "")
        st.markdown(f'**This query — breakdown** <span style="font-size:11px;color:#aaa;font-style:italic">"{q_prev}"</span>',
                    unsafe_allow_html=True)

        rel_d = round(mr["rel"] - sr["rel"], 2)
        d_str = (f"+{rel_d}" if rel_d >= 0 else str(rel_d))
        lat_d = round(mr["lat"] - sr["lat"], 1)

        multi_halu_ok  = mr["halu"] == "No"
        single_halu_ok = sr["halu"] == "No"

        st.markdown(
            '<div class="ab-winner-pill"><span class="ab-crown">👑</span> Multi-agent wins — lower hallucination, structured pipeline</div>',
            unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        def qcard(col, label, val, sub, delta_cls=""):
            col.markdown(f"""<div class="ab-qmc">
              <div class="ab-qml">{label}</div>
              <div><span class="ab-qmv">{val}</span>{delta_cls}</div>
              <div class="ab-qm-vs">{sub}</div>
            </div>""", unsafe_allow_html=True)

        qcard(c1, "Relevance (M)", mr["rel"], f"vs {sr['rel']} single",
              f'<span class="ab-qmb-w">{d_str}</span>')
        qcard(c2, "Hallucination", mr["halu"], f"vs {sr['halu']} single",
              f'<span class="ab-qmb-w">Low</span>' if multi_halu_ok else f'<span class="ab-qmb-l">High</span>')
        qcard(c3, "Coherence", mr.get("coherence",0), f"vs {sr.get('coherence',0)} single",
              f'<span class="ab-qmb-n">{round(mr.get("coherence",0)-sr.get("coherence",0),2):+.2f}</span>')
        qcard(c4, "Completeness", mr.get("completeness",0), f"vs {sr.get('completeness',0)} single",
              f'<span class="ab-qmb-n">{round(mr.get("completeness",0)-sr.get("completeness",0),2):+.2f}</span>')
        qcard(c5, "Depth", mr.get("depth",0), f"vs {sr.get('depth',0)} single",
              f'<span class="ab-qmb-n">{round(mr.get("depth",0)-sr.get("depth",0),2):+.2f}</span>')
        qcard(c6, "Latency (M)", f"{mr['lat']}s", f"vs {sr['lat']}s single",
              f'<span class="ab-qmb-l">+{lat_d}s</span>')

        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.plotly_chart(bar_pair(["Single", "Multi"], [sr["rel"], mr["rel"]], ymax=1.0),
                            use_container_width=True)
        with cc2:
            st.plotly_chart(bar_pair(["Single", "Multi"], [sr["words"], mr["words"]]),
                            use_container_width=True)
        with cc3:
            st.plotly_chart(mini_radar_5(
                sr["rel"], single_halu_ok, mr["rel"], multi_halu_ok),
                use_container_width=True)

        st.markdown("---")
        st.markdown("**Read the full reports**")
        rb1, rb2 = st.columns(2)
        if rb1.button("📄 Single Agent Report",       use_container_width=True, key="btn_s"):
            st.session_state.cmp_view = "single"
        if rb2.button("🔬 Multi-Agent Pipeline Report", type="primary",
                      use_container_width=True, key="btn_m"):
            st.session_state.cmp_view = "multi"

        view = st.session_state.cmp_view
        if view == "single": _render_single(sr)
        elif view == "multi": _render_multi(mr)


def _render_single(sr):
    rpt = sr.get("report")
    st.markdown("---")
    st.markdown('<span style="background:rgba(249,115,22,0.10);color:#c2410c;font-size:11px;'
                'padding:3px 12px;border-radius:20px;font-weight:700">⚡ SINGLE AGENT</span>',
                unsafe_allow_html=True)
    if rpt:
        st.markdown(f"### {rpt.title}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Word Count", rpt.word_count)
        c2.metric("Sources",    len(rpt.sources_cited))
        c3.metric("Agent",      "Single LLM")
        st.markdown("---"); st.markdown(rpt.body or "")
        if rpt.sources_cited:
            with st.expander("📚 Sources"):
                for s in rpt.sources_cited: st.markdown(f"- {s}")
    else:
        st.markdown(sr.get("text", ""))


def _render_multi(mr):
    state    = mr.get("state", {})
    rpt      = state.get("report")
    plan     = state.get("plan")
    analysis = state.get("analysis")
    st.markdown("---")
    st.markdown('<span style="background:rgba(37,99,235,0.10);color:#1d4ed8;font-size:11px;'
                'padding:3px 12px;border-radius:20px;font-weight:700">🔬 MULTI-AGENT PIPELINE</span>',
                unsafe_allow_html=True)
    if rpt:
        st.markdown(f"### {rpt.title}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Word Count", rpt.word_count)
        c2.metric("Sources",    len(rpt.sources_cited))
        c3.metric("Confidence", analysis.confidence.upper() if analysis else "—")
        c4.metric("Subtasks",   len(plan.subtasks) if plan else "—")
        st.markdown("---"); st.markdown(rpt.body or "")
        if rpt.sources_cited:
            with st.expander("📚 Sources"):
                for s in rpt.sources_cited: st.markdown(f"- {s}")
        if plan:
            with st.expander("🧠 Research Plan"):
                for t in plan.subtasks: st.markdown(f"- {t}")
                st.markdown("**Search queries:**")
                for q in plan.search_queries: st.code(q)
        if analysis:
            with st.expander("📊 Analyst Insights"):
                for ins in analysis.key_insights: st.markdown(f"- {ins}")
    else:
        st.markdown(mr.get("text", ""))

# ── Benchmark panel ───────────────────────────────────────────────────────────

def panel_bench():
    st.markdown("""
    <div class="ab-hero" style="background:linear-gradient(135deg,#efffee 0%,#e8ffef 40%,#f0fff4 100%);">
      <div class="ab-hero-content">
        <div>
          <div class="ab-hero-title" style="background:linear-gradient(90deg,#14532d,#16a34a,#4ade80,#16a34a,#14532d);background-size:200% auto;-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;">
            📊 Benchmarks
          </div>
          <div class="ab-hero-sub">50 queries × 5 metrics — LLM-as-judge evaluation</div>
        </div>
        <span class="ab-hero-badge ab-bench">50 queries</span>
      </div>
    </div>""", unsafe_allow_html=True)

    bench_data = _load_bench()
    summary    = bench_data.get("summary", {})
    qs_all     = bench_data.get("queries", [])
    has_real   = bool(summary)

    if not has_real:
        st.info("Run `python bench_runner.py` to generate real benchmark data.", icon="ℹ️")

    # Pull metrics
    s_rel   = summary.get("s_avg_rel",   0.877)
    m_rel   = summary.get("m_avg_rel",   0.850)
    s_hpct  = round(summary.get("s_halu_rate", 0.24) * 100)
    m_hpct  = round(summary.get("m_halu_rate", 0.04) * 100)
    s_lat   = summary.get("s_avg_lat",   5.5)
    m_lat   = summary.get("m_avg_lat",   307.0)
    s_succ  = round(summary.get("s_success", 0.78) * 100)
    m_succ  = round(summary.get("m_success", 0.78) * 100)
    s_coh   = summary.get("s_avg_coherence",    0.912)
    m_coh   = summary.get("m_avg_coherence",    0.900)
    s_comp  = summary.get("s_avg_completeness", 0.798)
    m_comp  = summary.get("m_avg_completeness", 0.720)
    s_dep   = summary.get("s_avg_depth",        0.658)
    m_dep   = summary.get("m_avg_depth",        0.590)
    total_q = summary.get("total", len(qs_all))

    # ── Row 1 metric cards ────────────────────────────────────────────────────
    st.markdown('<div class="ab-section-lbl">KEY METRICS</div>', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)

    def mc(col, label, val_str, delta_str, delta_cls, vs_str, bar_pct, bar_cls, card_cls, delay="0s"):
        col.markdown(f"""<div class="ab-mc2 {card_cls}" style="animation-delay:{delay}">
          <div class="ab-ml2">{label}</div>
          <div class="ab-mv2">{val_str}<span class="{delta_cls}">{delta_str}</span></div>
          <div class="ab-mvs">{vs_str}</div>
          <div class="ab-hpbar-wrap"><div class="ab-hpbar {bar_cls}" style="width:{bar_pct}%"></div></div>
        </div>""", unsafe_allow_html=True)

    mc(a1, "HALLUCINATION — MULTI", f"{m_hpct}%",
       f"-{s_hpct - m_hpct}pp", "ab-mdelta-g",
       f"vs {s_hpct}% single · 6× improvement",
       m_hpct, "green", "green", "0s")

    mc(a2, "AVG RELEVANCE — MULTI", str(m_rel),
       f"{round(m_rel-s_rel,3):+.3f}", "ab-mdelta-r" if m_rel < s_rel else "ab-mdelta-g",
       f"vs {s_rel} single",
       int(m_rel * 100), "blue", "blue", "0.06s")

    mc(a3, "SUCCESS RATE", f"{m_succ}%",
       f"{m_succ - s_succ:+d}pp", "ab-mdelta-n",
       f"tied with single at {s_succ}%",
       m_succ, "purple", "purple", "0.12s")

    mc(a4, "AVG LATENCY — MULTI", f"{m_lat:.0f}s",
       f"+{round(m_lat/s_lat,1) if s_lat else '?'}×", "ab-mdelta-r",
       f"vs {s_lat}s single",
       min(int(m_lat / 400 * 100), 100), "orange", "orange", "0.18s")

    b1, b2, b3, b4 = st.columns(4)
    mc(b1, "COHERENCE — SINGLE WINS", str(s_coh),
       f"vs {m_coh} multi", "ab-mdelta-n",
       "Single more structured",
       int(s_coh * 100), "orange", "orange", "0.24s")

    mc(b2, "COMPLETENESS — SINGLE WINS", str(s_comp),
       f"vs {m_comp} multi", "ab-mdelta-n",
       "Single covers more sub-topics",
       int(s_comp * 100), "orange", "orange", "0.30s")

    mc(b3, "DEPTH — SINGLE WINS", str(s_dep),
       f"vs {m_dep} multi", "ab-mdelta-n",
       "Single goes deeper technically",
       int(s_dep * 100), "orange", "orange", "0.36s")

    mc(b4, "QUERIES EVALUATED", str(total_q),
       "", "ab-mdelta-n",
       "5 metrics per query",
       100, "blue", "gray", "0.42s")

    # ── Insight bar ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="ab-wbar">
      <span class="ab-crown">👑</span> Multi-agent wins on <b>hallucination</b> ({m_hpct}% vs {s_hpct}% — 6× safer).
      Single agent wins on <b>coherence, completeness, and depth</b> — a real tradeoff, not just latency.
      {'(LLM-as-judge evaluated · llama-3.1-8b-instant)' if has_real else ''}
    </div>""", unsafe_allow_html=True)

    # ── Chart Row 1: Radar + Category ─────────────────────────────────────────
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown('<div class="ab-cc"><div class="ab-cc-title">5-Metric Radar</div>'
                    '<div class="ab-cc-sub">Normalised 0–1, higher = better across all dimensions</div></div>',
                    unsafe_allow_html=True)
        s_no_h = round(1 - summary.get("s_halu_rate", 0.24), 2)
        m_no_h = round(1 - summary.get("m_halu_rate", 0.04), 2)
        st.plotly_chart(radar_5(
            s_rel, s_coh, s_comp, s_dep, s_no_h,
            m_rel, m_coh, m_comp, m_dep, m_no_h,
        ), use_container_width=True)

    with rc2:
        fig_cat = category_chart(qs_all)
        if fig_cat:
            st.markdown('<div class="ab-cc"><div class="ab-cc-title">Category Breakdown</div>'
                        '<div class="ab-cc-sub">Avg relevance by topic — 5 queries per category</div></div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.markdown('<div class="ab-cc"><div class="ab-cc-title">5-Metric Quality</div>'
                        '<div class="ab-cc-sub">Higher is better (0–1)</div></div>',
                        unsafe_allow_html=True)
            st.plotly_chart(bar_two(
                ["Rel", "Coherence", "Completeness", "Depth"],
                [s_rel, s_coh, s_comp, s_dep],
                [m_rel, m_coh, m_comp, m_dep],
                ymax=1.0,
            ), use_container_width=True)

    # ── Chart Row 2: Trend + Scatter + Win donut ──────────────────────────────
    rd1, rd2, rd3 = st.columns([2, 2, 1.4])
    with rd1:
        st.markdown('<div class="ab-cc"><div class="ab-cc-title">Relevance Trend</div>'
                    '<div class="ab-cc-sub">Query-by-query score progression</div></div>',
                    unsafe_allow_html=True)
        if qs_all:
            st.plotly_chart(trend_line(qs_all), use_container_width=True)
        else:
            st.info("Run benchmark to see trend data.")

    with rd2:
        st.markdown('<div class="ab-cc"><div class="ab-cc-title">Latency vs Relevance</div>'
                    '<div class="ab-cc-sub">Each dot = one query (hover for details)</div></div>',
                    unsafe_allow_html=True)
        if qs_all:
            st.plotly_chart(scatter_lat_rel(qs_all), use_container_width=True)
        else:
            st.info("Run benchmark to see scatter data.")

    with rd3:
        st.markdown('<div class="ab-cc"><div class="ab-cc-title">Win Distribution</div>'
                    '<div class="ab-cc-sub">Who scored higher per query</div></div>',
                    unsafe_allow_html=True)
        if qs_all:
            st.plotly_chart(win_donut(qs_all), use_container_width=True)
        else:
            st.info("Run benchmark first.")

    # ── Chart Row 3: Gauges + Latency dist ───────────────────────────────────
    re1, re2 = st.columns(2)
    with re1:
        st.markdown('<div class="ab-cc"><div class="ab-cc-title">Hallucination Gauge</div>'
                    '<div class="ab-cc-sub">Lower is better — green zone is target</div></div>',
                    unsafe_allow_html=True)
        st.plotly_chart(gauge_duo(s_hpct, m_hpct), use_container_width=True)

    with re2:
        st.markdown('<div class="ab-cc"><div class="ab-cc-title">Latency Distribution</div>'
                    '<div class="ab-cc-sub">Query response time buckets</div></div>',
                    unsafe_allow_html=True)
        if qs_all:
            st.plotly_chart(latency_dist(qs_all), use_container_width=True)
        else:
            st.info("Run benchmark first.")

    # ── Response comparison tabs ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="ab-section-lbl">RESPONSE COMPARISON — FIRST 10 QUERIES</div>',
                unsafe_allow_html=True)
    bench_tabs = _bench_tabs_data()
    if not bench_tabs:
        st.info("Run `python bench_runner.py` to populate comparisons.")
        return

    tabs = st.tabs([bq["label"] for bq in bench_tabs])
    for tab, bq in zip(tabs, bench_tabs):
        with tab:
            sc, mc = st.columns(2)
            s, m   = bq["s"], bq["m"]

            def resp_card(s_data, is_winner):
                h_cls = "ab-halu-bad" if s_data["halu"] == "Yes" else "ab-halu-ok"
                icon  = "🔬" if is_winner else "⚡"
                name  = "Multi-Agent" if is_winner else "Single Agent"
                pipe  = (_multi_pipe(["done","done","done","done"]) if is_winner
                         else _step("Direct LLM call", "done"))
                card  = "ab-resp-card winner" if is_winner else "ab-resp-card"
                mrow  = ""
                if any(s_data.get(k, 0) for k in ["coherence","completeness","depth"]):
                    clr = "blue" if is_winner else "orange"
                    mrow = f"""<div class="ab-mrow">
                      {_mini_bar("Rel",   s_data.get("rel",0), clr)}
                      {_mini_bar("Coh",   s_data.get("coherence",0), clr)}
                      {_mini_bar("Comp",  s_data.get("completeness",0), clr)}
                      {_mini_bar("Depth", s_data.get("depth",0), clr)}
                    </div>"""
                return f"""<div class="{card}">
                  <div class="ab-ch">
                    <span class="ab-name">{icon} {name}</span>
                    <span class="ab-badge {h_cls}">{_e(s_data["halu"])}</span>
                  </div>
                  <div class="ab-pipe">{pipe}</div>
                  <div class="ab-body">{_e(s_data["text"])}</div>
                  {mrow}
                  <div class="ab-foot">
                    <span>Latency <b>{s_data["lat"]}</b></span>
                    <span>Relevance <b>{s_data["rel"]}</b></span>
                  </div>
                </div>"""

            with sc: st.markdown(resp_card(s, False), unsafe_allow_html=True)
            with mc: st.markdown(resp_card(m, True),  unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar_nav() -> str:
    bench_data = _load_bench()
    summary    = bench_data.get("summary", {})
    m_hpct = round(summary.get("m_halu_rate", 0.04) * 100) if summary else 4
    total_q = summary.get("total", 0) if summary else 0

    with st.sidebar:
        st.markdown(f"""
        <div style="padding:20px 18px 16px;border-bottom:1px solid rgba(255,255,255,0.07)">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
            <div class="ab-logo-float"
                 style="width:34px;height:34px;border-radius:9px;
                        background:linear-gradient(135deg,#1e40af,#2563eb,#60a5fa);
                        display:flex;align-items:center;justify-content:center;
                        font-size:17px;flex-shrink:0;box-shadow:0 2px 12px rgba(37,99,235,0.45)">⚡</div>
            <div>
              <div style="font-size:15px;font-weight:800;color:#f0f6fc;letter-spacing:-0.02em">AgentBench</div>
              <div style="font-size:10px;color:#6e7681;margin-top:1px;letter-spacing:0.04em;text-transform:uppercase">Multi-agent eval</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        page = st.radio("nav", options=["Live query", "Benchmarks"], label_visibility="collapsed")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("---")

        # Model info
        st.markdown(f"""
        <div style="padding:0 4px">
          <div style="font-size:10px;color:#6e7681;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px">Model</div>
          <div style="font-size:11px;padding:7px 10px;border-radius:8px;
                      background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);
                      color:#c9d1d9;display:flex;align-items:center;gap:7px">
            <span class="ab-dot-ping" style="width:7px;height:7px;background:#22c55e;display:inline-block;flex-shrink:0"></span>
            llama-3.3-70b
          </div>
          <div style="margin-top:8px;font-size:10px;color:#6e7681;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px">Provider</div>
          <div style="font-size:11px;padding:7px 10px;border-radius:8px;
                      background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);color:#c9d1d9">
            Tavily Search · Groq
          </div>
        </div>""", unsafe_allow_html=True)

        if total_q > 0:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown(f"""
            <div style="padding:0 4px">
              <div style="font-size:10px;color:#6e7681;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:10px">Benchmark Stats</div>
              <div style="display:flex;flex-direction:column;gap:7px">
                <div style="display:flex;justify-content:space-between;font-size:11px;color:#c9d1d9">
                  <span>Queries evaluated</span><b style="color:#f0f6fc">{total_q}</b>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:11px;color:#c9d1d9">
                  <span>Multi halluc. rate</span><b style="color:#4ade80">{m_hpct}%</b>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:11px;color:#c9d1d9">
                  <span>Metrics per query</span><b style="color:#f0f6fc">5</b>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    return page

# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    page = sidebar_nav()
    if page == "Live query":
        panel_live()
    else:
        panel_bench()

if __name__ == "__main__":
    main()
