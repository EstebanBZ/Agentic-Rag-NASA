# main.py
# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Any
from urllib.parse import quote

from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# ==== LangChain / Azure OpenAI / FAISS ====
from langchain_openai import AzureChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from graph import build_agentic_graph

# ==== Graph / Plotly ====
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# =======================================================
# GENERAL CONFIG
# =======================================================
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

INPUT_DIR = os.getenv("MD_INPUT_DIR", "./extracted_md")  # Folder with .md files
NGRAM_RANGE = (1, 2)
TOP_K_TERMS_PER_DOC = 100
MIN_DF = 2
EDGE_METRIC = "overlap"
WEIGHT_THRESHOLD = 1
SEED = 42

# ---------- Resolve paths relative to this file (robust to cwd changes) ----------
BASE_DIR = Path(__file__).resolve().parent  # folder where main.py lives

def resolve_path(p: str | os.PathLike, base: Path = BASE_DIR) -> Path:
    """Return absolute Path. If `p` is relative, resolve against `base` (main.py dir)."""
    p = Path(p)
    return p if p.is_absolute() else (base / p)

# PDFs folder (accepts absolute or relative to main.py)
PDF_DIR: Path = resolve_path(os.getenv("PDF_DIR", "./pdfs"))

# =======================================================
# FASTAPI INIT
# =======================================================
app = FastAPI(title="NASA Bio — Agentic RAG + Graph + PDF API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional static
if (BASE_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Mount PDFs folder if exists (served at /pdfs/<file.pdf>)
if PDF_DIR.exists():
    app.mount("/pdfs", StaticFiles(directory=str(PDF_DIR)), name="pdfs")

# =======================================================
# LOAD RAG STACK
# =======================================================
def load_stack():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.1,
        max_retries=2,
    )
    graph = build_agentic_graph(llm, db)
    return graph

GRAPH_PIPELINE = load_stack()

# =======================================================
# BUILD INTERACTIVE GRAPH
# =======================================================
MD_CODEBLOCK_RE = re.compile(r"```.*?```", re.S)
MD_INLINE_CODE_RE = re.compile(r"`[^`]+`")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
MD_IMAGE_RE = re.compile(r"!\[.*?\]\(.*?\)")
MD_HTML_TAG_RE = re.compile(r"<[^>]+>")
MD_REF_RE = re.compile(r"\[\d+\]")
MULTISPACE_RE = re.compile(r"\s+")

def clean_markdown(raw: str) -> str:
    x = MD_CODEBLOCK_RE.sub(" ", raw)
    x = MD_IMAGE_RE.sub(" ", x)
    x = MD_INLINE_CODE_RE.sub(" ", x)
    x = MD_LINK_RE.sub(r"\1", x)
    x = MD_HTML_TAG_RE.sub(" ", x)
    x = MD_REF_RE.sub(" ", x)
    x = re.sub(r"[#>*_~\-]+", " ", x)
    x = re.sub(r"\b\d+(\.\d+)?\b", " ", x)
    x = x.lower()
    x = MULTISPACE_RE.sub(" ", x).strip()
    return x

def read_markdown_files(folder: str) -> Dict[str, str]:
    p = resolve_path(folder)  # ensure relative INPUT_DIR is resolved to absolute
    files = sorted(p.glob("*.md")) if p.exists() else []
    return {f.stem: clean_markdown(f.read_text(encoding="utf-8", errors="ignore")) for f in files}

def top_k_terms_per_doc(tfidf, feat_names, k):
    tops = []
    for i in range(tfidf.shape[0]):
        row = tfidf[i].toarray().ravel()
        idx = np.argpartition(row, -k)[-k:]
        idx = idx[np.argsort(-row[idx])]
        tops.append([(feat_names[j], float(row[j])) for j in idx if row[j] > 0])
    return tops

def adjacency_from_concepts(sets: List[Set[str]], metric: str = "overlap"):
    n = len(sets)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            inter = sets[i].intersection(sets[j])
            if metric == "jaccard":
                union = sets[i].union(sets[j])
                w = len(inter) / len(union) if union else 0
            else:
                w = len(inter)
            A[i, j] = A[j, i] = w
    return A

def build_plotly_graph() -> go.Figure:
    docs = read_markdown_files(INPUT_DIR)
    if not docs:
        fig = go.Figure()
        fig.update_layout(
            title="Publications graph — (no .md data in INPUT_DIR)",
            height=650, showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        return fig

    names = list(docs.keys())
    texts = list(docs.values())
    stops = set(ENGLISH_STOP_WORDS).union({
        "et", "al", "figure", "fig", "table", "tables", "supplementary",
        "introduction", "methods", "materials", "discussion", "conclusion",
        "data", "result", "results", "based", "using", "use", "used", "shown",
        "analysis", "study", "paper", "however", "therefore", "thus", "also"
    })

    vectorizer = TfidfVectorizer(
        stop_words=list(stops),
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        token_pattern=r"(?u)\b[a-z][a-z\-]+\b"
    )
    X = vectorizer.fit_transform(texts)
    feats = vectorizer.get_feature_names_out()

    tops = top_k_terms_per_doc(X, feats, TOP_K_TERMS_PER_DOC)
    concept_sets = [set(t for t, _ in top) for top in tops]

    A = adjacency_from_concepts(concept_sets, EDGE_METRIC)
    G = nx.Graph()
    for i, n in enumerate(names):
        G.add_node(n, idx=i, terms=concept_sets[i])
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if A[i, j] > WEIGHT_THRESHOLD:
                G.add_edge(names[i], names[j], weight=A[i, j])

    pos = nx.spring_layout(G, seed=SEED)

    # --- Edges ---
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="#cccccc"), hoverinfo="none"
    )

    # --- Nodes (with customdata for the client) ---
    node_x, node_y, node_text, node_custom = [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        terms = sorted(list(G.nodes[n]["terms"]))
        node_custom.append(terms)  # <-- terms for the client
        node_text.append(f"<b>{n}</b><br>{', '.join(terms[:10])}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        text=node_text,
        hoverinfo="text",
        marker=dict(size=10, color="#9EA4A9", line=dict(width=0.5)),
        customdata=node_custom,
        name="nodes"
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Publications relationship graph",
        height=650, showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

PLOTLY_FIG_JSON = build_plotly_graph().to_json()

# =======================================================
# ENDPOINTS
# =======================================================
@app.get("/", response_class=HTMLResponse)
def index():
    """Serves index.html"""
    html = (BASE_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)

@app.get("/api/graph")
def get_graph():
    """Returns the Plotly figure (JSON serializable)."""
    return JSONResponse({"figure": PLOTLY_FIG_JSON})

@app.post("/api/chat")
async def chat(payload: Dict[str, Any]):
    """Receives the message history and returns the RAG answer."""
    messages: List[Dict[str, str]] = payload.get("messages", [])
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return JSONResponse({"answer_md": "No question received."})
    question = user_msgs[-1]["content"]

    state = {"question": question, "k_per_query": 6}
    out = GRAPH_PIPELINE.invoke(state)
    answer = out.get("answer_md", "No response.")
    return JSONResponse({"answer_md": answer})

@app.get("/api/search_pdfs")
def search_pdfs(q: str = ""):
    """
    Search PDFs by name (substring, case-insensitive) in PDF_DIR.
    Returns list [{name, filename, url, mtime}]
    """
    root = PDF_DIR  # already an absolute Path
    if not root.exists():
        return JSONResponse({"items": []})

    needle = (q or "").strip().lower()
    files = [f for f in root.glob("*.pdf") if f.is_file()]

    if needle:
        files = [f for f in files if needle in f.stem.lower()]

    files_sorted = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    items = [{
        "name": f.stem,
        "filename": f.name,
        "url": f"/pdfs/{quote(f.name)}",
        "mtime": int(f.stat().st_mtime)
    } for f in files_sorted[:100]]

    return JSONResponse({"items": items})
