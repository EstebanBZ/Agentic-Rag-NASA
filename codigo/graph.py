# graph.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import TypedDict, List, Tuple, Dict, Any
from dataclasses import dataclass
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI


# ============================================================
# Estado del agente (fluye entre nodos del grafo)
# ============================================================
class AgentState(TypedDict, total=False):
    question: str
    in_domain: bool
    domain_label: str
    queries: List[str]
    retrieved: List[Tuple[Document, float]]              # (doc, distance)
    filtered: List[Tuple[Document, float]]               # (doc, distance) que pasan umbral
    context_text: str
    answer_md: str
    citations: List[Dict[str, Any]]                      # citas usadas (con métricas)
    threshold: float                                     # umbral distancia (<= pasa)
    k_per_query: int
    top10: List[Dict[str, Any]]                          # top-10 con métricas y flag used
    threshold_similarity: float                          # umbral de similitud equivalente


# ============================================================
# Utilidades
# ============================================================
def _unique_docs_with_min_score(pairs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    """
    De-dup por (source, primeros 300 chars). Mantiene el menor score (mejor distancia) por clave.
    Devuelve ordenado ascendente por score (mejor primero).
    """
    seen: Dict[Tuple[str, str], Tuple[Document, float]] = {}
    for d, s in pairs:
        key = (d.metadata.get("source", ""), d.page_content.strip()[:300])
        if key not in seen or s < seen[key][1]:
            seen[key] = (d, s)
    return sorted(seen.values(), key=lambda x: x[1])


def _cosine_from_distance(dist: float) -> float:
    """
    Aprox. coseno para embeddings normalizados: sim = 1 - (dist**2)/2
    """
    return 1.0 - (dist * dist) / 2.0


# ============================================================
# Nodos del grafo
# ============================================================
@dataclass
class Nodes:
    llm: AzureChatOpenAI
    db: FAISS

    # --- 1) Clasificador de dominio (simple) ---
    def classify_domain(self, state: AgentState) -> AgentState:
        prompt = (
            "You are a domain verifier. Does the question belong to 'Biosciences'?"
            ' Return JSON: {"in_domain": true|false, "label": "NASA_BIO"|"OUT"}\n\n'
            f"Question: {state['question']}"
        )
        rsp = self.llm.invoke(prompt).content.strip()
        low = rsp.lower()
        in_domain = ("true" in low) and ("false" not in low)
        if "nasa_bio" in low:
            label = "NASA_BIO"
        elif "out" in low:
            label = "OUT"
        else:
            label = "NASA_BIO" if in_domain else "OUT"
        return {"in_domain": in_domain, "domain_label": label}

    # --- 2) Expansión multi-query ---
    def expand_queries(self, state: AgentState) -> AgentState:
        q = state["question"]
        prompt = (
            "Generate 4-6 reformulations in ENGLISH (multi-query) to improve dense recall.\n"
            'Respond ONLY JSON: {"queries": ["..."]}\n\n'
            f"User question: {q}"
        )
        text = self.llm.invoke(prompt).content
        import json, re
        queries: List[str] = []
        try:
            data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            queries = [s for s in data.get("queries", []) if isinstance(s, str)]
        except Exception:
            queries = [q, f"scientific findings related to: {q}"]
        if q not in queries:
            queries.insert(0, q)
        queries = list(dict.fromkeys([s.strip() for s in queries if s.strip()]))[:6]
        return {"queries": queries}

    # --- 3) Recuperación con FAISS ---
    def retrieve(self, state: AgentState) -> AgentState:
        k = state.get("k_per_query", 6)
        pairs: List[Tuple[Document, float]] = []
        for subq in state["queries"]:
            try:
                pairs.extend(self.db.similarity_search_with_score(subq, k=k))
            except Exception:
                # continuar aunque una subconsulta falle
                pass
        pairs = _unique_docs_with_min_score(pairs)
        return {"retrieved": pairs}

    # --- 4) Umbral (DISTANCIA), métricas y contexto ---
    def threshold_and_context(self, state: AgentState) -> AgentState:
        pairs = state["retrieved"]
        if not pairs:
            return {
                "filtered": [],
                "context_text": "",
                "citations": [],
                "top10": [],
                "threshold": float(state.get("threshold", 0.30)),
                "threshold_similarity": _cosine_from_distance(float(state.get("threshold", 0.30))),
            }

        # ✅ Usamos SIEMPRE modo distancia (menor = mejor)
        thr_dist = float(state.get("threshold", 0.30))
        thr_sim = _cosine_from_distance(thr_dist)

        def keep(dist: float) -> bool:
            return dist <= thr_dist

        # Enriquecer con métricas
        enriched = []
        for d, dist in pairs:
            sim = _cosine_from_distance(dist)
            enriched.append({
                "doc": d,
                "distance": float(dist),
                "similarity": float(sim),
                "use": keep(dist),
            })

        # Filtrar los que pasan umbral de distancia
        filtered: List[Tuple[Document, float]] = [(e["doc"], e["distance"]) for e in enriched if e["use"]]

        # Fallback mínimo: si nada pasa, usa los 2 mejores
        if not filtered and pairs:
            filtered = pairs[:2]

        # Top-10 para visualización (✅/❌)
        top10 = enriched[:10]
        top10_display = []
        for i, e in enumerate(top10, start=1):
            top10_display.append({
                "index": i,
                "source": e["doc"].metadata.get("source", "Unknown"),
                "distance": round(e["distance"], 4),
                "similarity": round(e["similarity"], 4),
                "use": e["use"],
                "mark": "✅" if e["use"] else "❌",
            })

        # Construir contexto SOLO con los válidos
        ctx_chunks: List[str] = []
        for idx, (d, dist) in enumerate(filtered, start=1):
            src = d.metadata.get("source", "Unknown")
            ctx_chunks.append(f"[{idx}] Source: {src}\n{d.page_content}")

        # Citas usadas (con ambas métricas)
        citations_used: List[Dict[str, Any]] = []
        for idx, (d, dist) in enumerate(filtered, start=1):
            citations_used.append({
                "index": idx,
                "source": d.metadata.get("source", "Unknown"),
                "distance": float(dist),
                "similarity": _cosine_from_distance(float(dist)),
                "use": True,
            })

        return {
            "filtered": filtered,
            "context_text": "\n\n---\n\n".join(ctx_chunks),
            "citations": citations_used,          # con distancia + similitud + use=True
            "threshold": thr_dist,
            "threshold_similarity": thr_sim,      # para mostrar equivalente
            "top10": top10_display,               # top-10 con métricas y flag used
        }

    # --- 5) Generación de respuesta ---
    def generate_answer(self, state: AgentState) -> AgentState:
        if not state.get("in_domain", False):
            msg = (
                "⚠️ The question is outside the 'NASA biosciences / space life sciences' domain "
                "or the classifier did not find a match.\n\n"
                "**Answer:** I don't have enough information to answer."
            )
            return {"answer_md": msg}

        if not state.get("filtered"):
            msg = (
                f"⚠️ No sufficient evidence found under the threshold "
                f"(distance ≤ {state.get('threshold', 0):.2f}).\n\n"
                "**Answer:** I don't have enough information to answer."
            )
            return {"answer_md": msg}

        # Citas realmente utilizadas (con ambas métricas)
        cites_lines = []
        for c in state["citations"]:
            cites_lines.append(
                f"- [{c['index']}] {c['source']} "
                f"(dist={c['distance']:.4f}, sim={c['similarity']:.4f})"
            )
        cites_block = "\n".join(cites_lines)

        # Top-10 para transparencia (se muestran todas, se usan solo las ✅)
        top10_extra = ""
        if "top10" in state and state["top10"]:
            hdr = (
                f"\n\n---\n**Top-10 retrieved** "
                f"(threshold: dist ≤ {state.get('threshold',0):.2f}, sim ≥ {state.get('threshold_similarity',0):.2f})\n"
            )
            lines = [
                f"{t['mark']} [{t['index']}] {t['source']} "
                f"(dist={t['distance']:.4f}, sim={t['similarity']:.4f})"
                for t in state["top10"]
            ]
            top10_extra = hdr + "\n".join(lines)

        # Prompt de generación
        prompt = f"""
You are a scientific assistant. Answer **only** using the CONTEXT.
If it is not enough, say: "I don't have enough information to answer."

Markdown format:
- Clear explanation with subtitles/bullets.
- Highlight *proteins/genes/pathways* with italics or **bold**.
- Use references [n] consistent with the CONTEXT.
- End with a **Summary** (3–5 lines).

CONTEXT:
{state['context_text']}

QUESTION:
{state['question']}
"""
        ans = self.llm.invoke(prompt).content
        ans += f"\n\n---\n**Citations used**\n{cites_block}{top10_extra}"
        return {"answer_md": ans}


# ============================================================
# Construcción del grafo Agentic RAG
# ============================================================

def build_agentic_graph(llm: AzureChatOpenAI, db: FAISS):
    nodes = Nodes(llm=llm, db=db)
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_domain", nodes.classify_domain)
    workflow.add_node("expand_queries", nodes.expand_queries)
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("threshold_and_context", nodes.threshold_and_context)
    workflow.add_node("generate_answer", nodes.generate_answer)

    workflow.set_entry_point("classify_domain")

    def route(state: AgentState) -> str:
        return "expand_queries" if state.get("in_domain", False) else "generate_answer"

    workflow.add_conditional_edges(
        "classify_domain",
        route,
        {"expand_queries": "expand_queries", "generate_answer": "generate_answer"},
    )
    workflow.add_edge("expand_queries", "retrieve")
    workflow.add_edge("retrieve", "threshold_and_context")
    workflow.add_edge("threshold_and_context", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()
