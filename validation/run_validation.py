"""Étape 7 : validation de la recherche et de l'agent.

Usage : python validation/run_validation.py [--retrieval-only]
Écrit outputs/validation_<modèle>.json et .md (modèle choisi par OLLAMA_MODEL).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config
from rag.agent import build_agent, run_agent
from rag.embeddings import ClipEmbedder
from rag.retrieval import Retriever
from rag.store import VectorStore

TEXT_CASES = [
    ("number of parallel attention layers h = 8 heads", 5),
    ("How many attention heads are used in the base model", 5),
    ("dropout rate Pdrop 0.1", 8),
    ("label smoothing value", 8),
    ("learning rate schedule warmup steps", 7),
    ("positional encoding sine and cosine functions", 6),
    ("BLEU score on English to German translation", 8),
    ("number of encoder and decoder layers N = 6", 3),
    ("dimension of the model output d_model = 512", 3),
    ("number of GPUs used for training", 7),
]

IMAGE_CASES = [
    ("Transformer model architecture diagram with encoder and decoder stacks", 1),
    ("diagram of scaled dot-product attention", 2),
    ("multi-head attention diagram with parallel layers", 2),
    ("attention visualization between words of a sentence", 3),
    ("anaphora resolution attention heads", 4),
    ("attention heads related to the structure of the sentence", 5),
    ("long-distance dependencies in encoder self-attention", 3),
]

AGENT_CASES = [
    {"id": "texte-fait", "question": "Combien de têtes d'attention utilise le modèle de base ?",
     "expected": ["8"], "pages": [5]},
    {"id": "texte-regularisation", "question": "Quelle valeur de dropout est utilisée pendant l'entraînement ?",
     "expected": ["0.1", "0,1"], "pages": [8]},
    {"id": "texte-resultat", "question": "Quel score BLEU le Transformer obtient-il en traduction anglais-allemand ?",
     "expected": ["28.4", "28,4"], "forbidden": ["41.8", "41,8"], "pages": [1, 8]},
    {"id": "figure", "question": "Quelle figure montre l'architecture complète du Transformer, et à quelle page ?",
     "expected": ["Figure 1"], "pages": [3]},
    {"id": "hybride-texte-image", "question": "Trouve le schéma de la multi-head attention, puis explique à partir du texte pourquoi l'article utilise plusieurs têtes.",
     "expected": ["Figure 2"], "pages": [4, 5]},
    {"id": "requete-image", "question": "À quelle figure de l'article correspond cette image, et quel mécanisme décrit-elle ?",
     "image": "data/images/page04_xref183.png", "expected": ["Figure 2"], "pages": [4]},
    {"id": "hors-document", "question": "Quel était le prix des GPU utilisés pour l'entraînement ?",
     "expected": [], "pages": [], "must_abstain": True},
]

ABSTENTION = ("ne contient", "ne mentionne", "pas d'information", "pas mentionné", "n'indique", "ne précise",
              "ne donne", "pas indiqué", "pas précisé", "introuvable", "ne figure", "not mention")


def evaluate_text(retriever: Retriever, use_lexical: bool) -> list[dict]:
    rows = []
    for query, page in TEXT_CASES:
        pages = [r.page for r in retriever.search(query, k=5, content_type="text", use_lexical=use_lexical)]
        rows.append({"query": query, "expected_page": page, "pages": pages,
                     "hit1": page in pages[:1], "hit3": page in pages[:3], "hit5": page in pages})
    return rows


def evaluate_images(retriever: Retriever, use_captions: bool) -> list[dict]:
    rows = []
    for query, figure in IMAGE_CASES:
        figures = [r.figure for r in retriever.search(query, k=3, content_type="image", use_captions=use_captions)]
        rows.append({"query": query, "expected_figure": figure, "figures": figures,
                     "hit1": figure in figures[:1], "hit3": figure in figures})
    return rows


def report_slug() -> str:
    return re.sub(r"\W+", "_", config.OLLAMA_MODEL)


def rate(rows: list[dict], key: str) -> str:
    return f"{sum(r[key] for r in rows)}/{len(rows)}"


def evaluate_agent(retriever: Retriever) -> list[dict]:
    agent = build_agent(retriever)
    rows = []
    for case in AGENT_CASES:
        started, trace, error = time.time(), [], None
        try:
            trace = list(run_agent(agent, case["question"], case.get("image")))
        except Exception as exc:  # une question en échec ne doit pas interrompre la campagne
            error = f"{type(exc).__name__}: {exc}"
        answer = next((e["text"] for e in reversed(trace) if e["step"] == "answer"), "")
        actions = [e["args"] for e in trace if e["step"] == "action"]
        pages = sorted({r["page"] for e in trace if e["step"] == "observation" for r in e["results"]})
        lowered = answer.lower()
        context_found = bool(pages) and (not case["pages"] or bool(set(case["pages"]) & set(pages)))
        if case.get("must_abstain"):
            correct = any(mark in lowered for mark in ABSTENTION) and "$" not in answer
        else:
            # Une bonne valeur sans contexte récupéré vient de la mémoire du LLM : elle ne compte pas.
            correct = (context_found and any(value.lower() in lowered for value in case["expected"])
                       and not any(value in lowered for value in case.get("forbidden", [])))
        rows.append({
            "id": case["id"], "question": case["question"], "image": case.get("image"),
            "tool_calls": len(actions), "actions": actions, "pages_retrieved": pages,
            "context_found": context_found,
            "answer": answer, "correct": correct, "error": error,
            "seconds": round(time.time() - started, 1),
        })
        print(f"{case['id']}: outil x{len(actions)}, correct={correct}, {rows[-1]['seconds']}s", flush=True)
    return rows


def write_report(report: dict) -> None:
    lines = ["# Validation du RAG multimodal", "",
             f"Modèle : {report['model']}. Recherche texte sur {len(TEXT_CASES)} requêtes, "
             f"recherche image sur {len(IMAGE_CASES)} requêtes.", "",
             "## Recherche", "",
             "| Mesure | Résultat |", "|---|---|"]
    dense, text, visual, hybrid = (report["text_clip"], report["text_hybrid"],
                                   report["images_visual"], report["images_hybrid"])
    lines += [f"| Texte, CLIP seul, rang 1 | {rate(dense, 'hit1')} |",
              f"| Texte, CLIP seul, top 3 | {rate(dense, 'hit3')} |",
              f"| Texte, CLIP seul, top 5 | {rate(dense, 'hit5')} |",
              f"| Texte, CLIP + BM25, rang 1 | {rate(text, 'hit1')} |",
              f"| Texte, CLIP + BM25, top 3 | {rate(text, 'hit3')} |",
              f"| Texte, CLIP + BM25, top 5 | {rate(text, 'hit5')} |",
              f"| Image, visuel seul, rang 1 | {rate(visual, 'hit1')} |",
              f"| Image, visuel seul, top 3 | {rate(visual, 'hit3')} |",
              f"| Image, visuel + légende, rang 1 | {rate(hybrid, 'hit1')} |",
              f"| Image, visuel + légende, top 3 | {rate(hybrid, 'hit3')} |", ""]
    if report["agent"]:
        lines += ["## Agent", "", "| Cas | Appels d'outil | Contexte trouvé | Réponse correcte | Durée (s) |",
                  "|---|---|---|---|---|"]
        for r in report["agent"]:
            lines.append(f"| {r['id']} | {r['tool_calls']} | {'oui' if r['context_found'] else 'non'} "
                         f"| {'oui' if r['correct'] else 'non'} | {r['seconds']} |")
        lines += ["", "## Traces", ""]
        for r in report["agent"]:
            lines += [f"### {r['id']}", "", f"**Question :** {r['question']}", ""]
            lines += [f"- Action : `{json.dumps(a, ensure_ascii=False)}`" for a in r["actions"]]
            lines += [f"- Pages récupérées : {r['pages_retrieved']}", "", f"**Réponse :** {r['answer'] or r['error']}", ""]
    path = config.OUTPUT_DIR / f"validation_{report_slug()}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapport : {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    retriever = Retriever(VectorStore(), ClipEmbedder())
    report = {
        "model": config.OLLAMA_MODEL,
        "text_clip": evaluate_text(retriever, use_lexical=False),
        "text_hybrid": evaluate_text(retriever, use_lexical=True),
        "images_visual": evaluate_images(retriever, use_captions=False),
        "images_hybrid": evaluate_images(retriever, use_captions=True),
        "agent": [] if args.retrieval_only else evaluate_agent(retriever),
    }
    (config.OUTPUT_DIR / f"validation_{report_slug()}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report)
    for name in ("text_clip", "text_hybrid"):
        print(f"{name} {rate(report[name], 'hit1')} / {rate(report[name], 'hit3')} / {rate(report[name], 'hit5')}")
    print(f"image visuel {rate(report['images_visual'], 'hit1')}, hybride {rate(report['images_hybrid'], 'hit1')}")


if __name__ == "__main__":
    main()
