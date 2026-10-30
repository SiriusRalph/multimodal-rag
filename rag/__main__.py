"""Interface en ligne de commande : python -m rag {index,search,ask}."""
import argparse
import json
import re

from . import config


def components():
    from .embeddings import ClipEmbedder
    from .retrieval import Retriever
    from .store import VectorStore

    embedder = ClipEmbedder()
    store = VectorStore()
    return store, embedder, Retriever(store, embedder)


def cmd_index(args) -> None:
    from .store import index_pdf

    store, embedder, _ = components()
    report = index_pdf(args.pdf, store, embedder, force=args.force)
    print(f"{report.status} | {report.texts} fragments | {report.images} images | sha256 {report.doc_id[:16]}")


def cmd_search(args) -> None:
    from .retrieval import format_results, write_html_preview

    _, _, retriever = components()
    results = retriever.search(args.query, args.image, args.k, args.type)
    print(format_results(results))
    title = args.query or f"Image : {args.image}"
    slug = re.sub(r"\W+", "_", title).strip("_")[:50]
    print(f"\nAperçu : {write_html_preview(results, title, config.OUTPUT_DIR / f'search_{slug}.html')}")


def cmd_ask(args) -> None:
    from .agent import build_agent, run_agent

    _, _, retriever = components()
    agent = build_agent(retriever)
    for event in run_agent(agent, args.question, args.image):
        if event["step"] == "reasoning":
            print(f"\n[REASONING] {event['text']}")
        elif event["step"] == "action":
            print(f"\n[ACTION] {event['tool']} {json.dumps(event['args'], ensure_ascii=False)}")
        elif event["step"] == "observation":
            print(f"\n[OBSERVATION]\n{event['text']}")
        else:
            print(f"\n[RÉPONSE]\n{event['text']}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m rag")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="indexer le PDF")
    index.add_argument("--pdf", type=config.Path, default=config.PDF_PATH)
    index.add_argument("--force", action="store_true", help="réindexer même si le fichier est inchangé")
    index.set_defaults(func=cmd_index)

    search = commands.add_parser("search", help="recherche directe, sans LLM")
    search.add_argument("query", nargs="?", help="requête texte")
    search.add_argument("--image", help="image à utiliser comme requête")
    search.add_argument("-k", type=int, default=5)
    search.add_argument("--type", choices=["text", "image", "all"], default="all")
    search.set_defaults(func=cmd_search)

    ask = commands.add_parser("ask", help="poser une question à l'agent")
    ask.add_argument("question")
    ask.add_argument("--image", help="image jointe à la question")
    ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
