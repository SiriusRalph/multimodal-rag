"""Étape 4 : recherche multimodale par similarité cosinus et restitution des résultats."""
import html
import re
from dataclasses import dataclass
from itertools import chain, zip_longest
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .embeddings import ClipEmbedder
from .store import VectorStore

ALL_IMAGES = 100
FUSION_CANDIDATES = 30
RRF_CONSTANT = 60


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class SearchResult:
    content_type: str
    score: float
    page: int
    content: str
    image_path: str | None = None
    figure: int | None = None

    def label(self) -> str:
        if self.content_type == "image":
            return f"Figure {self.figure}" if self.figure else "Image"
        return "Texte"


class Retriever:
    def __init__(self, store: VectorStore, embedder: ClipEmbedder):
        self.store = store
        self.embedder = embedder
        self._bm25 = None

    def _search_type(self, embedding, k: int, content_type: str) -> list[SearchResult]:
        raw = self.store.query(embedding, k, content_type)
        results = []
        for content, metadata, distance in zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]):
            results.append(
                SearchResult(
                    content_type=content_type,
                    score=1.0 - distance,
                    page=metadata["page"],
                    content=content,
                    image_path=metadata.get("image_path"),
                    figure=metadata.get("figure") or None,
                )
            )
        return results

    def _bm25_index(self) -> tuple[BM25Okapi, list[str], list[dict]]:
        if self._bm25 is None:
            corpus = self.store.collection.get(where={"content_type": "text"}, include=["documents", "metadatas"])
            tokens = [tokenize(text) for text in corpus["documents"]]
            self._bm25 = (BM25Okapi(tokens), corpus["documents"], corpus["metadatas"])
        return self._bm25

    def _search_text(self, query: str | None, embedding, k: int, use_lexical: bool) -> list[SearchResult]:
        """Fusionne CLIP et BM25 par rangs réciproques (RRF).

        Les vecteurs CLIP captent le thème d'un passage mais pas les faits précis
        comme « h = 8 » : BM25 retrouve ces mots exacts. Une requête image reste
        purement vectorielle, faute de mots à comparer.
        """
        if query is None or not use_lexical:
            return self._search_type(embedding, k, "text")

        bm25, documents, metadatas = self._bm25_index()
        dense = self._search_type(embedding, FUSION_CANDIDATES, "text")
        order = np.argsort(bm25.get_scores(tokenize(query)))[::-1][:FUSION_CANDIDATES]

        by_content = {r.content: r for r in dense}
        fused = {r.content: 1 / (RRF_CONSTANT + rank) for rank, r in enumerate(dense, 1)}
        for rank, index in enumerate(order, 1):
            content = documents[index]
            fused[content] = fused.get(content, 0.0) + 1 / (RRF_CONSTANT + rank)
            by_content.setdefault(content, SearchResult("text", 0.0, metadatas[index]["page"], content))

        ranked = sorted(fused, key=fused.get, reverse=True)[:k]
        results = [by_content[content] for content in ranked]
        for result in results:
            result.score = fused[result.content]
        return results

    def _search_images_by_text(self, embedding, k: int, use_captions: bool) -> list[SearchResult]:
        """Classe les images selon la moyenne des cosinus visuel et légende.

        CLIP distingue mal des schémas techniques à partir du seul visuel ; la légende
        du PDF apporte le vocabulaire qui manque, sans remplacer le vecteur de l'image.
        """
        if not use_captions:
            return self._search_type(embedding, k, "image")
        results = self._search_type(embedding, ALL_IMAGES, "image")
        if not results:
            return []
        captions = self.embedder.embed_texts([r.content for r in results])
        for result, caption_score in zip(results, captions @ embedding):
            result.score = (result.score + float(caption_score)) / 2
        return sorted(results, key=lambda r: r.score, reverse=True)[:k]

    def search(self, query: str | None = None, image_path: str | Path | None = None,
               k: int = 5, content_type: str = "all", use_captions: bool = True,
               use_lexical: bool = True) -> list[SearchResult]:
        """Retourne les k meilleurs résultats pour une requête texte ou image.

        En mode "all", chaque modalité est classée séparément puis les listes sont
        alternées : les similarités texte-texte de CLIP sont nettement plus élevées
        que les similarités texte-image, et un classement unique écarterait les images.
        """
        if (query is None) == (image_path is None):
            raise ValueError("Fournir soit une requête texte, soit une image.")
        if content_type not in {"text", "image", "all"}:
            raise ValueError(f"content_type inconnu : {content_type}")

        if query is not None:
            embedding = self.embedder.embed_texts([query])[0]
        else:
            path = Path(image_path)
            if not path.is_file():
                raise FileNotFoundError(f"Image introuvable : {path}")
            embedding = self.embedder.embed_images([path])[0]

        def search_images() -> list[SearchResult]:
            if query is None:
                return self._search_type(embedding, k, "image")
            return self._search_images_by_text(embedding, k, use_captions)

        if content_type == "image":
            return search_images()
        if content_type == "text":
            return self._search_text(query, embedding, k, use_lexical)

        texts = self._search_text(query, embedding, k, use_lexical)
        images = search_images()
        merged = chain.from_iterable(zip_longest(images, texts))
        return [r for r in merged if r is not None][:k]


def format_results(results: list[SearchResult]) -> str:
    """Met en forme les résultats pour le terminal et pour l'agent."""
    if not results:
        return "Aucun résultat."
    lines = []
    for rank, r in enumerate(results, 1):
        lines.append(f"[{rank}] {r.label()} | page {r.page} | score {r.score:.3f}")
        if r.image_path:
            lines.append(f"    fichier : {r.image_path}")
        lines.append(f"    {r.content}")
    return "\n".join(lines)


def write_html_preview(results: list[SearchResult], title: str, path: Path) -> Path:
    """Écrit une page de résultats avec les aperçus d'images."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for rank, r in enumerate(results, 1):
        picture = ""
        if r.image_path:
            uri = (config.ROOT / r.image_path).resolve().as_uri()
            picture = f'<img src="{uri}" alt="{html.escape(r.label())}">'
        cards.append(
            f"<article><h2>{rank}. {html.escape(r.label())}, page {r.page} "
            f"<small>score {r.score:.3f}</small></h2>{picture}<p>{html.escape(r.content)}</p></article>"
        )
    path.write_text(
        '<!doctype html><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:system-ui;max-width:760px;margin:2rem auto;padding:0 1rem}"
        "article{border-top:1px solid #ccc;padding:1rem 0}h2{font-size:1rem}"
        "small{color:#666;font-weight:400}img{max-width:100%;max-height:420px}</style>"
        f"<h1>{html.escape(title)}</h1>{''.join(cards)}",
        encoding="utf-8",
    )
    return path
