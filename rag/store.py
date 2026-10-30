"""Étape 3 : indexation dans ChromaDB, avec réindexation pilotée par SHA-256."""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np

from . import config
from .embeddings import ClipEmbedder
from .extraction import extract_pdf


def document_id(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with pdf_path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class IndexReport:
    doc_id: str
    status: str  # "indexed" ou "unchanged"
    texts: int
    images: int


class VectorStore:
    def __init__(self, path: Path = config.DB_DIR, collection: str = config.COLLECTION):
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def count(self, doc_id: str, content_type: str) -> int:
        where = {"$and": [{"doc_id": doc_id}, {"content_type": content_type}]}
        return len(self.collection.get(where=where, include=[])["ids"])

    def add(self, ids: list[str], embeddings: np.ndarray, documents: list[str], metadatas: list[dict]) -> None:
        self.collection.add(ids=ids, embeddings=embeddings.tolist(), documents=documents, metadatas=metadatas)

    def query(self, embedding: np.ndarray, k: int, content_type: str) -> dict:
        """Interroge un seul type de contenu ; `distances` vaut 1 - similarité cosinus."""
        available = len(self.collection.get(where={"content_type": content_type}, include=[])["ids"])
        return self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(k, available),
            where={"content_type": content_type},
            include=["documents", "metadatas", "distances"],
        ) if available else {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


def index_pdf(pdf_path: Path, store: VectorStore, embedder: ClipEmbedder, force: bool = False) -> IndexReport:
    """Indexe le PDF. Un fichier inchangé est ignoré ; une nouvelle version remplace l'ancienne."""
    doc_id = document_id(pdf_path)
    if not force and store.count(doc_id, "text"):
        return IndexReport(doc_id, "unchanged", store.count(doc_id, "text"), store.count(doc_id, "image"))

    # La suppression vise le nom de fichier pour retirer aussi les versions précédentes.
    store.collection.delete(where={"source": pdf_path.name})
    chunks, images = extract_pdf(pdf_path, config.IMAGE_DIR, embedder.count_tokens)

    base = {"doc_id": doc_id, "source": pdf_path.name}
    store.add(
        ids=[f"{doc_id[:16]}:text:{c.chunk_index}" for c in chunks],
        embeddings=embedder.embed_texts([c.text for c in chunks]),
        documents=[c.text for c in chunks],
        metadatas=[{**base, "content_type": "text", "page": c.page, "chunk_index": c.chunk_index} for c in chunks],
    )
    store.add(
        ids=[f"{doc_id[:16]}:image:{i}" for i in range(len(images))],
        embeddings=embedder.embed_images([img.path for img in images]),
        documents=[img.caption or f"Image de la page {img.page}" for img in images],
        metadatas=[
            {
                **base,
                "content_type": "image",
                "page": img.page,
                "image_path": str(img.path.relative_to(config.ROOT)),
                "figure": img.figure or 0,
                "origin": img.origin,
            }
            for img in images
        ],
    )
    return IndexReport(doc_id, "indexed", len(chunks), len(images))
