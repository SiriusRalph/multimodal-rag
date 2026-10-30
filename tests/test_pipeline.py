"""Tests du pipeline sans LLM."""
import shutil

import numpy as np
import pytest

from rag import config
from rag.embeddings import ClipEmbedder
from rag.extraction import extract_pdf
from rag.retrieval import Retriever
from rag.store import VectorStore, document_id, index_pdf


@pytest.fixture(scope="module")
def embedder():
    return ClipEmbedder()


@pytest.fixture(scope="module")
def extracted(embedder, tmp_path_factory):
    return extract_pdf(config.PDF_PATH, tmp_path_factory.mktemp("images"), embedder.count_tokens)


@pytest.fixture()
def indexed(embedder, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGE_DIR", config.ROOT / "data" / "images")
    store = VectorStore(tmp_path / "db", "test")
    report = index_pdf(config.PDF_PATH, store, embedder)
    return store, report


def test_extraction_finds_raster_and_vector_figures(extracted):
    _, images = extracted
    assert sorted(i.figure for i in images) == [1, 2, 2, 3, 4, 5]
    assert {i.origin for i in images} == {"embedded", "rendered"}
    assert all(i.caption.startswith("Figure") for i in images)


def test_chunks_fit_clip_context(extracted, embedder):
    chunks, _ = extracted
    assert {c.page for c in chunks} == set(range(1, 16))
    assert max(embedder.count_tokens(c.text) for c in chunks) <= config.CHUNK_MAX_TOKENS


def test_figure_text_is_excluded_from_chunks(extracted):
    chunks, _ = extracted
    assert not any("<pad>" in c.text for c in chunks)


def test_embeddings_are_normalized_in_shared_space(embedder):
    texts = embedder.embed_texts(["a diagram", "attention"])
    images = embedder.embed_images([sorted(config.IMAGE_DIR.glob("*.png"))[0]])
    assert texts.shape[1] == images.shape[1]
    assert np.allclose(np.linalg.norm(np.vstack([texts, images]), axis=1), 1.0, atol=1e-5)


def test_reindexing_is_skipped_then_replaces_changed_document(indexed, embedder, tmp_path):
    store, first = indexed
    assert first.status == "indexed"
    assert index_pdf(config.PDF_PATH, store, embedder).status == "unchanged"

    changed = tmp_path / config.PDF_PATH.name
    shutil.copy(config.PDF_PATH, changed)
    with changed.open("ab") as file:
        file.write(b"\n% nouvelle version\n")
    assert document_id(changed) != first.doc_id

    second = index_pdf(changed, store, embedder)
    assert second.status == "indexed"
    assert store.count(first.doc_id, "text") == 0
    assert store.count(second.doc_id, "text") == first.texts


def test_metadata_and_image_query(indexed, embedder):
    store, report = indexed
    assert (report.texts, report.images) == (store.count(report.doc_id, "text"), 6)
    retriever = Retriever(store, embedder)
    path = config.ROOT / "data" / "images" / "page03_xref128.png"
    best = retriever.search(image_path=path, k=1, content_type="image")[0]
    assert (best.figure, best.page) == (1, 3)
    assert best.score == pytest.approx(1.0, abs=1e-4)


def test_search_requires_exactly_one_query(indexed, embedder):
    retriever = Retriever(indexed[0], embedder)
    with pytest.raises(ValueError):
        retriever.search()
    with pytest.raises(ValueError):
        retriever.search("x", "y.png")
