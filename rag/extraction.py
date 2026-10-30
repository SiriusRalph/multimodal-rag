"""Étape 1 : extraction du texte et des images du PDF."""
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from . import config

CAPTION_RE = re.compile(r"^Figure\s+(\d+):")


@dataclass
class TextChunk:
    text: str
    page: int
    chunk_index: int


@dataclass
class ExtractedImage:
    path: Path
    page: int
    figure: int | None
    caption: str
    origin: str  # "embedded" (image du PDF) ou "rendered" (figure vectorielle)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _captions(page: pymupdf.Page) -> list[tuple[int, str, pymupdf.Rect]]:
    """Retourne les légendes de figures de la page : (numéro, texte, position)."""
    captions = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        text = _clean(text)
        if match := CAPTION_RE.match(text):
            captions.append((int(match.group(1)), text, pymupdf.Rect(x0, y0, x1, y1)))
    return captions


def _nearest_caption(rect: pymupdf.Rect, captions: list) -> tuple[int | None, str]:
    """Associe une image à la première légende située sous elle."""
    below = [c for c in captions if c[2].y0 >= rect.y1 - 5]
    if not below:
        return None, ""
    number, text, _ = min(below, key=lambda c: c[2].y0)
    return number, text


def _save_embedded(doc: pymupdf.Document, img: tuple, path: Path) -> None:
    xref, smask = img[0], img[1]
    pixmap = pymupdf.Pixmap(doc, xref)
    if smask:
        pixmap = pymupdf.Pixmap(pixmap, pymupdf.Pixmap(doc, smask))
    if pixmap.colorspace and pixmap.colorspace.n > 3:
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
    pixmap.save(path)


def _vector_figure_rect(page: pymupdf.Page, caption_rect: pymupdf.Rect) -> pymupdf.Rect | None:
    """Calcule la zone d'une figure vectorielle : tout ce qui est tracé au-dessus de sa légende."""
    rects = [d["rect"] for d in page.get_drawings() if d["rect"].y1 <= caption_rect.y0]
    if not rects:
        return None
    region = pymupdf.Rect(rects[0])
    for rect in rects[1:]:
        region |= rect
    for x0, y0, x1, y1, *_ in page.get_text("blocks"):
        block = pymupdf.Rect(x0, y0, x1, y1)
        if block.y1 <= caption_rect.y0 and block.intersects(region):
            region |= block
    return (region + (-4, -4, 4, 4)) & page.rect


def extract_images(doc: pymupdf.Document, output_dir: Path) -> tuple[list[ExtractedImage], dict[int, list]]:
    """Sauvegarde les images du PDF et rend les figures vectorielles.

    Retourne aussi, par page, les zones de figures à exclure du texte, car les
    mots dessinés dans les figures 3 à 5 pollueraient les fragments.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images, figure_regions, seen = [], {}, set()

    for page in doc:
        number = page.number + 1
        captions = _captions(page)
        figures_with_raster = set()

        for img in page.get_images(full=True):
            xref, width, height = img[0], img[2], img[3]
            if xref in seen or min(width, height) < config.MIN_IMAGE_SIDE:
                continue
            seen.add(xref)
            rects = page.get_image_rects(xref)
            figure, caption = _nearest_caption(rects[0], captions) if rects else (None, "")
            figures_with_raster.add(figure)
            path = output_dir / f"page{number:02d}_xref{xref}.png"
            _save_embedded(doc, img, path)
            images.append(ExtractedImage(path, number, figure, caption, "embedded"))

        for figure, caption, rect in captions:
            if figure in figures_with_raster:
                continue
            region = _vector_figure_rect(page, rect)
            if region is None:
                continue
            path = output_dir / f"page{number:02d}_figure{figure}.png"
            page.get_pixmap(clip=region, dpi=config.FIGURE_DPI).save(path)
            images.append(ExtractedImage(path, number, figure, caption, "rendered"))
            figure_regions.setdefault(number, []).append(region)

    return images, figure_regions


def page_text(page: pymupdf.Page, excluded: list[pymupdf.Rect]) -> str:
    """Extrait le texte d'une page sans les numéros de page ni le texte des figures."""
    blocks = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks", flags=pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_DEHYPHENATE, sort=True):
        block = pymupdf.Rect(x0, y0, x1, y1)
        text = _clean(text)
        if text and not text.isdigit() and not any(block.intersects(r) for r in excluded):
            blocks.append(text)
    return "\n".join(blocks)


def chunk_words(text: str, count_tokens: Callable[[str], int]) -> list[str]:
    """Découpe un texte en fenêtres glissantes qui tiennent dans la limite de CLIP."""
    words = text.split()
    costs = [count_tokens(word) for word in words]
    chunks, start = [], 0
    while start < len(words):
        end, used = start, 0
        while end < len(words) and used + costs[end] <= config.CHUNK_MAX_TOKENS:
            used += costs[end]
            end += 1
        end = max(end, start + 1)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        next_start, overlap = end, 0
        while next_start - 1 > start and overlap + costs[next_start - 1] <= config.CHUNK_OVERLAP_TOKENS:
            next_start -= 1
            overlap += costs[next_start]
        start = next_start
    return chunks


def extract_pdf(pdf_path: Path, image_dir: Path, count_tokens: Callable[[str], int]) -> tuple[list[TextChunk], list[ExtractedImage]]:
    with pymupdf.open(pdf_path) as doc:
        images, figure_regions = extract_images(doc, image_dir)
        chunks = []
        for page in doc:
            number = page.number + 1
            text = page_text(page, figure_regions.get(number, []))
            for piece in chunk_words(text, count_tokens):
                chunks.append(TextChunk(piece, number, len(chunks)))
    return chunks, images
