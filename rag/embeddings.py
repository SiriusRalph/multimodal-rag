"""Étape 2 : vectorisation du texte et des images dans l'espace latent de CLIP."""
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

from . import config


def pad_to_square(image: Image.Image) -> Image.Image:
    """Complète l'image en carré blanc : le recadrage central de CLIP couperait les schémas hauts."""
    side = max(image.size)
    square = Image.new("RGB", (side, side), "white")
    square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return square


def load_rgb(path: Path) -> Image.Image:
    """Charge une image en aplatissant la transparence sur un fond blanc."""
    image = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    return Image.alpha_composite(background, image).convert("RGB")


class ClipEmbedder:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _batches(self, items: list) -> Iterable[list]:
        for start in range(0, len(items), self.batch_size):
            yield items[start : start + self.batch_size]

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        return features / features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        outputs = [
            self._normalize(self.model.encode_text(self.tokenizer(batch))).numpy()
            for batch in self._batches(texts)
        ]
        return np.vstack(outputs)

    @torch.no_grad()
    def embed_images(self, paths: list[Path]) -> np.ndarray:
        outputs = []
        for batch in self._batches(paths):
            pixels = torch.stack([self.preprocess(pad_to_square(load_rgb(p))) for p in batch])
            outputs.append(self._normalize(self.model.encode_image(pixels)).numpy())
        return np.vstack(outputs)
