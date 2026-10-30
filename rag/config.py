"""Paramètres centralisés du projet."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PDF_PATH = DATA_DIR / "Attention is all you need.pdf"
IMAGE_DIR = DATA_DIR / "images"
DB_DIR = DATA_DIR / "chroma"
OUTPUT_DIR = ROOT / "outputs"

COLLECTION = "attention_multimodal"

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# CLIP accepte 77 jetons, dont 2 réservés au début et à la fin de séquence.
CHUNK_MAX_TOKENS = 75
CHUNK_OVERLAP_TOKENS = 15

MIN_IMAGE_SIDE = 64
FIGURE_DPI = 200

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_NUM_CTX = 8192
MAX_AGENT_STEPS = 10
