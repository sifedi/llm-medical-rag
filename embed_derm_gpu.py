import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ===== CONFIG =====
DATA_PATH = Path("data/derm_chunks.jsonl")
OUT_EMB = Path("data/derm_embeddings.npy")
OUT_META = Path("data/derm_metadata.json")

MODEL_NAME = "intfloat/e5-large-v2"
BATCH_SIZE = 64  # augmente si tu as assez de VRAM (ex: 128)
DOC_PREFIX = "passage: "  # E5 doc prefix

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing: {DATA_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} ...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    texts = []
    metadata = []

    print("Loading data...")
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = (obj.get("text") or "").strip()
            if not text:
                continue

            chunk_id = obj.get("id")
            if not chunk_id:
                # sécurité: si jamais un id manque
                continue

            texts.append(DOC_PREFIX + text)
            metadata.append({
                "id": chunk_id,
                "title": obj.get("title"),
                "journal": obj.get("journal"),
                "year": obj.get("year"),
                "pmcid": obj.get("pmcid"),
                "doi": obj.get("doi"),
                "chunk_index": obj.get("chunk_index"),
                "source": obj.get("source"),
                "query": obj.get("query"),
            })

    print(f"Chunks loaded: {len(texts)}")
    if not texts:
        raise RuntimeError("No texts loaded from JSONL.")

    print("Computing embeddings (GPU if available)...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    print("Embeddings shape:", embeddings.shape)

    OUT_EMB.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_EMB, embeddings)

    OUT_META.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"DONE\nEmbeddings: {OUT_EMB}\nMetadata: {OUT_META}")

if __name__ == "__main__":
    main()
