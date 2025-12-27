import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ===== CONFIG =====
DATA_PATH = "data/derm_chunks.jsonl"
OUT_EMB = "data/derm_embeddings.npy"
OUT_META = "data/derm_metadata.json"

MODEL_NAME = "intfloat/e5-large-v2"
BATCH_SIZE = 32

# ===== LOAD MODEL =====
print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

# E5 needs prefixes
DOC_PREFIX = "passage: "

texts = []
metadata = []

print("Loading data...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        text = obj.get("text", "").strip()
        if not text:
            continue

        texts.append(DOC_PREFIX + text)
        metadata.append({
            "id": obj.get("id"),
            "title": obj.get("title"),
            "journal": obj.get("journal"),
            "year": obj.get("year"),
            "pmcid": obj.get("pmcid"),
            "doi": obj.get("doi"),
        })

print(f"Chunks loaded: {len(texts)}")

# ===== EMBEDDING =====
print("Computing embeddings...")
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(embeddings, dtype="float32")

print("Embeddings shape:", embeddings.shape)

# ===== SAVE =====
np.save(OUT_EMB, embeddings)

with open(OUT_META, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("DONE")
