import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("data/derm_faiss.index")
META_PATH  = Path("data/derm_metadata.json")
CHUNKS_PATH = Path("data/derm_chunks.jsonl")

MODEL_NAME = "intfloat/e5-large-v2"
TOP_K = 5
QUERY_PREFIX = "query: "  # E5 query prefix

# Filtre derm (optionnel mais recommandé)
DERM_KEYS = [
    "dermat", "skin", "psoriasis", "eczema", "melanoma", "acne", "urticaria",
    "vitiligo", "pemph", "alopecia", "dermis", "cutaneous", "atopic dermatitis"
]

def load_chunks_map(jsonl_path: Path):
    mp = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj.get("id")
            if not cid:
                continue
            mp[cid] = obj.get("text", "")
    return mp

def looks_derm(m):
    """Heuristique simple: titre/journal/id/text contiennent des mots derm."""
    blob = " ".join([
        str(m.get("title") or ""),
        str(m.get("journal") or ""),
        str(m.get("id") or ""),
    ]).lower()
    return any(k in blob for k in DERM_KEYS)

def main():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing index: {INDEX_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing metadata: {META_PATH}")
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing chunks jsonl: {CHUNKS_PATH}")

    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_PATH))

    print("Loading metadata...")
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

    print("Loading chunks map...")
    chunks_map = load_chunks_map(CHUNKS_PATH)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # IMPORTANT: si ton index est en inner product, tes embeddings doivent être normalisés
    # et query aussi (on fait normalize_embeddings=True ci-dessous)

    while True:
        q = input("\nQuestion (enter pour quitter): ").strip()
        if not q:
            break

        q_vec = model.encode(QUERY_PREFIX + q, normalize_embeddings=True)
        q_vec = np.asarray([q_vec], dtype="float32")

        D, I = index.search(q_vec, max(TOP_K * 5, TOP_K))  # on récupère plus, puis on filtre

        print("\nTop results:")
        shown = 0

        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            if idx < 0 or idx >= len(metadata):
                continue

            m = metadata[idx]

            # Filtre derm (désactive si tu veux)
            if not looks_derm(m):
                continue

            chunk_id = m.get("id")
            text = chunks_map.get(chunk_id, "")
            preview = (text[:400] + "...") if len(text) > 400 else text

            shown += 1
            print(f"\n#{shown}  score={float(score):.4f}")
            print(f"  id: {chunk_id}")
            print(f"  pmcid: {m.get('pmcid')}")
            print(f"  title: {m.get('title')}")
            print(f"  journal: {m.get('journal')} ({m.get('year')})")
            print(f"  text: {preview}")

            if shown >= TOP_K:
                break

        if shown == 0:
            print("\n(No derm-like results found. Try a different question or disable the filter.)")

if __name__ == "__main__":
    main()
