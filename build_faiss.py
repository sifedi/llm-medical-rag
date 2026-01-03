import json
import numpy as np
import faiss
from pathlib import Path

EMB_PATH = Path("data/derm_embeddings.npy")
META_PATH = Path("data/derm_metadata.json")
OUT_INDEX = Path("data/derm_faiss.index")

def main():
    print("Loading embeddings...")
    X = np.load(EMB_PATH).astype("float32")   # shape: (N, d)
    print("Embeddings:", X.shape)

    print("Loading metadata...")
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    assert len(metadata) == X.shape[0], "metadata length != embeddings rows"

    d = X.shape[1]

    # On utilise cosine similarity:
    # - embeddings normalisés => cosine = dot product
    # - IndexFlatIP = inner product
    print("Building FAISS index (IndexFlatIP)...")
    index = faiss.IndexFlatIP(d)
    index.add(X)

    print("Index size:", index.ntotal)
    faiss.write_index(index, str(OUT_INDEX))
    print("Saved index to:", OUT_INDEX)

if __name__ == "__main__":
    main()
