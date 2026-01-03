import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

import ollama
from ollama import ResponseError

# ===== PATHS =====
INDEX_PATH = Path("data/derm_faiss.index")
META_PATH  = Path("data/derm_metadata.json")
CHUNKS_PATH = Path("data/derm_chunks.jsonl")

# ===== MODELS =====
EMBED_MODEL = "intfloat/e5-large-v2"
# LLM Ollama (choisis un modèle léger compatible 16GB RAM)
OLLAMA_MODEL = "medllama2:latest"   # ou "biomistral:latest"

TOP_K = 5
MAX_CONTEXT_CHARS = 4500  # limite de contexte pour éviter trop de tokens/RAM
TEMPERATURE = 0.2

DOC_PREFIX = "passage: "
QUERY_PREFIX = "query: "


def load_chunks_map(jsonl_path: Path):
    mp = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            _id = obj.get("id")
            if _id:
                mp[_id] = obj.get("text", "")
    return mp


def build_rag_context(hits, chunks_map):
    """hits = list of dict: {id, pmcid, title, journal, year, score}"""
    parts = []
    total = 0
    for i, h in enumerate(hits, start=1):
        chunk_id = h["id"]
        txt = chunks_map.get(chunk_id, "").strip()
        if not txt:
            continue

        header = f"[Source {i}] PMCID={h.get('pmcid')} | {h.get('title')} | {h.get('journal')} ({h.get('year')}) | chunk_id={chunk_id}\n"
        block = header + txt + "\n"

        # tronquer si ça devient trop long
        if total + len(block) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total
            if remaining > 200:
                parts.append(block[:remaining] + "\n")
            break

        parts.append(block)
        total += len(block)

    return "\n".join(parts).strip()


def main():
    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_PATH))

    print("Loading metadata...")
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

    print("Loading chunks map...")
    chunks_map = load_chunks_map(CHUNKS_PATH)

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"Using Ollama model: {OLLAMA_MODEL}")

    while True:
        q = input("\nQuestion (enter pour quitter): ").strip()
        if not q:
            break

        # ===== RETRIEVE =====
        q_vec = embedder.encode(QUERY_PREFIX + q, normalize_embeddings=True)
        q_vec = np.asarray([q_vec], dtype="float32")
        D, I = index.search(q_vec, TOP_K)

        hits = []
        for score, idx in zip(D[0], I[0]):
            m = metadata[idx]
            hits.append({
                "score": float(score),
                "id": m.get("id"),
                "pmcid": m.get("pmcid"),
                "title": m.get("title"),
                "journal": m.get("journal"),
                "year": m.get("year"),
            })

        print("\nTop results (retriever):")
        for i, h in enumerate(hits, start=1):
            preview = (chunks_map.get(h["id"], "")[:140] + "...").replace("\n", " ")
            print(f"#{i} score={h['score']:.4f} id={h['id']} pmcid={h['pmcid']} title={h['title']}\n  preview: {preview}\n")

        rag_context = build_rag_context(hits, chunks_map)

        system_prompt = (
            "Tu es un assistant médical éducatif. "
            "Tu dois répondre en FRANÇAIS, clairement, et rester prudent. "
            "Tu utilises uniquement le CONTEXTE fourni. "
            "Si le contexte ne suffit pas, dis-le et propose quoi chercher. "
            "Ne donne pas de diagnostic. Ajoute une courte note de sécurité."
        )

        user_prompt = f"""QUESTION:
{q}

CONTEXTE (extraits d'articles):
{rag_context}

INSTRUCTIONS:
- Réponds en français.
- Donne une définition simple d'abord, puis une explication plus détaillée.
- Cite les sources sous forme: (PMCID=..., chunk_id=...).
- Termine par une courte note: "Ceci ne remplace pas un avis médical."
"""

        # ===== GENERATE =====
        try:
            resp = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    # réduire la mémoire/charge
                    "temperature": TEMPERATURE,
                    "num_ctx": 2048,
                    "num_predict": 512,
                },
            )
            answer = resp["message"]["content"]
            print("\n===== RAG ANSWER =====\n")
            print(answer)

        except ResponseError as e:
            msg = str(e)
            print("\n[ERROR] Ollama failed:", msg)

            if "requires more system memory" in msg.lower():
                print("\n👉 Ton modèle Ollama est trop gros pour ta RAM (16 Go).")
                print("✅ Solution rapide (modèles légers):")
                print("   - ollama pull medllama2:latest")
                print("   - ollama pull biomistral:latest")
                print("Puis relance le script.")
            else:
                print("\n👉 Vérifie que Ollama tourne et que le modèle existe:")
                print("   - ollama serve")
                print("   - ollama list")


if __name__ == "__main__":
    main()
