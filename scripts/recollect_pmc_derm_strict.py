import json
import re
import time
import socket
import http.client
from pathlib import Path
import xml.etree.ElementTree as ET

from Bio import Entrez
from tqdm import tqdm

# =========================
# CONFIG
# =========================
Entrez.email = "YOUR_EMAIL@example.com"
# Entrez.api_key = "YOUR_NCBI_API_KEY"  # optionnel

QUERY = 'dermatology AND "open access"[filter]'

TARGET_ARTICLES_WRITTEN = 1000  # on boucle jusqu'à écrire 1000 articles (pas juste tenter)
PAGE_SIZE = 10
SLEEP_BETWEEN_REQUESTS = 0.5
MAX_RETRIES = 8
SOCKET_TIMEOUT = 30  # ✅ évite les "bloquages" longs

CHUNK_WORDS = 450
OVERLAP_WORDS = 60
MIN_DOC_WORDS = 200

OUT_JSONL = Path("data/derm_chunks.jsonl")

# =========================
# Helpers
# =========================
def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def itertext(elem) -> str:
    if elem is None:
        return ""
    return norm_space(" ".join(elem.itertext()))

def get_first_text(article, xpaths):
    for xp in xpaths:
        el = article.find(xp)
        txt = itertext(el)
        if txt:
            return txt
    return ""

def extract_pmcid(article) -> str:
    pmcid = get_first_text(article, [
        ".//front//article-meta//article-id[@pub-id-type='pmc']",
        ".//front//article-meta//article-id[@pub-id-type='pmcid']",
        ".//article-meta//article-id[@pub-id-type='pmc']",
        ".//article-meta//article-id[@pub-id-type='pmcid']",
    ])
    pmcid = pmcid.strip()
    if not pmcid:
        return ""
    if pmcid.isdigit():
        pmcid = "PMC" + pmcid
    m = re.search(r"(PMC\d+)", pmcid)
    return m.group(1) if m else ""

def chunk_words(text: str, chunk_size: int, overlap: int):
    words = text.split()
    if not words:
        return []
    out = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        out.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return out

def backoff_sleep(attempt: int):
    time.sleep(min(30, (2 ** attempt)) + 0.2)

def safe_read(handle) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            data = handle.read()
            if isinstance(data, bytes):
                data = data.decode("utf-8", errors="replace")
            return data
        except (http.client.IncompleteRead, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
            print(f"[WARN] read failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            backoff_sleep(attempt)
    raise RuntimeError("Failed to read after retries")

def parse_articles(xml_text: str):
    try:
        root = ET.fromstring("<root>" + xml_text + "</root>")
        return root.findall(".//article")
    except Exception:
        arts = re.findall(r"(<article\b.*?</article>)", xml_text, flags=re.DOTALL)
        parsed = []
        for a in arts:
            try:
                parsed.append(ET.fromstring(a))
            except Exception:
                pass
        return parsed

def extract_fields(article):
    pmcid = extract_pmcid(article)
    title = get_first_text(article, [".//front//article-title", ".//article-title"])
    journal = get_first_text(article, [".//front//journal-title", ".//journal-title"])
    year = get_first_text(article, [".//front//pub-date//year", ".//pub-date//year"])
    doi = get_first_text(article, [
        ".//front//article-meta//article-id[@pub-id-type='doi']",
        ".//article-meta//article-id[@pub-id-type='doi']",
    ])
    abstract = get_first_text(article, [".//front//abstract", ".//abstract"])
    body = itertext(article.find(".//body"))

    parts = []
    if title:
        parts.append(title)
    if abstract:
        parts.append("Abstract: " + abstract)
    if body:
        parts.append(body)
    full_text = "\n\n".join(parts)

    return {
        "pmcid": pmcid,
        "title": title,
        "journal": journal,
        "year": year,
        "doi": doi,
        "full_text": full_text,
    }

# =========================
# Main
# =========================
def main():
    socket.setdefaulttimeout(SOCKET_TIMEOUT)  # ✅ important

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    print("Searching PMC...")
    h = Entrez.esearch(db="pmc", term=QUERY, usehistory="y", retmax=0)
    r = Entrez.read(h)
    webenv = r["WebEnv"]
    query_key = r["QueryKey"]
    total = int(r["Count"])
    print(f"PMC count: {total}")

    articles_seen = 0
    articles_written = 0
    chunks_written = 0
    skipped_no_pmcid = 0
    skipped_short = 0

    retstart = 0

    with OUT_JSONL.open("w", encoding="utf-8") as fout:
        pbar = tqdm(total=TARGET_ARTICLES_WRITTEN, desc="Articles written")

        while articles_written < TARGET_ARTICLES_WRITTEN and retstart < total:
            retmax = PAGE_SIZE

            xml_text = None
            for attempt in range(MAX_RETRIES):
                try:
                    fh = Entrez.efetch(
                        db="pmc",
                        webenv=webenv,
                        query_key=query_key,
                        retstart=retstart,
                        retmax=retmax,
                        rettype="full",
                        retmode="xml",
                    )
                    xml_text = safe_read(fh)
                    break
                except Exception as e:
                    print(f"[WARN] efetch failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                    backoff_sleep(attempt)

            if xml_text is None:
                print("[WARN] page failed permanently, skipping retstart:", retstart)
                retstart += retmax
                continue

            articles = parse_articles(xml_text)
            if not articles:
                # parfois page vide (rare) -> avancer
                retstart += retmax
                time.sleep(SLEEP_BETWEEN_REQUESTS)
                continue

            for art in articles:
                if articles_written >= TARGET_ARTICLES_WRITTEN:
                    break

                articles_seen += 1
                fields = extract_fields(art)

                pmcid = fields["pmcid"]
                if not pmcid:
                    skipped_no_pmcid += 1
                    continue

                full_text = fields["full_text"]
                if len(full_text.split()) < MIN_DOC_WORDS:
                    skipped_short += 1
                    continue

                chunks = chunk_words(full_text, CHUNK_WORDS, OVERLAP_WORDS)
                if not chunks:
                    skipped_short += 1
                    continue

                for i, ch in enumerate(chunks):
                    rec = {
                        "id": f"pmc:{pmcid}#chunk{i}",
                        "source": "PMC",
                        "query": QUERY,
                        "title": fields["title"],
                        "journal": fields["journal"],
                        "year": fields["year"],
                        "pmcid": pmcid,
                        "doi": fields["doi"],
                        "chunk_index": i,
                        "text": ch,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    chunks_written += 1

                articles_written += 1
                pbar.update(1)

            retstart += retmax
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        pbar.close()

    # Check final
    empty = 0
    total_lines = 0
    with OUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            obj = json.loads(line)
            if not obj.get("pmcid"):
                empty += 1

    print("\nDONE")
    print(f"JSONL: {OUT_JSONL}")
    print(f"articles_seen: {articles_seen}")
    print(f"articles_written: {articles_written}")
    print(f"skipped_no_pmcid: {skipped_no_pmcid}")
    print(f"skipped_short: {skipped_short}")
    print(f"chunks_written: {chunks_written}")
    print(f"CHECK: total_chunks={total_lines}, pmcid_empty={empty}")

if __name__ == "__main__":
    main()
