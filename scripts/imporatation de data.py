from Bio import Entrez
import time
import json
import re
import http.client
import xml.etree.ElementTree as ET
from typing import Optional, Dict, List

# =========================
# CONFIG
# =========================
Entrez.email = "fedibouattour22@gmail.com"
# Entrez.api_key = "VOTRE_CLE_API"  # optionnel mais recommandé

QUERY = 'dermatology AND "open access"[filter]'
TARGET_N = 1000
FETCH_PAGE = 10
SLEEP_SEC = 0.8
MAX_RETRIES = 6

OUT_JSONL = r"E:\rag\derm_chunks.jsonl"

CHUNK_WORDS = 450
OVERLAP_WORDS = 60
MIN_DOC_WORDS = 250


# =========================
# HELPERS
# =========================
def backoff(attempt: int):
    time.sleep((2 ** attempt) * 1.0)

def safe_read(handle):
    for attempt in range(MAX_RETRIES):
        try:
            data = handle.read()
            # ✅ FIX CRITIQUE: toujours convertir en str ici
            if isinstance(data, bytes):
                data = data.decode("utf-8", errors="replace")
            return data
        except (http.client.IncompleteRead, ConnectionError, TimeoutError) as e:
            print(f"[WARN] read error -> retry {attempt+1}/{MAX_RETRIES}: {e}")
            backoff(attempt)
    raise RuntimeError("Echec read() après retries")

def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def itertext(elem):
    if elem is None:
        return ""
    return norm_text(" ".join(elem.itertext()))

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

def extract_articles(xml_text: str):
    """
    ✅ Robust extraction:
    - d'abord tentative XML wrap
    - sinon fallback regex <article>...</article>
    """
    try:
        wrapped = "<root>" + xml_text + "</root>"
        root = ET.fromstring(wrapped)
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

def get_text(article, xpath: str) -> str:
    el = article.find(xpath)
    return itertext(el)

def extract_fields(article):
    title = get_text(article, ".//front//article-title")
    abstract = get_text(article, ".//front//abstract")
    body = itertext(article.find(".//body"))

    pmcid = get_text(article, ".//front//article-meta//article-id[@pub-id-type='pmc']")
    doi = get_text(article, ".//front//article-meta//article-id[@pub-id-type='doi']")
    journal = get_text(article, ".//front//journal-title")
    year = get_text(article, ".//front//pub-date//year")

    lic_el = article.find(".//permissions//license")
    if lic_el is None:
        lic_el = article.find(".//license")

    license_type = ""
    if lic_el is not None:
        license_type = lic_el.attrib.get("license-type", "")

    license_text = itertext(lic_el)

    return {
        "title": title,
        "abstract": abstract,
        "body": body,
        "pmcid": pmcid,
        "doi": doi,
        "journal": journal,
        "year": year,
        "license_type": norm_text(license_type),
        "license_text": norm_text(license_text),
    }


# =========================
# MAIN
# =========================
def main():
    h = Entrez.esearch(db="pmc", term=QUERY, usehistory="y", retmax=0)
    r = Entrez.read(h)
    webenv = r["WebEnv"]
    query_key = r["QueryKey"]
    total = int(r["Count"])

    print("PMC count:", total)
    to_fetch = min(TARGET_N, total)
    print("Will fetch:", to_fetch)

    articles_written = 0
    chunks_written = 0

    with open(OUT_JSONL, "w", encoding="utf-8") as fout:
        retstart = 0
        while retstart < to_fetch:
            retmax = min(FETCH_PAGE, to_fetch - retstart)

            # efetch retries
            xml_text = None
            for attempt in range(MAX_RETRIES):
                try:
                    fh = Entrez.efetch(
                        db="pmc",
                        rettype="full",
                        retmode="xml",
                        retstart=retstart,
                        retmax=retmax,
                        webenv=webenv,
                        query_key=query_key,
                    )
                    xml_text = safe_read(fh)  # ✅ toujours str ici
                    break
                except Exception as e:
                    print(f"[WARN] efetch error -> retry {attempt+1}/{MAX_RETRIES}: {e}")
                    backoff(attempt)

            if xml_text is None:
                raise RuntimeError("efetch page failed after retries")

            # parse articles from this page
            arts = extract_articles(xml_text)
            if not arts:
                print("[WARN] page had 0 parsable articles, continue")
                retstart += retmax
                time.sleep(SLEEP_SEC)
                continue

            for art in arts:
                fields = extract_fields(art)
                full_text = "\n\n".join([t for t in [fields["title"], fields["abstract"], fields["body"]] if t])

                if len(full_text.split()) < MIN_DOC_WORDS:
                    continue

                chunks = chunk_words(full_text, CHUNK_WORDS, OVERLAP_WORDS)
                if not chunks:
                    continue

                base_id = f"pmc:{fields['pmcid']}" if fields["pmcid"] else f"offset:{retstart}"

                for i, ch in enumerate(chunks):
                    rec = {
                        "id": f"{base_id}#chunk{i}",
                        "source": "PMC",
                        "query": QUERY,
                        "title": fields["title"],
                        "journal": fields["journal"],
                        "year": fields["year"],
                        "pmcid": fields["pmcid"],
                        "doi": fields["doi"],
                        "license_type": fields["license_type"],
                        "license_text": fields["license_text"],
                        "chunk_index": i,
                        "text": ch
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    chunks_written += 1

                articles_written += 1

            retstart += retmax
            print(f"Progress: {retstart}/{to_fetch} | articles_written={articles_written} | chunks_written={chunks_written}")
            time.sleep(SLEEP_SEC)

    print("DONE")
    print("JSONL:", OUT_JSONL)
    print("articles_written:", articles_written)
    print("chunks_written:", chunks_written)

if __name__ == "__main__":
    main()