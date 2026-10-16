# src/build_data.py
import json, re, sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "passages.json"
SEED = ROOT / "data" / "passages_seed.json"
TODAY = date.today().isoformat()

URLS = {
    "ordinary-passport": "https://dgip.gov.pk/passport/ordinary-passport.php",
    "process": "https://dgip.gov.pk/passport/process.php",
    "types": "https://dgip.gov.pk/passport/types.php",
    "online-passport": "https://www.dgip.gov.pk/eServices/online-passport.php",
    "faq": "https://dgip.gov.pk/faq/faq.php",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
NOISE = re.compile(r"(^|[-_ ])(nav|navbar|menu|footer|header|sidebar|breadcrumb|topbar|social|cookie|modal|search)([-_ ]|$)", re.I)
MIN_W, MAX_W = 100, 150


def fetch(doc_id, url):
    RAW.mkdir(parents=True, exist_ok=True)
    try:
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
        except requests.exceptions.SSLError:
            import urllib3
            urllib3.disable_warnings()
            r = requests.get(url, headers=HEADERS, timeout=30, verify=False)
        r.raise_for_status()
        if url.lower().endswith(".pdf") or "pdf" in r.headers.get("content-type", "").lower():
            import fitz
            doc = fitz.open(stream=r.content, filetype="pdf")
            text = "\n".join(p.get_text() for p in doc)
            (RAW / f"{doc_id}.pdf").write_bytes(r.content)
        else:
            (RAW / f"{doc_id}.html").write_bytes(r.content)
            text = html_to_text(r.content)
        if len(text.split()) < 50:
            raise ValueError("too little text")
        return text, "download"
    except Exception as e:
        print(f"[warn] {doc_id}: download failed ({e})", file=sys.stderr)
    txt = RAW / f"{doc_id}.txt"
    if txt.exists():
        return txt.read_text(encoding="utf-8"), "manual_txt"
    html = RAW / f"{doc_id}.html"
    if html.exists():
        return html_to_text(html.read_bytes()), "cached_html"
    print(f"[error] {doc_id}: no fallback at {txt}", file=sys.stderr)
    return "", "missing"


def html_to_text(content):
    soup = BeautifulSoup(content, "html.parser")
    for t in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "iframe", "svg", "button", "select"]):
        t.decompose()
    for t in soup.find_all(True):
        if t.attrs is None:
            continue
        ident = " ".join([t.get("id") or ""] + (t.get("class") or []))
        if ident.strip() and NOISE.search(ident):
            t.decompose()
    body = soup.body or soup
    for br in body.find_all("br"):
        br.replace_with("\n")
    for li in body.find_all("li"):
        li.insert(0, "• ")
    return body.get_text("\n")


def clean(text):
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\r", "")
    lines, prev = [], None
    for ln in text.split("\n"):
        ln = re.sub(r"[ \t]+", " ", ln).strip(" •\t")
        if not ln or ln == prev:
            continue
        if len(ln.split()) < 3 and not re.search(r"\d", ln):
            continue
        if re.search(r"(copyright|all rights reserved|visitor|last updated|skip to)", ln, re.I) and len(ln.split()) < 12:
            continue
        lines.append(ln)
        prev = ln
    return lines


def sentences(lines):
    out = []
    for ln in lines:
        if not re.search(r"[.!?۔:;]$", ln):
            ln += "."
        parts = re.split(r"(?<=[.!?۔])\s+(?=[A-Z0-9\"'(\u0600-\u06FF])", ln)
        out.extend(p.strip() for p in parts if p.strip())
    return out


def split_long(s):
    w = s.split()
    return [" ".join(w[i:i + MAX_W]) for i in range(0, len(w), MAX_W)]


def pack(sents):
    units = []
    for s in sents:
        units.extend(split_long(s) if len(s.split()) > MAX_W else [s])
    chunks, cur, n = [], [], 0
    for u in units:
        k = len(u.split())
        if cur and n + k > MAX_W and n >= MIN_W:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(u)
        n += k
    if cur:
        tail = " ".join(cur)
        if chunks and n < MIN_W:
            if len(chunks[-1].split()) + n <= MAX_W + 40:
                chunks[-1] += " " + tail
            elif n >= 30:
                chunks.append(tail)
        else:
            chunks.append(tail)
    return chunks


def main():
    passages, prov, sid = [], [], 0
    for doc_id, url in URLS.items():
        text, how = fetch(doc_id, url)
        chunks = pack(sentences(clean(text))) if text else []
        for c in chunks:
            sid += 1
            passages.append({
                "doc_id": doc_id, "url": url, "passage_id": f"S{sid:03d}",
                "text": c, "date_accessed": TODAY, "origin": "scraped",
            })
        prov.append({"doc_id": doc_id, "url": url, "passages": len(chunks), "origin": f"scraped ({how})"})

    if SEED.exists():
        seed = json.loads(SEED.read_text(encoding="utf-8"))
        counts = {}
        for p in seed:
            p["origin"] = "gold_seed"
            p.setdefault("date_accessed", TODAY)
            counts[p["doc_id"]] = counts.get(p["doc_id"], 0) + 1
        passages.extend(seed)
        for d, n in counts.items():
            u = next(p["url"] for p in seed if p["doc_id"] == d)
            prov.append({"doc_id": d, "url": u, "passages": n, "origin": "gold_seed"})
    else:
        print("[warn] data/passages_seed.json missing, run prep_dataset.py first", file=sys.stderr)

    OUT.write_text(json.dumps(passages, ensure_ascii=False, indent=1), encoding="utf-8")
    print(pd.DataFrame(prov).to_string(index=False))
    print(f"\nTotal passages: {len(passages)} -> {OUT}")


if __name__ == "__main__":
    main()