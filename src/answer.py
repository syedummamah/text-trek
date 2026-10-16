"""Policy-aware, extractive answers for the passport information corpus."""
import re

from src.normalize import normalize, prepare_query

# Similarity / retrieval policy. E5 cosine is normalized to [-1, 1].
THRESH = 0.35
DOMAIN_SEM = 0.85
BM25_MIN = 1e-8
MAX_EVIDENCE = 3
MAX_SENTENCES = 3
MIN_QUERY_WORDS = 2
RRF_K = 60

TEMPLATES = {
    "en": {
        "clarify": "Please specify the passport service and applicant category (for example, new or renewal, normal or urgent, adult or minor, and number of pages) so I can give the right information.",
        "abstain": "I could not find sufficiently reliable official information for that question. Please check the official Directorate General of Immigration & Passports website.",
        "current": "Fees can change. Please verify the current amount on the live official website. The source passage I found was accessed on {date}.",
        "conflict": "The retrieved official passages give different fee amounts; please verify the applicable fee on the live official website.",
        "no_evidence": "I could not find a reliable matching passage. Please check the official Directorate General of Immigration & Passports website.",
    },
    "ur": {
        "clarify": "براہِ کرم پاسپورٹ کی سروس اور درخواست گزار کی قسم بتائیں (مثلاً نیا یا تجدید، نارمل یا ارجنٹ، بالغ یا نابالغ، اور صفحات کی تعداد) تاکہ درست معلومات دی جا سکیں۔",
        "abstain": "اس سوال کے لیے کافی قابلِ اعتماد سرکاری معلومات نہیں ملیں۔ براہِ کرم ڈائریکٹوریٹ جنرل آف امیگریشن اینڈ پاسپورٹس کی سرکاری ویب سائٹ دیکھیں۔",
        "current": "فیس تبدیل ہو سکتی ہے۔ موجودہ رقم سرکاری ویب سائٹ پر براہِ راست تصدیق کریں۔ دستیاب ماخذ {date} کو دیکھا گیا تھا۔",
        "conflict": "تلاش کیے گئے سرکاری ماخذ فیس کی مختلف رقم بتاتے ہیں؛ براہِ کرم سرکاری ویب سائٹ پر قابلِ اطلاق فیس کی تصدیق کریں۔",
        "no_evidence": "اس سوال سے مطابقت رکھنے والا قابلِ اعتماد ماخذ نہیں ملا۔ براہِ کرم ڈائریکٹوریٹ جنرل آف امیگریشن اینڈ پاسپورٹس کی سرکاری ویب سائٹ دیکھیں۔",
    },
}

_INJECTION = re.compile(
    r"\b(ignore|disregard|forget)\b.{0,80}\b(previous|prior|above|system|instructions?)\b[^.!?]*[.!?]?",
    re.I,
)
_OUTCOME = re.compile(r"\b(will|would|can)\s+(my\s+)?(application|passport)\s+be\s+approved\b|\bapproval prediction\b", re.I)
_GUARANTEE = re.compile(r"\b(guarantee|guaranteed|exactly)\b.{0,50}\b(process(?:ing)? time|days?|hours?)\b|\bhow many days\b.{0,50}\bguarantee\b", re.I)
_CURRENT = re.compile(r"\b(still current|current today|today's fee|fee today|as of today|up[- ]to[- ]date fee)\b", re.I)
_FEE = re.compile(r"\b(fee|fees|cost|costs|price|charge|charges|kitn[iae]|kharcha)\b|فیس|قیمت|خرچہ", re.I)
_DOCS = re.compile(r"\b(document|documents|requirements?|kaghzat|dastaweez)\b|دستاویز|کاغذات", re.I)
_SERVICE = re.compile(r"\b(passport|renew(?:al)?|new|first[- ]time|lost|damaged|urgent|normal|fast[- ]track|epassport|e[- ]passport|mrp|delivery|modif(?:y|ication)|visa)\b|پاسپورٹ|تجدید|گم", re.I)
_APPLICANT = re.compile(r"\b(adult|minor|child|children|under\s*18|above\s*18|government employee|dual[- ]national|pages?|36|72|100|urgent|normal|overseas|abroad)\b|بالغ|نابالغ|بچہ", re.I)
_OUT_OF_DOMAIN = re.compile(r"\b(weather|recipe|football|cricket|world cup|match|election|stock price|write (?:me )?a poem|capital of|solve this|homework)\b", re.I)
_NUMBERS = re.compile(r"(?<!\w)(?:Rs\.?\s*)?([0-9][0-9,]*(?:\.[0-9]+)?)(?:\s*(?:PKR|rupees?|Rs\.?))?", re.I)


def _lang(lang):
    return "ur" if str(lang).lower().startswith(("ur", "roman", "mixed")) else "en"


def _clean_query(query):
    # Prompt-injection text is discarded; it is never passed to retrieval or followed.
    return re.sub(r"\s+", " ", _INJECTION.sub(" ", str(query or ""))).strip()


_SPECIFIC = re.compile(r"\b(renew(?:al)?|new|first[- ]time|lost|damaged|urgent|normal|fast[- ]track|e[- ]?passport|mrp|modif(?:y|ication)|tajdeed|gum|online|name change)\b|تجدید|گم|نیا|آن لائن|ارجنٹ|نارمل", re.I)
_DOMAIN = re.compile(r"pas+a?port|cnic|nicop|dgip|immigration|renew|tajdeed|biometric|\bfees?\b|document|dastaweez|پاسپورٹ|شناختی|تجدید|دستاویز|فیس", re.I)


def _needs_clarification(query):
    words = re.findall(r"[\w\u0600-\u06ff]+", query, flags=re.UNICODE)
    fee, docs = bool(_FEE.search(query)), bool(_DOCS.search(query))
    if len(words) < MIN_QUERY_WORDS:
        return fee or docs or bool(_SERVICE.search(query))
    if fee:
        return not (_SPECIFIC.search(query) or _APPLICANT.search(query))
    if docs:
        return not _SERVICE.search(query)
    return False


def _sentences(text):
    # Preserve Urdu and Latin punctuation while keeping each cited extract concise.
    return [s.strip() for s in re.split(r"(?<=[.!?؟۔])\s+|\n+", str(text)) if s.strip()]


def _half(text, lang):
    if "||" not in text:
        return text
    left, right = text.split("||", 1)
    return right.strip() if lang == "ur" else left.strip()


def _terms(text):
    return set(re.findall(r"[\w\u0600-\u06ff]+", normalize(text)))


def _rank_sentences(query, passage_text, lang):
    query_terms = _terms(query)
    items = []
    for position, sentence in enumerate(_sentences(_half(passage_text, lang))):
        terms = _terms(sentence)
        overlap = len(query_terms & terms) / max(1, len(query_terms))
        # Prefer query-relevant, informative sentences; original order breaks ties.
        information = min(len(terms), 30) / 30
        score = overlap + 0.04 * information - position * 1e-5
        if terms and score > 0.02:
            items.append((score, position, sentence))
    return sorted(items, key=lambda item: (-item[0], item[1]))


def _fee_numbers(text):
    vals = set()
    for m in _NUMBERS.finditer(text):
        raw = m.group(0).lower().replace(",", "").replace(" ", "")
        # Ignore years, page counts, and telephone-like numbers.
        if re.search(r"\b(?:19|20)\d{2}\b", raw) or len(re.sub(r"\D", "", raw)) >= 8:
            continue
        if "rs" in raw or "pkr" in raw or "rupee" in raw:
            vals.add(re.sub(r"[^0-9.]", "", raw))
    return vals


def _evidence(retriever, ranked):
    out = []
    for idx, score in ranked:
        if not isinstance(idx, int) or idx < 0 or idx >= len(retriever.passages):
            continue
        p = retriever.passages[idx]
        out.append({
            "passage_id": p.get("passage_id", str(idx)),
            "doc_id": p.get("doc_id", ""),
            "url": p.get("url", ""),
            "score": float(score),
            "text": p.get("text", ""),
        })
    return out


def answer(query, retriever, mode="hybrid", lang="en"):
    """Return a policy decision and cited evidence for a user query."""
    language = _lang(lang)
    t = TEMPLATES[language]
    safe_query = _clean_query(query)
    if not safe_query:
        return {"status": "clarify", "text": t["clarify"], "evidence": []}

    if _CURRENT.search(safe_query):
        results, sem_score = retriever.hybrid(safe_query, k=MAX_EVIDENCE, return_sem=True)
        ev = _evidence(retriever, results)
        dates = [p.get("date_accessed") for p in retriever.passages
                 if p.get("passage_id") in {e["passage_id"] for e in ev} and p.get("date_accessed")]
        date = max(dates) if dates else "unknown date"
        return {"status": "abstain", "text": t["current"].format(date=date), "evidence": ev}

    if _needs_clarification(safe_query):
        return {"status": "clarify", "text": t["clarify"], "evidence": []}

    if _OUTCOME.search(safe_query) or _GUARANTEE.search(safe_query) or _OUT_OF_DOMAIN.search(safe_query):
        return {"status": "abstain", "text": t["abstain"], "evidence": []}

    prepared = prepare_query(safe_query)
    if len(_terms(safe_query)) < MIN_QUERY_WORDS:
        return {"status": "abstain", "text": t["no_evidence"], "evidence": []}
    if mode == "hybrid":
        results, semantic_score = retriever.hybrid(safe_query, k=MAX_EVIDENCE, return_sem=True)
        bm25 = retriever.bm25(safe_query, k=1)
        bm25_score = bm25[0][1] if bm25 else 0.0
    elif mode in ("bm25", "char", "semantic"):
        results = retriever.search(mode, safe_query, k=MAX_EVIDENCE)
        sem = retriever.semantic(safe_query, k=1)
        semantic_score = sem[0][1] if sem else 0.0
        bm = retriever.bm25(safe_query, k=1)
        bm25_score = bm[0][1] if bm else 0.0
    else:
        raise ValueError("mode must be hybrid, bm25, char, or semantic")

    ev = _evidence(retriever, results)
    off_domain = not _DOMAIN.search(safe_query) and semantic_score < DOMAIN_SEM
    if semantic_score < THRESH or bm25_score <= BM25_MIN or off_domain or not ev:
        return {"status": "abstain", "text": t["no_evidence"], "evidence": ev}

    query_for_rank = prepared["query"]
    chosen = []
    for e in ev:
        ranked = _rank_sentences(query_for_rank, e["text"], language)
        for sentence_score, _, sentence in ranked:
            chosen.append((sentence_score, e, sentence))
    chosen.sort(key=lambda item: item[0], reverse=True)
    chosen = chosen[:MAX_SENTENCES]
    if not chosen:
        return {"status": "abstain", "text": t["no_evidence"], "evidence": ev}

    citations, lines = [], []
    for _, e, sentence in chosen:
        cite = f"[{e['doc_id']}|{e['passage_id']}|{e['url']}]"
        lines.append(f"{sentence} {cite}")
        citations.append((e, sentence))
    text = " ".join(lines)
    if _FEE.search(safe_query):
        amounts = set()
        for e, _ in citations:
            amounts.update(_fee_numbers(e["text"]))
        if len(amounts) > 1:
            text += " " + t["conflict"]
    return {"status": "answered", "text": text, "evidence": ev}

import inspect as _i
_answer_orig = answer
_params = _i.signature(_answer_orig).parameters

def answer(*args, method=None, **kw):
    if method is not None:
        for n in ("mode", "retriever_mode", "retrieval_mode", "mode_name"):
            if n in _params:
                kw[n] = method
                break
    return _answer_orig(*args, **kw)