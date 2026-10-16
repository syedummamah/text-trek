# src/normalize.py
import re
import unicodedata

from rapidfuzz import fuzz, process

_ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u200e\u200f\u2060\ufeff"), None)
_DIGITS = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_DIGITS.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})
_CHARMAP = {
    "ي": "ی", "ى": "ی", "ې": "ی", "ێ": "ی",
    "ك": "ک", "ڪ": "ک",
    "ه": "ہ", "ۃ": "ہ", "ة": "ہ", "ۀ": "ہ",
    "ۓ": "ے",
    "أ": "ا", "إ": "ا", "ٱ": "ا",
    "ؤ": "و",
}
_CHARMAP = {ord(k): v for k, v in _CHARMAP.items()}
_AR_MARKS = re.compile("[\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e4\u06e7\u06e8\u06ea-\u06ed\u0640]")
_AR_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))

VARIANTS = {
    "renew": ["renu", "renue", "renw", "renewal", "renewl", "tajdeed", "tajdid", "tajdeet", "tajdeedi", "renval"],
    "hai": ["hy", "hay", "hae", "hai", "hey"],
    "hain": ["hen", "hyn", "hayn", "hain"],
    "kya": ["kia", "kiya", "kyaa", "kya"],
    "documents": ["document", "documnts", "documnt", "docs", "doc", "dastaweez", "dastavez", "dastawez", "dastawaiz", "dastavaiz", "dastaveez", "kaghzat", "kagzat"],
    "chahiye": ["chahiyay", "chahye", "chahyen", "chahyie", "chaiye", "chahie", "chahiyen", "chahiey", "chaahiye", "chahiya"],
    "karwana": ["karwanay", "karwany", "karwane", "karvana", "karwna", "karwani", "karwanai", "karvanay"],
    "karna": ["krna", "karnaa", "karny", "karnay"],
    "passport": ["pasport", "paspot", "passpot", "pasprt", "passprt", "paasport", "pasaport"],
    "fee": ["fees", "fess", "kharcha", "kharch"],
    "lost": ["gum", "gumshuda", "gom", "gumm"],
    "child": ["bachay", "bachy", "bache", "bachey", "bacha", "bachon", "bachhay", "bachha", "bachcha", "children", "bachay"],
    "how much": ["kitni", "kitna", "kitnee", "kitnaa"],
    "how many": ["kitne", "kitnay", "kitny", "kitnai"],
    "sakta": ["skta", "sakte", "sakta", "saktaa", "sktay", "saktay"],
    "sakti": ["skti", "saktee", "sakti"],
    "kon": ["kaun", "kon", "koun", "kaon"],
    "zaroori": ["zarori", "zaruri", "zaroori", "lazmi", "lazim", "zarur", "zaroorat"],
    "saal": ["sal", "saal", "saaal", "saall"],
    "din": ["din", "dino", "dinon", "diin"],
    "meyad": ["miyad", "myad", "muddat", "mudat", "meeyad", "meyad", "myaad"],
    "wafat": ["wafaat", "faut", "wafat", "inteqal", "intiqal"],
    "talaq": ["talak", "talaq", "divorced", "divorce"],
    "tabdeel": ["tabdil", "tabdeeli", "tabdeel", "badalna", "badal"],
    "naam": ["nam", "naam", "name"],
    "shanakhti": ["shanakhti", "shanakhati", "shinakhti"],
    "dohri": ["dohri", "dohra", "dohry", "dual"],
    "shehriyat": ["shehriyat", "shahriyat", "shehryat", "shahriat", "nationality"],
    "sarkari": ["sarkari", "sarkary", "govt", "gov", "government"],
    "mulazim": ["mulazim", "mulazam", "employe", "employee", "employees"],
    "ijazat": ["ijazat", "ijazt", "ijaazat", "permission"],
    "wapis": ["wapis", "wapas", "refund", "refundable"],
    "darkhwast": ["darkhwast", "darkhast", "darkhuwast", "application"],
    "tarika": ["tarika", "tareeqa", "tariqa", "tareqa", "tarikay"],
    "ghar": ["ghar", "ghr"],
    "pohanch": ["pohanch", "pohnch", "pohanchay", "pohanchna", "pohonchay", "pohanchega", "pohanchay"],
    "tasveer": ["tasveer", "tasweer", "tasvir", "tasvir"],
    "safhat": ["safhat", "safhaat", "safhe", "pages"],
    "khatam": ["khatam", "khtm", "khatm", "exhausted"],
    "purana": ["purana", "purani", "purane", "puranay"],
    "naya": ["naya", "nayi", "nayee", "nayay"],
    "fori": ["fori", "foree", "foori"],
    "jaldi": ["jaldi", "jldi", "jaldy"],
    "ungli": ["ungli", "unglian", "fingerprints", "fingerprint"],
    "kab": ["kab", "kub"],
    "kahan": ["kahan", "kaha", "kahin", "kidhar"],
    "kaise": ["kaise", "kesay", "kese", "kaisay", "kesey"],
    "shuru": ["shuru", "shuro", "shuru"],
    "liye": ["liye", "liyay", "lye", "liay", "lie"],
    "nahi": ["nahi", "nahin", "nai", "nhi", "nahe"],
    "mein": ["mein", "mai", "mayn", "me"],
    "aur": ["aur", "or", "ur"],
}

ROMAN_TO_EN = {
    "renew": ["renewal", "renew", "expired", "expiry"],
    "documents": ["documents", "required", "requirements"],
    "chahiye": ["required", "requirements", "bring"],
    "zaroori": ["required", "mandatory"],
    "fee": ["fee", "charges", "payment"],
    "how much": ["fee", "charges"],
    "lost": ["lost", "police report", "previous passport number"],
    "child": ["minor", "child", "CRC", "Form-B", "parents"],
    "shanakhti": ["CNIC", "identity card"],
    "wafat": ["deceased", "death certificate"],
    "talaq": ["divorced", "separated", "custody", "court"],
    "tabdeel": ["change", "modify", "amend"],
    "naam": ["name", "CNIC"],
    "saal": ["years", "validity"],
    "din": ["days", "working days"],
    "meyad": ["validity", "expiry"],
    "dohri": ["dual nationality", "foreign passport"],
    "shehriyat": ["nationality", "citizenship"],
    "sarkari": ["government", "employee", "NOC"],
    "mulazim": ["employee", "government", "NOC"],
    "ijazat": ["permission", "consent"],
    "wapis": ["refund", "refundable"],
    "darkhwast": ["application", "apply"],
    "tarika": ["procedure", "process", "steps"],
    "ghar": ["delivery", "courier"],
    "pohanch": ["delivery", "courier"],
    "tasveer": ["photograph"],
    "safhat": ["pages", "36 pages", "72 pages", "100 pages"],
    "khatam": ["exhausted", "expired"],
    "purana": ["previous", "old", "expired"],
    "naya": ["new", "fresh"],
    "fori": ["urgent"],
    "jaldi": ["urgent", "fast track"],
    "ungli": ["fingerprints", "biometrics"],
    "kab": ["when", "time"],
    "kahan": ["where", "office"],
    "kaise": ["how", "process"],
    "shuru": ["start", "process"],
}

_RU_STOP = {
    "ke", "ki", "ka", "ko", "kay", "se", "ya", "mera", "meri", "mere", "mujhay", "mujhe", "agar", "to", "ho",
    "hoga", "hota", "hoti", "hotay", "jaye", "jayen", "karo", "karein", "karen", "bas", "koi", "bhi", "hi",
    "wala", "walay", "walay", "ab", "tak", "par", "pe", "ne", "yeh", "ye", "woh", "wo", "apna", "apnay",
    "apne", "kisi", "konsa", "konsi", "kon", "kar", "kal", "aaj", "pehla", "pehlay", "baad", "sath", "saath",
    "chotay", "baray", "zyada", "kam", "gaya", "gya", "gaye", "gayi", "lana", "lanay", "lani", "dena", "lena",
}

_VOCAB_LIST = []
_VOCAB = set()


def _collapse(s):
    return re.sub(r"(.)\1+", r"\1", s)


_VARIANT_MAP = {}
for _canon, _vs in VARIANTS.items():
    for _v in _vs + [_canon]:
        if " " not in _v:
            _VARIANT_MAP.setdefault(_collapse(_v), _canon)
for _canon in VARIANTS:
    _VARIANT_MAP[_collapse(_canon)] = _canon if " " not in _canon else _VARIANT_MAP.get(_collapse(_canon), _canon)

_PROTECTED = set(_RU_STOP)
for _c in VARIANTS:
    _PROTECTED.update(_c.split())
_RU_MARKERS = set(_RU_STOP) | {w for w in _VARIANT_MAP if w not in {"or", "me", "gov", "doc", "dual", "name", "pages", "ur", "lie"}}
_RU_MARKERS |= set(VARIANTS) - {"passport", "renew", "documents", "fee", "lost", "child", "naam"}
_RU_MARKERS |= {"mein", "mai", "main_", "liye", "lye", "kya", "kia", "hai", "hy", "hain", "nahi", "karna", "aur"}
_RU_MARKERS -= {"name", "pages", "dual", "government", "refund", "application", "permission", "fingerprints"}


def normalize(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.translate(_ZW).translate(_DIGITS).translate(_CHARMAP)
    text = _AR_MARKS.sub("", text)
    out = []
    for ch in text:
        o = ord(ch)
        if o < 0x250 and unicodedata.category(ch).startswith("L"):
            ch = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        cat = unicodedata.category(ch[0]) if ch else "Zs"
        if ch in "'’`":
            continue
        out.append(" " if cat[0] in "PSC" and ch not in "\n\t" else ch)
    text = "".join(out).lower()
    return re.sub(r"\s+", " ", text).strip()


def set_vocab(texts):
    global _VOCAB, _VOCAB_LIST
    toks = set()
    for t in texts:
        toks.update(re.findall(r"[a-z]{4,}", normalize(t)))
    _VOCAB = toks
    _VOCAB_LIST = sorted(toks)


def _snap(tok, vocab, vlist):
    if len(tok) < 4 or not tok.isascii() or not tok.isalpha() or tok in vocab or tok in _PROTECTED:
        return tok
    m = process.extractOne(tok, vlist, scorer=fuzz.ratio, score_cutoff=85)
    return m[0] if m else tok


def roman_normalize(text, vocab=None):
    text = normalize(text)
    if vocab is None:
        v, vl = _VOCAB, _VOCAB_LIST
    else:
        v = set(vocab)
        vl = sorted(v)
    out = []
    for tok in text.split():
        if not tok.isascii() or tok.isdigit():
            out.append(tok)
            continue
        if tok in v and tok not in _VARIANT_MAP and not any(tok == k for k in VARIANTS):
            out.append(tok)
            continue
        canon = _VARIANT_MAP.get(_collapse(tok))
        if canon:
            out.append(canon)
            continue
        out.append(_snap(tok, v, vl) if vl else tok)
    return " ".join(out)


def detect_script(text):
    t = normalize(text)
    ar = sum(1 for ch in t if any(a <= ord(ch) <= b for a, b in _AR_RANGES))
    la = sum(1 for ch in t if "a" <= ch <= "z")
    if ar and la:
        return "urdu" if la / (la + ar) < 0.15 else "mixed"
    if ar:
        return "urdu"
    if not la:
        return "english"
    toks = re.findall(r"[a-z]+", t)
    hits = sum(1 for w in toks if w in _RU_MARKERS or _collapse(w) in _RU_MARKERS)
    return "roman" if toks and hits / len(toks) >= 0.2 else "english"


def expand_roman(text):
    toks = roman_normalize(text).split()
    seen, extra = set(toks), []
    keys = list(toks) + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    for k in keys:
        for w in ROMAN_TO_EN.get(k, []):
            if w.lower() not in seen:
                seen.add(w.lower())
                extra.append(w)
    return " ".join(extra)


def prepare_query(text, vocab=None):
    script = detect_script(text)
    norm = normalize(text)
    rn = roman_normalize(text, vocab) if script in ("roman", "mixed", "english") else norm
    exp = expand_roman(text) if script in ("roman", "mixed") else ""
    return {
        "script": script,
        "normalized": norm,
        "roman_normalized": rn,
        "expansion": exp,
        "query": f"{rn} {exp}".strip(),
    }