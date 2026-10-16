# src/retrievers.py
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from src.normalize import normalize as norm_text
from src.normalize import prepare_query, set_vocab

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "multilingual-e5-small"
EMB_PATH = ROOT / "data" / "emb.npy"
EMB_PATH_RAW = ROOT / "data" / "emb_raw.npy"


@lru_cache(maxsize=1)
def load_model(path=str(MODEL_PATH)):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(path, device="cpu")


def _tok(s):
    return re.findall(r"\w+", s.lower())


def _top(scores, k, positive_only=False):
    scores = np.asarray(scores, dtype=float)
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(int(i), float(scores[i])) for i in idx if not positive_only or scores[i] > 0]


class Retriever:
    def __init__(self, passages_path="data/passages.json", normalize=True, use_cache=True):
        p = Path(passages_path)
        self.path = p if p.is_absolute() else ROOT / p
        self.passages = json.loads(self.path.read_text(encoding="utf-8"))
        self.normalize = normalize
        self.use_cache = use_cache
        raw = [p["text"] for p in self.passages]
        self.docs = [norm_text(t) for t in raw] if normalize else [t.lower() for t in raw]
        if normalize:
            set_vocab(raw)
        self.bm25_index = BM25Okapi([_tok(d) for d in self.docs])
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.tfidf_matrix = self.tfidf.fit_transform(self.docs)
        self._emb = None

    def _q(self, q):
        if not self.normalize:
            return q.lower(), q.lower()
        pq = prepare_query(q)
        return pq["query"], pq["roman_normalized"]

    def bm25(self, q, k=5):
        full, _ = self._q(q)
        toks = _tok(full)
        if not toks:
            return []
        return _top(self.bm25_index.get_scores(toks), k, positive_only=True)

    def char(self, q, k=5):
        _, plain = self._q(q)
        if not plain.strip():
            return []
        v = self.tfidf.transform([plain])
        return _top((self.tfidf_matrix @ v.T).toarray().ravel(), k, positive_only=True)

    def _embeddings(self):
        if self._emb is not None:
            return self._emb
        cache = EMB_PATH if self.normalize else EMB_PATH_RAW
        meta = cache.with_suffix(".meta")
        h = hashlib.sha1(("\n".join(self.docs) + str(MODEL_PATH.name)).encode("utf-8")).hexdigest()
        if self.use_cache and cache.exists() and meta.exists() and meta.read_text() == h:
            emb = np.load(cache)
            if emb.shape[0] == len(self.docs):
                self._emb = emb
                return emb
        model = load_model()
        emb = model.encode(["passage: " + d for d in self.docs], batch_size=16,
                           normalize_embeddings=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        if self.use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, emb)
            meta.write_text(h)
        self._emb = emb
        return emb

    def semantic(self, q, k=5):
        _, plain = self._q(q)
        if not plain.strip():
            return []
        emb = self._embeddings()
        qv = load_model().encode(["query: " + plain], normalize_embeddings=True, show_progress_bar=False)[0]
        return _top(emb @ qv.astype(np.float32), k)

    def hybrid(self, q, k=5, w=(1, 1, 1), rrf_k=60, return_sem=True):
        """Fuse BM25, character TF-IDF, and semantic rankings with weighted RRF.

        Returns ``(ranked_results, top_semantic_cosine)`` by default. Set
        ``return_sem=False`` for the legacy ranked-results-only shape.
        """
        if k < 1:
            return ([], 0.0) if return_sem else []
        if len(w) != 3 or any(float(weight) < 0 for weight in w):
            raise ValueError("w must contain three non-negative weights")
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        fused, top_sem = {}, 0.0
        for wt, name in zip(w, ("bm25", "char", "semantic")):
            if not wt:
                continue
            res = getattr(self, name)(q, k=max(k, 20))
            if name == "semantic" and res:
                top_sem = float(res[0][1])
            for rank, (i, _) in enumerate(res, start=1):
                fused[i] = fused.get(i, 0.0) + float(wt) / (rrf_k + rank)
        if not w[2]:
            semantic = self.semantic(q, k=1)
            top_sem = float(semantic[0][1]) if semantic else 0.0
        out = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k]
        return (out, top_sem) if return_sem else out

    def search(self, *args, mode=None, method=None, k=5, normalize=None):
        new = mode is not None or method is not None
        if new:
            m, q = (mode or method), args[0]
        else:
            m, q = args[0], args[1]
        if normalize is not None and normalize != self.normalize:
            alts = self.__dict__.setdefault("_alts", {})
            if normalize not in alts:
                alts[normalize] = Retriever(normalize=normalize)
            return alts[normalize].search(q, mode=m, k=k)
        if m == "hybrid":
            r = self.hybrid(q, k=k, return_sem=not new)
            return r[0] if (new and isinstance(r, tuple)) else r
        return getattr(self, m)(q, k=k)
