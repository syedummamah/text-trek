# src/evaluate.py
import re
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import answer as answer_mod
from src.normalize import normalize
from src.retrievers import Retriever

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "test_queries.csv"
OUT = ROOT / "results"
MODES = ["bm25", "char", "semantic", "hybrid"]
MODE_ARG = {m: m for m in MODES}  # edit if Retriever.search uses other mode names
K = 5

# Assumed: Retriever.search(query, mode=, k=, normalize=) -> list of dicts/objects
# with passage_id, doc_id, score, text. If it lacks `normalize=`, the query is
# normalized here instead (and the retriever is assumed not to normalize itself).
clarify_fn = getattr(answer_mod, "needs_clarification", None) or getattr(answer_mod, "is_ambiguous", None)


def g(obj, *names, default=None):
    for n in names:
        if isinstance(obj, dict) and obj.get(n) is not None:
            return obj[n]
        if not isinstance(obj, dict) and getattr(obj, n, None) is not None:
            return getattr(obj, n)
    return default


def search(ret, q, mode, norm=True):
    try:
        res = ret.search(q, mode=MODE_ARG[mode], k=K, normalize=norm)
    except TypeError:
        res = ret.search(normalize(q) if norm else q, mode=MODE_ARG[mode], k=K)
    return [
        {
            "passage_id": str(g(r, "passage_id")),
            "doc_id": g(r, "doc_id", default=""),
            "score": float(g(r, "score", default=0.0)),
            "text": str(g(r, "text", default="")),
        }
        for r in list(res)[:K]
    ]


def gold_set(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return set()
    return {s.strip() for s in re.split(r"[|;,]", str(v)) if s.strip()}


def rank_of(ids, gold):
    for i, p in enumerate(ids, 1):
        if p in gold:
            return i
    return None


def decide(q, hits, thresh):
    try:
        if clarify_fn and clarify_fn(q):
            return "clarify"
    except Exception:
        pass
    if not hits or hits[0]["score"] < thresh:
        return "abstain"
    return "answer"


def abst_ok(status, answerable):
    return (status == "answer") == bool(answerable)


def best_thresh(rows):
    tops = sorted({h[0]["score"] for _, h, _ in rows if h})
    if not tops:
        return 0.0, np.nan
    cands = [tops[0] - 1e-9] + [(a + b) / 2 for a, b in zip(tops, tops[1:])] + [tops[-1] + 1e-9]
    best = None
    for t in cands:
        acc = float(np.mean([abst_ok(decide(q, h, t), a) for q, h, a in rows]))
        if best is None or acc > best[1] + 1e-12:
            best = (t, acc)
    return best


def md(df):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "n/a" if np.isnan(v) else f"{v:.3f}"
        return str(v).replace("|", "/").replace("\n", " ")

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def retrieval_scores(sub):
    r = sub["rank"]
    return (r.notna().mean(), r.map(lambda x: 1 / x if pd.notna(x) else 0.0).mean())


def mode_metrics(sub, mode, thresh):
    ret = sub[(sub.answerable == 1) & sub.has_gold]
    rec, mrr = retrieval_scores(ret) if len(ret) else (np.nan, np.nan)
    ans = ret[ret.status == "answer"]
    cite = ans["top1_ok"].mean() if len(ans) else np.nan
    cons, n_groups = np.nan, 0
    if len(ret):
        vals = []
        for _, grp in ret.dropna(subset=["group_id"]).groupby("group_id"):
            if grp["language"].nunique() >= 2:
                vals.append(grp["rank"].notna().mean())
        n_groups = len(vals)
        cons = float(np.mean(vals)) if vals else np.nan
    return {
        "section": "mode",
        "mode": mode,
        "thresh": thresh,
        "recall@5": rec,
        "mrr@5": mrr,
        "citation_support": cite,
        "abstention_acc": sub["abst_correct"].mean(),
        "cross_script_consistency": cons,
        "latency_mean_ms": sub["latency_ms"].mean(),
        "latency_p95_ms": np.percentile(sub["latency_ms"], 95),
        "n_queries": len(sub),
        "n_retrieval": len(ret),
        "n_answered": len(ans),
        "n_unanswerable": int((sub.answerable == 0).sum()),
        "n_groups": n_groups,
    }


def fmt_top(hits, n=3):
    return "; ".join(f"{h['passage_id']} ({h['score']:.3f}): {h['text'][:60]}" for h in hits[:n])


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    df["answerable"] = df["answerable"].astype(int)
    df["gold"] = df["expected_passage_id"].map(gold_set)
    if "test" not in set(df["split"]):
        raise SystemExit("No split=test rows in data/test_queries.csv")

    ret = Retriever()
    for m in MODES:
        search(ret, df["query"].iloc[0], m)

    raw = {}
    for r in df.itertuples():
        for m in MODES:
            t0 = time.perf_counter()
            hits = search(ret, r.query, m)
            raw[(r.query_id, m)] = (hits, (time.perf_counter() - t0) * 1000)

    dev = df[df.split == "dev"]
    thresh, sweep = {}, []
    for m in MODES:
        rows = [(r.query, raw[(r.query_id, m)][0], r.answerable) for r in dev.itertuples()]
        if not rows:
            print(f"WARNING: no dev rows; THRESH=0 for {m}")
            thresh[m] = 0.0
            continue
        t, acc = best_thresh(rows)
        thresh[m] = t
        sweep.append({"mode": m, "best_THRESH": t, "dev_abstention_acc": acc, "n_dev": len(rows)})
    if sweep:
        print("\n## Dev threshold sweep (best per mode)\n")
        print(md(pd.DataFrame(sweep)))

    recs = []
    for r in df.itertuples():
        for m in MODES:
            hits, ms = raw[(r.query_id, m)]
            ids = [h["passage_id"] for h in hits]
            status = decide(r.query, hits, thresh[m])
            recs.append(
                {
                    "query_id": r.query_id,
                    "split": r.split,
                    "language": r.language,
                    "group_id": r.group_id,
                    "answerable": r.answerable,
                    "expected_passage_id": r.expected_passage_id,
                    "mode": m,
                    "has_gold": bool(r.gold),
                    "rank": rank_of(ids, r.gold) if r.gold else np.nan,
                    "top1_id": ids[0] if ids else "",
                    "top1_score": hits[0]["score"] if hits else np.nan,
                    "top1_ok": bool(ids and ids[0] in r.gold),
                    "status": status,
                    "abst_correct": abst_ok(status, r.answerable),
                    "latency_ms": ms,
                    "top5_ids": "|".join(ids),
                }
            )
    pq = pd.DataFrame(recs)

    test = pq[pq.split == "test"]
    rows = [mode_metrics(test[test["mode"] == m], m, thresh[m]) for m in MODES]

    tq = df[df.split == "test"]
    roman = tq[(tq.language == "roman") & (tq.answerable == 1) & tq.gold.map(bool)]
    abl = []
    if len(roman):
        raw_rank = []
        for r in roman.itertuples():
            hits = search(ret, r.query, "hybrid", norm=False)
            raw_rank.append(rank_of([h["passage_id"] for h in hits], r.gold))
        base = test[(test["mode"] == "hybrid") & test.query_id.isin(roman.query_id)]
        norm_rec, norm_mrr = retrieval_scores(base)
        s = pd.Series(raw_rank, dtype="float")
        raw_rec, raw_mrr = retrieval_scores(pd.DataFrame({"rank": s}))
        for name, rec, mrr in [("hybrid+norm", norm_rec, norm_mrr), ("hybrid-raw", raw_rec, raw_mrr)]:
            abl.append({"section": "ablation_roman", "mode": name, "recall@5": rec, "mrr@5": mrr, "n_retrieval": len(roman)})
    metrics = pd.concat([pd.DataFrame(rows), pd.DataFrame(abl)], ignore_index=True)

    fails = []
    for r in df.itertuples():
        hits, _ = raw[(r.query_id, "hybrid")]
        p = pq[(pq.query_id == r.query_id) & (pq["mode"] == "hybrid")].iloc[0]
        ftype = None
        if r.answerable == 1 and r.gold:
            if pd.isna(p["rank"]):
                ftype = "retrieval_miss"
            elif p["status"] != "answer":
                ftype = "false_abstain"
            elif p["rank"] > 1:
                ftype = "wrong_top1"
        elif r.answerable == 1 and p["status"] != "answer":
            ftype = "false_abstain"
        elif r.answerable == 0 and p["status"] == "answer":
            ftype = "false_answer"
        if ftype:
            fails.append(
                {
                    "query_id": r.query_id,
                    "split": r.split,
                    "language": r.language,
                    "query": r.query,
                    "answerable": r.answerable,
                    "failure_type": ftype,
                    "expected_passage_id": r.expected_passage_id,
                    "rank": p["rank"],
                    "status": p["status"],
                    "top3_retrieved": fmt_top(hits),
                    "expected_behavior": r.expected_behavior,
                }
            )
    fdf = pd.DataFrame(
        fails,
        columns=["query_id", "split", "language", "query", "answerable", "failure_type", "expected_passage_id", "rank", "status", "top3_retrieved", "expected_behavior"],
    )

    manual = tq[(tq.answerable == 1) & ~tq.gold.map(bool)]
    mrev = pd.DataFrame(
        [
            {
                "query_id": r.query_id,
                "language": r.language,
                "query": r.query,
                "expected_answer_en": r.expected_answer_en,
                "hybrid_top3": fmt_top(raw[(r.query_id, "hybrid")][0]),
            }
            for r in manual.itertuples()
        ]
    )

    pq_out = pq.copy()
    pq_out["rank"] = pq_out["rank"].astype("Int64")
    metrics.to_csv(OUT / "metrics.csv", index=False)
    pq_out.to_csv(OUT / "per_query.csv", index=False)
    fdf.to_csv(OUT / "failures.csv", index=False)
    mrev.to_csv(OUT / "manual_review.csv", index=False)

    cols = ["recall@5", "mrr@5", "citation_support", "abstention_acc", "cross_script_consistency"]
    mm = metrics[metrics.section == "mode"]
    x = np.arange(len(cols))
    w = 0.8 / len(MODES)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(MODES):
        vals = mm[mm["mode"] == m][cols].iloc[0].astype(float).fillna(0).values
        ax.bar(x + i * w, vals, w, label=m)
    ax.set_xticks(x + w * (len(MODES) - 1) / 2)
    ax.set_xticklabels(cols, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title("Test split: retrieval and abstention metrics by retriever")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "chart.png", dpi=150)
    plt.close(fig)

    print(f"\n## Test metrics (split=test, n={len(tq)} queries)\n")
    print(md(mm.drop(columns=["section"])))
    if abl:
        print("\n## Normalization ablation (hybrid, roman, test)\n")
        print(md(pd.DataFrame(abl).drop(columns=["section"])))
    print(f"\n## Hybrid failures: {len(fdf)} (test: {int((fdf.split == 'test').sum())})\n")
    if len(fdf):
        print(md(fdf[fdf.split == "test"][["query_id", "language", "failure_type", "rank", "status"]]))
    print(f"\n## Test answerable rows without expected_passage_id (manual review, n={len(mrev)})\n")
    if len(mrev):
        print(md(mrev[["query_id", "language", "query", "hybrid_top3"]]))
    print(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()